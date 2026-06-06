import os, math
from typing import Optional, Tuple, Union, Literal
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torch.nn.functional as F
torch.set_printoptions(profile="full")

class GELUNew(nn.Module):
    def forward(self, x):
        return F.gelu(x, approximate="tanh")

def get_activation_module(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu": return nn.GELU()
    if name == "gelu_new": return GELUNew()
    if name in {"silu", "swish"}: return nn.SiLU()
    if name == "relu": return nn.ReLU()
    if name == "tanh": return nn.Tanh()
    raise ValueError(f"Unknown activation function: {name}")

PE_TYPE = Literal[
    "none",
    "absolute", 
    "sinusoidal",
    "rotary",
    "scaler",
    "alibi",
    "relative",
]

class RotaryPE(nn.Module):
    def __init__(self, head_dim, max_position_embeddings, base=10000, scaling_factor=1.0):
        super().__init__()
        
        if head_dim % 2 != 0: raise ValueError(f"Rotary head_dim must be even, got {head_dim}.")
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.scaling_factor = scaling_factor
        self.base = base
        
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, q, k, position_ids, device_type, dtype):
        """
            q, k: [batch, heads, seq_len, head_dim]
            position_ids: [batch, seq_len]
            returns: cos, sin: [batch, seq_len, head_dim]
            
            head_dim is inferred from the size of self.inv_freq.
            frequencies are computed in float32, then cast back to q/k dtype.
        """
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        
        # Force float32 computation for RoPE frequencies.
        with torch.autocast(device_type=device_type, enabled=False):
            pos = position_ids.float() / self.scaling_factor
            inv_freq = self.inv_freq.float()
            # outer product
            freqs = torch.einsum("bs,d->bsd", pos, inv_freq)
            
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos().to(dtype=dtype)
            sin = emb.sin().to(dtype=dtype)

        cos = cos[:, None, :, :].to(q.device)
        sin = sin[:, None, :, :].to(q.device)
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)

        return q, k
    
    @staticmethod
    def _rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class SinusoidalPE(nn.Module):
    def __init__(self, hidden_size, max_position_embeddings):
        super().__init__()
        lookup = torch.zeros(max_position_embeddings, hidden_size)
        position = torch.arange(
            0, 
            max_position_embeddings,
            dtype=torch.float,
        ).unsqueeze(1)

        div_term = torch.exp(
            (torch.arange(0, hidden_size, 2, dtype=torch.float)
            * -(math.log(10000.0) / hidden_size))
        )
        
        lookup[:, 0::2] = torch.sin(position * div_term)
        lookup[:, 1::2] = torch.cos(position * div_term[: lookup[:, 1::2].shape[1]]) # handles odd hidden_size
        self.register_buffer('lookup', lookup, persistent=False)

    def forward(self, x):
        return self.lookup[x] # x is already in index tensor


class ALiBiPE(nn.Module):
    def __init__(self, num_attn_heads):
        super().__init__()

        slopes = self._get_alibi_slopes(num_attn_heads)
        self.register_buffer("slopes", slopes, persistent=False)

    def forward(self, position_ids, dtype):
        """
            position_ids: [batch, seq_len]
            returns: [batch, num_heads, seq_len, seq_len]
        """
        q_pos = position_ids[:, :, None]
        k_pos = position_ids[:, None, :]
        distance = (q_pos - k_pos).abs().float()[:, None, :, :] # [batch, 1, seq_len, seq_len]
        slopes = self.slopes[None, :, None, None].to(device=position_ids.device) # [1, num_heads, 1, 1]
        bias = -slopes * distance # [batch, num_heads, seq_len, seq_len]
        return bias.to(dtype=dtype)

    @staticmethod
    def _get_alibi_slopes(num_attn_heads):
        """
            paper: https://arxiv.org/pdf/2108.12409
            Static, non-learned bias: softmax(qK^T + m x [-(i-1), ..., -2, -1, 0]).
            m is a head-specific slope. 
            To extend it to non-causal, bias[head_idx, i, j] = - m_{head_idx} x |i-j|.

            In general, for n heads, our set of slopes is the geometric sequence that starts
            at 2^{-8/n} and uses that same value as its ratio.
        """
        # get slopes
        ratio = 2 ** (-8 / num_attn_heads)
        slopes = np.array([ratio ** (i + 1) for i in range(num_attn_heads)])
        return torch.from_numpy(slopes).float()

class T5RelativePE(nn.Module):
    """
        Each relative distance bucket ges a **learned**, **scalar** bias per head.
        It modifies attention logits as:
            scores[b, h, i, j] += bias[h, bucket(position_j - position_i)]
        Unlike Shaw/Huang/Transformer-XL relative PE, this does NOT depend on q_i
            and therefore can be cleanly implemented as an additive attention bias.
    """
    def __init__(
        self,
        num_attn_heads,
        num_buckets,
        max_distance,
        causal: bool,
    ):
        super().__init__()

        self.num_attn_heads = num_attn_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.causal = causal

        self.lookup = nn.Embedding(num_buckets, num_attn_heads)

    def forward(self, position_ids, dtype):
        """
            position_ids: [batch, seq_len]
            returns: [batch, num_heads, seq_len, seq_len]
        """
        q_pos = position_ids[:, :, None]
        k_pos = position_ids[:, None, :]
        distance = k_pos - q_pos # [batch, seq_len, seq_len]

        # bucket the relative distances
        buckets = self._get_relative_position_buckets(
            distance,
            causal=self.causal,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )

        # get bias from embedding
        bias = self.lookup(buckets) # [batch, seq_len, seq_len, num_heads]
        bias = bias.permute(0, 3, 1, 2) # [batch, num_heads, seq_len, seq_len]

        return bias.to(dtype=dtype)

    @staticmethod
    def _get_relative_position_buckets(
        distance, 
        causal, 
        num_buckets, # B
        max_distance, # D
    ):
        """
        For causal=True:
            future positions are mapped as if they are distance 0, since they
            should already be removed by the causal mask. Past positions are
            bucketed by distance.
        For causal=False:
            half the buckets are used for negative relative positions and half
            for positive relative positions.
        """
        result = torch.zeros_like(distance, dtype=torch.long)

        if causal:
            # distance = j-i <= 0
            distance = torch.clamp(distance, max=0) # future positions are mapped as if they are distance 0
            distance = -distance
        else:
            num_buckets //= 2
            result += (distance > 0).long() * num_buckets # future positions get a head start of B//2
            distance = distance.abs()
        
        # Now distance is non-negative.
        max_exact = num_buckets // 2 # M
        is_small = distance < max_exact

        # large distances are bucketed logarithmically
        distance_if_large = max_exact + (
            torch.log(distance.float() / max_exact + 1e-6)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).long()
        distance_if_large = torch.clamp(distance_if_large, max=num_buckets-1)
        
        result += torch.where(is_small, distance.long(), distance_if_large)

        return result

    
class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pe_type = config.pe_type
        self.hidden_size = config.hidden_size
        self.num_attn_heads = config.num_attn_heads
        self.head_dim = self.hidden_size // self.num_attn_heads

        if self.pe_type not in {"none", "absolute", "sinusoidal", "rotary", "scaler", "alibi", "relative"}:
            raise ValueError(f"Unknown pe_type: {self.pe_type}. \nMust be one of 'none', 'absolute', 'sinusoidal', 'rotary', 'scaler', 'alibi', or 'relative'.")

        if self.pe_type == "absolute":
            self.absolute = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        elif self.pe_type == "sinusoidal":
            self.sinusoidal = SinusoidalPE(self.hidden_size, config.max_position_embeddings)
        elif self.pe_type == "rotary":
            self.rotary = RotaryPE(self.head_dim, config.max_position_embeddings)
        elif self.pe_type == "scaler":
            self.scaler = nn.Linear(1, self.hidden_size) # Better than concatenating raw position_ids directly.
        elif self.pe_type == "alibi":
            self.alibi = ALiBiPE(config.num_attn_heads)
        elif self.pe_type == "relative":
            self.relative = T5RelativePE(
                num_attn_heads=config.num_attn_heads,
                num_buckets=config.relative_pe_num_buckets,
                max_distance=config.relative_pe_max_distance,
                causal=getattr(config, "causal"),
            )
    
    def apply_to_embeddings(self, x, position_ids):
        if self.pe_type == "absolute":
            return x + self.absolute(position_ids)
        elif self.pe_type == "sinusoidal":
            return x * math.sqrt(self.hidden_size) + self.sinusoidal(position_ids)
        elif self.pe_type == "scaler":
            # normalize position_ids to [0,1]
            y = position_ids.float().unsqueeze(-1) / self.config.max_position_embeddings
            return x + self.scaler(y)
        else:
            return x # for rotary, alibi, and relative, this step is a no-op.

    def apply_to_qk(self, q, k, position_ids):
        if self.pe_type != "rotary": 
            return q, k
        return self.rotary(q, k, position_ids, q.device.type, q.dtype)
    
    def make_position_bias(self, position_ids, dtype):
        if self.pe_type == "alibi":
            return self.alibi(position_ids, dtype)
        elif self.pe_type == "relative":
            return self.relative(position_ids, dtype)
        else:
            return None # for absolute, sinusoidal, rotary, and scaler, there is no separate bias term.

    
class Attention(nn.Module):
    def __init__(
        self,
        config,
        pe: PositionalEncoding,
    ):

        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attn_heads = config.num_attn_heads
        self.head_dim = self.hidden_size // self.num_attn_heads
        
        if self.head_dim * self.num_attn_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_attn_heads (got `hidden_size`: {self.hidden_size} and `num_attn_heads`:"
                f" {self.num_attn_heads})."
            )

        self.pe = pe
        self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj.RESIDUAL_SCALED_INIT = True
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
    
    def merge_heads(self, x):
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, num_heads * head_dim)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_attn_heads, self.head_dim)
        return x.transpose(1, 2)  # (batch, num_attn_heads, seq_length, head_dim)

    def _make_padding_mask(self, binary_mask, device, dtype):
        """
        sequence_collator() should create the padding mask
        This function converts the collator's padding mask into the numerical form that attention needs.
        sequence_collator creates attention_mask: [batch_size, seq_len]
            1 = real token, 0 = pad
        Attention._mask_padding_mask() converts it to additive attention bias: [batch, 1, 1, seq_len]
            0.0 = allowed to attend
            large negative value = forbidden to attend
        """
        binary_mask = binary_mask.to(device=device)

        # Unknown shape
        if binary_mask.dim() != 2:
            raise ValueError(
                "binary_mask should have shape [batch, seq_len] "
                "or additive shape [batch, 1, 1, seq_len]. "
                f"Got shape {tuple(binary_mask.shape)}."
            )
        """
        attention_scores will have shape [batch_size, num_heads, seq_len, seq_len]
        padding mask only depends on the key position, not on head or query position.
        batch_size, 1, 1, seq_len] will be broadcasted to [batch_size, num_heads, seq_len, seq_len]
        """
        result = 1.0 - binary_mask[:, None, None, :].to(dtype=dtype) # adds singleton dimensions for broadcasting
        result = result * self._mask_value(dtype)
        return result
    
    def _make_causal_mask(self, seq_len, device, dtype):
        """
        Returns additive causal attention mask: [1, 1, seq_len, seq_len]
        0.0 = allowed, large negative = masked
        """
        causal_mask = torch.ones(
            seq_len,
            seq_len,
            device=device,
            dtype=torch.bool,
        ).tril()

        result = torch.zeros(
            seq_len,
            seq_len,
            device=device,
            dtype=dtype,
        )

        result = result.masked_fill(
            ~causal_mask,
            self._mask_value(dtype),
        )

        return result[None, None, :, :]
    
    def _mask_value(self, dtype):
        return -1e4 if dtype == torch.float16 else torch.finfo(dtype).min
    
    def _add_bias(self, a, b):
        if a is None:
            return b
        if b is None:
            return a
        return a + b
    
    def _build_attn_mask(
        self,
        binary_mask, # padding mask
        seq_len,
        position_ids,
        device,
        dtype,
    ):
        """ Returns an additive attention bias broadcastable to:
            [batch, num_heads, seq_len, seq_len]

            values: 0.0 -- attention; large negative value -- masked
        """
        result = None

        # A. Padding mask.
        if binary_mask is not None:
            padding_mask = self._make_padding_mask(binary_mask, device, dtype)
            result = self._add_bias(result, padding_mask)

        # B. Causal mask.
        if getattr(self.config, "causal", True):
            causal_mask = self._make_causal_mask(seq_len, device, dtype)
            result = self._add_bias(result, causal_mask)

        # C. Positional attention bias, e.g. ALiBi or T5Relative. Otherwise, no-op.
        position_bias = self.pe.make_position_bias(position_ids=position_ids, dtype=dtype)
        result = self._add_bias(result, position_bias)

        return result
    
    def forward(
        self, 
        hidden_states, 
        binary_mask=None, 
        position_ids=None,
        expose_attn_weights=False,
    ):
        """
        hidden_states: [batch, seq_len, hidden_size]

        binary_mask: [batch, seq_len], where 1 = keep, 0 = pad -> this is the padding mask
            Do NOT pre-convert it into additive -inf form outside this class.
            We will build additive attn_mask that incorporates causal mask and positional bias.
            So eventually, attn_mask = padding_mask + causal_mask + positional_bias

        position_ids: [batch, seq_len]        
        
        q, k, v: [batch, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Project to q/k/v.
        query, key, value = self.qkv_proj(hidden_states).chunk(3, dim=-1)

        # 2. Split heads.
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        # 3. Positional encoding may modify Q/K, e.g. Rotary. Otherwise, no-op.
        query, key = self.pe.apply_to_qk(query, key, position_ids)
        
        # 4. Build attention mask.
        additive_mask = self._build_attn_mask(
            binary_mask=binary_mask,
            seq_len=seq_len,
            position_ids=position_ids,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        # 5. SDPA computes softmax[(QK^T / sqrt(head_dim)) + attn_bias] @ V
        if expose_attn_weights:
            result, attn_weights = self._manual_attn(
                query,
                key,
                value,
                additive_mask=additive_mask,
            )
        else:
            result = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=additive_mask,
                dropout_p=0.0,
                is_causal=False, # we always fold causality into the additive mask
            )

        # 6. Merge heads and project out.
        result = self.merge_heads(result)
        result = self.out_proj(result)
        result = self.resid_dropout(result)

        if expose_attn_weights: 
            return result, attn_weights
        else: 
            return result
    
    def _manual_attn(self, query, key, value, additive_mask=None):
        """
        query, key, value: [batch, heads, seq_len, head_dim]

        additive_mask (already additive, so post self._build_attn_mask): 
            [batch or 1, heads or 1, seq_len, seq_len]

        Returns:
            attn_output:  [batch, heads, seq_len, head_dim]
            attn_weights: [batch, heads, seq_len, seq_len]
        """

        scale = 1.0 / math.sqrt(query.size(-1))

        pre_softmax = torch.matmul(
            query,
            key.transpose(-2, -1),
        ) * scale

        if additive_mask is not None: 
            pre_softmax = pre_softmax + additive_mask
        
        attn_weights = torch.softmax(pre_softmax, dim=-1)
        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights


class Block(nn.Module):
    def __init__(
        self, 
        config,
        pe: PositionalEncoding
    ):
        super().__init__()
        hidden_size = config.hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = Attention(config, pe)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def forward(
        self, 
        x, 
        binary_mask=None, 
        position_ids=None,
        expose_attn_weights=False,
    ):
        if expose_attn_weights:
            attn_output, attn_weights = self.attn(
                self.ln_1(x), 
                binary_mask=binary_mask, 
                position_ids=position_ids, 
                expose_attn_weights=expose_attn_weights
            )
        else:
            attn_output = self.attn(
                self.ln_1(x), 
                binary_mask=binary_mask, 
                position_ids=position_ids, 
                expose_attn_weights=expose_attn_weights
            )
        
        x = x + attn_output # residual connection
        x = x + self.mlp(self.ln_2(x))

        if expose_attn_weights:
            return x, attn_weights
        else:
            return x  


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        hidden_size = config.hidden_size
        mlp_inner_dim = config.mlp_dim_multiplier * hidden_size
        
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mlp_inner_dim),
            get_activation_module(config.activation_function),
            nn.Linear(mlp_inner_dim, hidden_size),
            nn.Dropout(config.resid_pdrop),
        )
        self.net[2].weight.RESIDUAL_SCALED_INIT = True # final linear layer of MLP should also be scaled by 1/sqrt(2L) per GPT-2 paper.

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.hidden_size = config.hidden_size
        self.padding_idx = config.in_vocab.index("<pad>") if "<pad>" in config.in_vocab else None # config.in_vocab is a list. Find the index of "<pad>"
        
        self.we = nn.Embedding(
            num_embeddings=len(config.in_vocab), 
            embedding_dim=self.hidden_size, 
            padding_idx=self.padding_idx
        )
        self.pe = PositionalEncoding(config)
        
        self.drop = nn.Dropout(config.embd_pdrop)
        
        self.blocks = nn.ModuleList([
            Block(config, self.pe) 
            for _ in range(config.num_hidden_layers)
        ])
        self.ln_f = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.hidden_size, len(config.out_vocab), bias=False)

        # Initialize weights
        self.apply(self._init_weights)
        

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            std = self.config.initializer_range

            if getattr(module, "RESIDUAL_SCALED_INIT", False): # default is False
                std = std / math.sqrt(2 * self.config.num_hidden_layers)
            
            module.weight.data.normal_(mean=0.0, std=std)
            
            if module.bias is not None: 
                module.bias.data.zero_()
        
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids,
        position_ids,
        binary_mask=None, 
        expose_attn_weights=False,
    ):

        hidden_states = self.we(input_ids)
        hidden_states = self.pe.apply_to_embeddings(hidden_states, position_ids)
        hidden_states = self.drop(hidden_states)

        # collect per-layer attn_weights
        attn_weights_list = [] # a list of [batch, heads, seq_len, seq_len] tensors
        for block in self.blocks:
            if expose_attn_weights:
                hidden_states, attn_weights = block(
                    hidden_states, 
                    binary_mask=binary_mask, 
                    position_ids=position_ids, 
                    expose_attn_weights=expose_attn_weights
                )
                attn_weights_list.append(attn_weights)
            else:
                hidden_states = block(
                    hidden_states, 
                    binary_mask=binary_mask, 
                    position_ids=position_ids, 
                    expose_attn_weights=expose_attn_weights
                )

        hidden_states = self.ln_f(hidden_states)
        lm_logits = self.lm_head(hidden_states) # loss calculated outside

        if expose_attn_weights:
            result = {'logits': lm_logits, 'attn_weights': attn_weights_list}
        else:
            result = {'logits': lm_logits}
        return result