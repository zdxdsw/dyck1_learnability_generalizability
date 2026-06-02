import os, math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torch.nn.functional as F
from transformers.activations import ACT2FN
torch.set_printoptions(profile="full")

"""
ACT2CLS = {
    "gelu": GELUActivation,
    "gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10}),
    "gelu_fast": FastGELUActivation,
    "gelu_new": NewGELUActivation,
    "gelu_python": (GELUActivation, {"use_gelu_python": True}),
    "gelu_pytorch_tanh": GELUTanh,
    "gelu_python_tanh": (GELUTanh, {"use_gelu_tanh_python": True}),
    "gelu_accurate": AccurateGELUActivation,
    "hardswish": nn.Hardswish,
    "laplace": LaplaceActivation,
    "leaky_relu": nn.LeakyReLU,
    "linear": LinearActivation,
    "mish": MishActivation,
    "quick_gelu": QuickGELUActivation,
    "relu": nn.ReLU,
    "relu2": ReLUSquaredActivation,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": SiLUActivation,
    "sqrtsoftplus": SqrtSoftplusActivation,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
    "prelu": nn.PReLU,
    "xielu": XIELUActivation,
}
ACT2FN = ClassInstantier(ACT2CLS)
"""

class RotaryEmbedding(nn.Module):
    """
    References:
    https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/models/llama/modeling_llama.py#L96 
    """
    def __init__(self, dim, max_position_embeddings, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos_cached", emb.cos().to(torch.get_default_dtype()), persistent=False)
        self.register_buffer("_sin_cached", emb.sin().to(torch.get_default_dtype()), persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_heads, seq_len, head_dim]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class SinusoidalEmbedding(nn.Module):
    """
    Reference: https://github.com/butsugiri/shape/blob/main/onmt/modules/embeddings.py
    """

    def __init__(self, dim, max_position_embeddings):
        
        pe = torch.zeros(max_position_embeddings, dim)
        position = torch.arange(0, max_position_embeddings).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        #pe = pe.unsqueeze(1)
        super(SinusoidalEmbedding, self).__init__()
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: position_ids (bs, seq_len)
        pe = torch.stack([self.pe[_x] for _x in x])
        #print(pe.size())
        #print(pe[:5, :30, 6])
        return pe
    
"""
References:
https://github.com/huggingface/transformers/blob/v4.5.0/src/transformers/models/gpt2/modeling_gpt2.py#L125

"""
class Attention(nn.Module):
    def __init__(
        self,
        config,
        # log_activations=False,
    ):

        super().__init__()

        self.config = config
        # self.max_seq_len = config.max_seq_len
        # self.log_activations = log_activations

        self.hidden_size = config.hidden_size
        self.num_attn_heads = config.num_attn_heads
        self.head_dim = self.hidden_size // self.num_attn_heads
        
        if self.head_dim * self.num_attn_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_attn_heads (got `hidden_size`: {self.hidden_size} and `num_attn_heads`:"
                f" {self.num_attn_heads})."
            )

        
        self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        # if self.log_activations:
        #     self.activation_stats = {
        #         "attentiscon_weight_max": None,
        #         "attention_weight_min": None,
        #     }

        if self.config.rotary_PE:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.config.max_position_embeddings,
            )
    
    def merge_heads(self, x):
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, num_heads * head_dim)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_attn_heads, self.head_dim)
        return x.transpose(1, 2)  # (batch, num_attn_heads, seq_length, head_dim)

    def _mask_padding_mask(self, binary_mask, device, dtype):
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
        attention_scores will have shape [batch_size, num_heads, query_len, key_len]
        padding mask only depends on the key position, not on head or query position.
        batch_size, 1, 1, key_len] will be broadcasted to [batch_size, num_heads, query_len, key_len]
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
    
    def _make_alibi_bias(self, seq_len, device, dtype):
        # Expected shape: [1, num_heads, query_len, key_len]
        raise NotImplementedError("ALiBi is not implemented yet.")
    
    def _make_relative_position_bias(self, seq_len, device, dtype):
        # Expected shape: [1, num_heads, query_len, key_len]
        raise NotImplementedError("Relative Position is not implemented yet.")
    
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
        query_len,
        key_len,
        device,
        dtype,
    ):
        """ Returns an additive attention bias broadcastable to:
            [batch, num_heads, query_len, key_len]

            values: 0.0 -- attention; large negative value -- masked
        """
        result = None

        # A. Padding mask.
        if binary_mask is not None:
            padding_mask = self._mask_padding_mask(binary_mask, device, dtype)
            result = self._add_bias(result, padding_mask)

        # B. Causal mask.
        if getattr(self.config, "causal", True):
            causal_mask = self._make_causal_mask(key_len, device, dtype)
            result = self._add_bias(result, causal_mask)

        # C. ALiBi bias.
        if getattr(self.config, "alibi_PE", False):
            alibi_bias = self._make_alibi_bias(key_len, device, dtype)
            result = self._add_bias(result, alibi_bias)

        # D. Relative Position bias.
        if getattr(self.config, "relative_PE", False):
            relative_position_bias = self._make_relative_position_bias(key_len, device, dtype)
            result = self._add_bias(result, relative_position_bias)

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

        # 3. Rotary PE modifies q/k before attention
        if getattr(self.config, "rotary_PE", False):
            if position_ids is None:
                raise ValueError("position_ids must be provided when using rotary positional embeddings.")
            cos, sin = self.rotary_emb(value, position_ids)
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
        
        # 4. Build attention mask.
        additive_mask = self._build_attn_mask(
            binary_mask=binary_mask,
            query_len=seq_len,
            key_len=seq_len,
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
            [batch or 1, heads or 1, query_len, key_len]

        Returns:
            attn_output:  [batch, heads, query_len, head_dim]
            attn_weights: [batch, heads, query_len, key_len]
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
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        mlp_inner_dim = config.mlp_dim_multipler * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        
        self.mlp = MLP(mlp_inner_dim, config)

    def forward(
        self, 
        hidden_states, 
        binary_mask=None, 
        position_ids=None,
        expose_attn_weights=False,
    ):
        if expose_attn_weights:
            attn_output, attn_weights = self.attn(self.ln_1(hidden_states), binary_mask, position_ids, expose_attn_weights)
        else:
            attn_output = self.attn(self.ln_1(hidden_states), binary_mask, position_ids, expose_attn_weights)
        hidden_states = attn_output + hidden_states # residual connection

        feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states))
        hidden_states = hidden_states + feed_forward_hidden_states # residual connection

        if expose_attn_weights:
            return hidden_states, attn_weights
        else:
            return hidden_states  


class MLP(nn.Module):
    def __init__(self, mlp_inner_dim, config):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        embed_dim = config.hidden_size
        self.in_proj = nn.Linear(embed_dim, mlp_inner_dim)
        self.out_proj = nn.Linear(mlp_inner_dim, embed_dim)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.in_proj(x))
        h2 = self.out_proj(h)
        return self.dropout(h2)


class Causal_Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.vocab_size = len(config.vocab)

        eye = torch.eye(self.config.max_seq_len).unsqueeze(0)
        self.register_buffer("eye", eye, persistent=False)

        num_embeddings = self.embed_dim-1 if config.scaler_posemb else self.embed_dim
        self.wte = nn.Embedding(self.vocab_size, num_embeddings)

        if self.config.absolute_posemb: 
            self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
            if self.config.absolute_posemb_freeze: self.wpe.weight.requires_grad = False
            if self.config.absolute_posemb_initfromsine: 
                self.wpe.weight.data = SinusoidalEmbedding(self.embed_dim, config.max_position_embeddings).pe
        if self.config.sinusoidal_posemb: self.sine = SinusoidalEmbedding(self.embed_dim, config.max_position_embeddings)

        
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)
        

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

         # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name == "out_proj.weight":
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers)))



    def forward(
        self,
        input_ids,
        binary_mask=None, 
        position_ids=None,
        expose_attn_weights=False,
    ):

        input_shape = input_ids.size() # bs, seq_len
        input_ids = input_ids.view(-1, input_shape[-1])
        device = input_ids.device

        inputs_embeds = self.wte(input_ids)
        hidden_states = inputs_embeds

        position_ids = position_ids.view(-1, input_shape[-1])
        
        # past_length = 0        
        # noshift_position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
        # noshift_position_ids = noshift_position_ids.unsqueeze(0).view(-1, input_shape[-1])
        
        if self.config.absolute_posemb:
            #_ = position_ids if self.config.absolute_posemb_shift else noshift_position_ids
            #position_embeds = self.wpe(_)
            position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds

        if self.config.rotary_posemb:
            position_ids_for_rotary = position_ids #if self.config.rotary_posemb_shift else noshift_position_ids
        else:
            position_ids_for_rotary = None

        if self.config.scaler_posemb:
            hidden_states = torch.cat([hidden_states, position_ids.unsqueeze(-1)], dim=-1)

        if self.config.sinusoidal_posemb:
            position_embeds = self.sine(position_ids)
            hidden_states = hidden_states * math.sqrt(self.embed_dim) + position_embeds
            #print(position_embeds[:5, :30, 6])
        
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),) # bs, seq_len, hidden_size

        # collect per-layer attn_weights
        attn_weights_list = [] # a list of [batch, heads, query_len, key_len] tensors
        for block in self.h:
            if expose_attn_weights:
                hidden_states, attn_weights = block(hidden_states, binary_mask, position_ids_for_rotary, expose_attn_weights)
                attn_weights_list.append(attn_weights)
            else:
                hidden_states = block(hidden_states, binary_mask, position_ids_for_rotary)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(*output_shape)

        lm_logits = self.lm_head(hidden_states) # loss calculated outside

        if expose_attn_weights:
            return lm_logits, attn_weights_list
        else:
            return lm_logits