"""
    I implemented two Attention.forward() pathways. 
    One using F.SPDA, the other is a manual implementation to expose attn_weights (post-softmax).
    This test checks the equivalence of these two pathways in core attention operation.
"""
import torch
import torch.nn.functional as F

### Test function: direct tensor-level equivalence
def test_sdpa_matches_manual_attention(
    attention_module,
    batch_size=2,
    seq_len=8,
    dtype=torch.float32,
    device=None,
    causal=True,
    use_padding_mask=True,
    atol=1e-6, # absolute tolerance for torch.testing.assert_close
    rtol=1e-5, # relative tolerance for torch.testing.assert_close
):
    """
    Tests that:

        F.scaled_dot_product_attention(q, k, v, attn_mask=additive_mask, dropout_p=0.0, is_causal=False)
        ==
        softmax[(QK^T / sqrt(head_dim)) + additive_mask)] @ V

    Assumes Attention._build_attn_mask and Attention._manual_attn exist.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    attention_module = attention_module.to(device=device)
    attention_module.eval()

    old_causal = getattr(attention_module.config, "causal", True)
    attention_module.config.causal = causal

    try:
        query = torch.randn(
            batch_size,
            attention_module.num_attn_heads,
            seq_len,
            attention_module.head_dim,
            device=device,
            dtype=dtype,
        )
        key = torch.randn_like(query)
        value = torch.randn_like(query)

        if use_padding_mask:
            # [batch, seq], 1 = keep, 0 = pad
            binary_mask = torch.ones(
                batch_size,
                seq_len,
                device=device,
                dtype=torch.long,
            )

            # Make the final few tokens padding for one example.
            if seq_len >= 4:
                binary_mask[0, -2:] = 0
        else:
            attention_mask = None

        additive_mask = attention_module._build_attn_mask(
            binary_mask=binary_mask,
            query_len=seq_len,
            key_len=seq_len,
            device=device,
            dtype=dtype,
        )

        sdpa_out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=additive_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        manual_out, manual_weights = attention_module._manual_attn(
            query=query,
            key=key,
            value=value,
            additive_mask=additive_mask,
            expose_attn_weights=True,
        )

        torch.testing.assert_close(
            sdpa_out,
            manual_out,
            atol=atol,
            rtol=rtol,
        )

        print("✅ SDPA and manual attention outputs match.")
        print(f"max_abs_diff = {(sdpa_out - manual_out).abs().max().item():.3e}")
        print(f"sdpa_out shape = {tuple(sdpa_out.shape)}")
        print(f"manual_weights shape = {tuple(manual_weights.shape)}")

        return {
            "sdpa_out": sdpa_out,
            "manual_out": manual_out,
            "manual_weights": manual_weights,
            "additive_mask": additive_mask,
        }

    finally:
        attention_module.config.causal = old_causal

### Test function: full Attention.forward equivalence
def test_attention_forward_sdpa_matches_manual(
    attention_module,
    batch_size=2,
    seq_len=8,
    dtype=torch.float32,
    device=None,
    atol=1e-6, # absolute tolerance for torch.testing.assert_close
    rtol=1e-5, # relative tolerance for torch.testing.assert_close
):
    """
    Tests full Attention.forward equivalence between:
        attention_impl="sdpa", i.e. expose_attn_weights=False
    and:
        attention_impl="manual", i.e. expose_attn_weights=True

    Run in eval mode, no dropout.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    attention_module = attention_module.to(device=device, dtype=dtype)
    attention_module.eval()

    hidden_states = torch.randn(
        batch_size,
        seq_len,
        attention_module.hidden_size,
        device=device,
        dtype=dtype,
    )

    binary_mask = torch.ones(
        batch_size,
        seq_len,
        device=device,
        dtype=torch.long,
    )

    if seq_len >= 4:
        binary_mask[0, -2:] = 0

    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(
        batch_size,
        -1,
    )

    with torch.no_grad():
        sdpa_out = attention_module(
            hidden_states=hidden_states,
            binary_mask=binary_mask,
            position_ids=position_ids,
            expose_attn_weights=False,
        )

        manual_out, manual_weights = attention_module(
            hidden_states=hidden_states,
            binary_mask=binary_mask,
            position_ids=position_ids,
            expose_attn_weights=True,
        )

    torch.testing.assert_close(
        sdpa_out,
        manual_out,
        atol=atol,
        rtol=rtol,
    )

    print("✅ Full Attention.forward SDPA/manual paths match.")
    print(f"max_abs_diff = {(sdpa_out - manual_out).abs().max().item():.3e}")
    print(f"output shape = {tuple(sdpa_out.shape)}")
    print(f"attention weights shape = {tuple(manual_weights.shape)}")

    return {
        "sdpa_out": sdpa_out,
        "manual_out": manual_out,
        "manual_weights": manual_weights,
    }
# main
if __name__ == "__main__":
    from transformer_model import Attention
    from types import SimpleNamespace
    torch.manual_seed(0)
    """
    Make a tiny config object with only the fields Attention actually needs
    """
    def make_test_config(hidden_size, num_attention_heads, max_position_embeddings, causal, rotary_PE, alibi_PE, relative_PE):
        return SimpleNamespace(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            resid_pdrop=0.0,
            causal=causal,
            rotary_PE=rotary_PE,
            alibi_PE=alibi_PE,
            relative_PE=relative_PE,
            max_position_embeddings=max_position_embeddings,
        )

    config = make_test_config(
        hidden_size=64,
        num_attention_heads=8,
        max_position_embeddings=122,
        causal=True,
        rotary_PE=False,
        alibi_PE=False,
        relative_PE=False
    )
    attn = Attention(config)
    attn.eval()

    # 1. CPU float32, no padding, causal
    test_sdpa_matches_manual_attention(attn, device="cpu", dtype=torch.float32, use_padding_mask=False)

    # 2. CPU float32, padding + causal
    test_sdpa_matches_manual_attention(attn, device="cpu", dtype=torch.float32, use_padding_mask=True)

    # 3. Full forward, CPU float32
    test_attention_forward_sdpa_matches_manual(attn, device="cpu", dtype=torch.float32)

    # 4. CUDA float32
    test_attention_forward_sdpa_matches_manual(attn, device="cuda", dtype=torch.float32, atol=1e-5, rtol=1e-5)

    # 5. CUDA float16, loosen tolerance
    test_sdpa_matches_manual_attention(attn, device="cuda",dtype=torch.float16, atol=1e-3, rtol=1e-3)
    test_attention_forward_sdpa_matches_manual(attn, device="cuda", dtype=torch.float16, atol=1e-3, rtol=1e-3)

