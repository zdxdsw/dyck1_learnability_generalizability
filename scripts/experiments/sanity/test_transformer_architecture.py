import os, sys
from types import SimpleNamespace
from contextlib import redirect_stdout
import torch

# Make project_root importable when running from sanity/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformer_model import TransformerModel


PE_TYPES = [
    "none",
    "absolute",
    "sinusoidal",
    "rotary",
    "scaler",
    "alibi",
    "relative",
]


def build_config(pe_type: str):
    """
    Minimal config object matching what transformer_model.py currently expects.
    Flattened to incorporate PositionEncodingConfig.
    Additional non-model-specific fields: in_vocab, out_vocab.
    """
    return SimpleNamespace(
        num_hidden_layers=2,
        hidden_size=64,
        num_attn_heads=8,
        initializer_range=0.02,

        embd_pdrop=0.1,
        resid_pdrop=0.1,
        layer_norm_epsilon=1e-5,

        causal=True,
        mlp_dim_multiplier=4,
        activation_function="gelu",

        # Positional encoding fields
        pe_type=pe_type,
        pe_aug_type="shifted",
        max_position_embeddings=128,
        relative_pe_num_buckets=32,
        relative_pe_max_distance=128,

        in_vocab=["(", ")", "<eos>", "<pad>"],
        out_vocab=["a", "r"],
    )


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_summary(pe_type: str):
    print("=" * 100)
    print(f"PE type: {pe_type}")
    print("=" * 100)

    config = build_config(pe_type)
    model = TransformerModel(config)

    total, trainable = count_parameters(model)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print()

    print("Named parameters:")
    for name, param in model.named_parameters():
        print(
            f"{name:<32} "
            f"shape={tuple(param.shape)!s:<11} "
            f"numel={param.numel():>8,} "
            f"requires_grad={param.requires_grad}"
        )

    print()
    print("Named buffers:")
    buffers = list(model.named_buffers())
    if len(buffers) == 0:
        print("(none)")
    else:
        for name, buffer in buffers:
            print(
                f"{name:<32} "
                f"shape={tuple(buffer.shape)!s:<11} "
                f"numel={buffer.numel():>8,} "
                f"dtype={buffer.dtype}"
            )

    print()
    print("Forward pass sanity check:")

    batch_size = 32
    seq_len = 41

    input_ids = torch.randint(
        low=0,
        high=len(config.in_vocab),
        size=(batch_size, seq_len),
    )

    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    binary_mask = torch.ones(batch_size, seq_len)

    model.eval()
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            position_ids=position_ids,
            binary_mask=binary_mask,
            expose_attn_weights=True,
        )

    print(f"logits shape: {tuple(output['logits'].shape)}")
    print(f"num attention layers returned: {len(output['attn_weights'])}")

    for layer_idx, attn in enumerate(output["attn_weights"]):
        print(f"attn_weights[{layer_idx}] shape: {tuple(attn.shape)}")

    print()


def main():
    output_path = os.path.join(
        os.path.dirname(__file__),
        "test_transformer_architecture.txt",
    )

    with open(output_path, "w") as f:
        with redirect_stdout(f):
            print(f"Architecture sanity check")
            print(f"Output file: {output_path}")
            print()

            for pe_type in PE_TYPES:
                print_model_summary(pe_type)


if __name__ == "__main__":
    main()