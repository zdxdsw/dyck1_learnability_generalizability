from dataclasses import dataclass, field
from typing import Literal

PEAugType = Literal["none", "shifted", "randomized"]

""" Position-id augmentation.
    "none":       [0, 1, ..., seq_len - 1]
    "shifted":    contiguous positions with random offset
    "randomized": sorted random positions
"""

@dataclass
class RNNConfig:
    num_hidden_layers: int = 1
    hidden_size: int = 32
    initializer_range: float = 0.02
    dropout: float = 0.1


@dataclass
class PositionEncodingConfig:
    pe_type: str = "relative"
    pe_aug_type: PEAugType = "randomized"
    max_position_embeddings: int = 128
    relative_pe_num_buckets: int = 32
    relative_pe_max_distance: int = 128


@dataclass
class TransformerConfig:
    num_hidden_layers: int = 2
    hidden_size: int = 64
    num_attn_heads: int = 8
    initializer_range: float = 0.02

    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5

    causal: bool = False
    mlp_dim_multiplier: int = 4
    activation_function: str = "gelu"

    position_encoding: PositionEncodingConfig = field(default_factory=PositionEncodingConfig)

def model_config_builder(model: str) -> RNNConfig | TransformerConfig:
    if model == "RNN":
        return RNNConfig()

    if model == "Transformer":
        return TransformerConfig()

    raise ValueError(f"Unknown model type: {model}")