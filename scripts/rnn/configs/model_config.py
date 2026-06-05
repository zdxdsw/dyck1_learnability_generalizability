from dataclasses import dataclass, field

@dataclass
class RNNConfig:
    num_hidden_layers: int = 1
    hidden_size: int = 64
    initializer_range: float = 0.02
    dropout: float = 0.1


@dataclass
class PositionEncodingConfig:
    pe_type: str = "none"
    max_position_embeddings: int = 512
    relative_pe_num_buckets: int = 32
    relative_pe_max_distance: int = 128


@dataclass
class TransformerConfig:
    num_hidden_layers: int = 4
    hidden_size: int = 64
    num_attn_heads: int = 4
    initializer_range: float = 0.02

    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5

    causal: bool = True
    mlp_dim_multiplier: int = 4
    activation_function: str = "gelu"

    position_encoding: PositionEncodingConfig = field(default_factory=PositionEncodingConfig)