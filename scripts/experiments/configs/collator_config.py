from dataclasses import dataclass

@dataclass
class CollatorConfig:
    max_seq_len: int = 41
    ignore_idx: int = -1