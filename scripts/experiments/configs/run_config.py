from dataclasses import dataclass, field

@dataclass
class RunConfig:
    # lists are mutable objects, so we use field(default_factory=...) to avoid all instances sharing the same list
    seed: int | list[int] = field(default_factory=lambda: [31, 42, 49, 57, 61, 77, 79, 89, 91])
    model: str = "Transformer"
    task: str = ""
    date: str = ""

    hf_cache_dir: str = "/data/yingshac/hf_cache"
    output_dir: str = "output/S1_d_[2-8]_4B/"
    ckpt_dir: str = "/data/yingshac/tmlr/ckpts"
    load_from_dir: str | None = None
    init_from_ckpt: str | None = None