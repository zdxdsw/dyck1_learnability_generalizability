from dataclasses import dataclass, field

task_suite = "S2_d_trulyleq[2-8]_F"
@dataclass
class RunConfig:
    # lists are mutable objects, so we use field(default_factory=...) to avoid all instances sharing the same list
    # [31, 42, 49, 57, 61, 77, 79, 89, 91]
    seed: int | list[int] = field(default_factory=lambda: [42, 49, 57, 61, 77, 79, 89, 91])
    model: str = "RNN"  # "RNN" or "Transformer"
    task: str = ""
    date: str = ""

    hf_cache_dir: str = "/data/yingshac/hf_cache"
    output_dir: str = f"output/{task_suite}/"
    ckpt_dir: str = f"/data/yingshac/tmlr/ckpts/{task_suite}/"
    load_from_dir: str | None = None
    init_from_ckpt: str | None = None