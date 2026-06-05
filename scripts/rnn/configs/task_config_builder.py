from configs import TaskConfig
from typing import Literal
import re

SupervisionLevel = Literal["S1", "S2", "S3"]
DepthMode = Literal["d", "leqd", "trulyleqd"]
LengthVariant = Literal["original", "shorter"]

LENGTH_VARIANT_SPECS = {
    "original": {
        "root": "/data/yingshac/tmlr",
        "min_test_len": 41,
        "max_test_len": 120,
    },
    "shorter": {
        "root": "/data/yingshac/tmlr/shorter",
        "min_test_len": 21,
        "max_test_len": 60,
    },
}
VAL_DEPTHS = list(range(1, 17)) # validation is always 1..16, just hard-code it


def make_path(root: str, split: str, d: int) -> str:
    return f"{root}/{split}/fix_d_{d}.json"


def make_train_files(
    root: str,
    target_d: int,
    depth_mode: DepthMode,
) -> list[str]:
    """
    Construct train file paths based on depth mode.

    depth_mode:
        "d"          -> train on exactly target_d
        "leqd"       -> train on d = 2, ..., target_d
        "trulyleqd"  -> train on d = 1, ..., target_d
    """
    if depth_mode == "d":
        train_depths = [target_d]

    elif depth_mode == "leqd":
        train_depths = list(range(2, target_d + 1))

    elif depth_mode == "trulyleqd":
        train_depths = list(range(1, target_d + 1))

    else:
        raise ValueError(f"Unknown depth_mode: {depth_mode}")

    return [
        make_path(root, "train", d)
        for d in train_depths
    ]

def make_val_files(
    root: str,
    target_d: int,
) -> list[tuple[str, str, int | None]]:
    """
    Put target_d first, then all other validation depths.
    """
    ordered_depths = [target_d] + [d for d in VAL_DEPTHS if d != target_d]

    return [
        (make_path(root, "validation", d), f"d_{d}", None)
        for d in ordered_depths
    ]


def make_test_files(
    root: str,
    target_d: int,
    min_test_len: int,
    max_test_len: int,
) -> list[tuple[str, str, int]]:
    return [
        (make_path(root, "test", target_d), f"d_{target_d}_l_{l}", l)
        for l in range(min_test_len, max_test_len + 1)
    ]

# So S1/S2/S3 can use use_eos=True, while S1a_/S2a_/S3a_ can use use_eos=False.
def make_in_vocab(use_eos: bool) -> list[str]:
    vocab = ["(", ")"]
    if use_eos: vocab.append("<eos>")
    vocab.append("<pad>")
    return vocab

def add_supervision_specific_configs(
    cfg: TaskConfig,
    supervision_level: SupervisionLevel,
) -> TaskConfig:
    
    if supervision_level == "S1":
        return cfg

    if supervision_level == "S2":
        cfg.out_S2_vocab = ["0", "1"]
        cfg.S2_loss_weight = 0.5
        return cfg

    if supervision_level == "S3":
        cfg.out_S3_vocab = [f"d{d}" for d in range(-1, 18)]
        cfg.S3_loss_weight = 1.0
        return cfg

    raise ValueError(f"Unknown supervision_level: {supervision_level}")

def parse_task_name(task_name: str) -> dict:
    """
    Return builder-internal variables. They are not stored in TaskConfig.
    {
        "dataset": str,
        "supervision_level": SupervisionLevel,
        "use_eos": bool,
        "depth_mode": DepthMode,
        "target_d": int,
        "length_variant": LengthVariant,
    }
    """
    is_shorter = task_name.startswith("shorter")
    name = task_name
    if is_shorter: name = name.removeprefix("shorter")

    match = re.fullmatch(
        r"(?P<family>S[123]a?)_d_(?P<mode>trulyleq|leq)?(?P<d>\d+)",
        name,
    )

    if match is None:
        raise ValueError(f"Cannot parse task name: {task_name}")
    
    family = match.group("family")
    mode = match.group("mode")
    target_d = int(match.group("d"))

    supervision_level = family[:2]  # S1/S2/S3/S1a/S2a/S3a
    use_eos = not family.endswith("a")  # S1/S2/S3 use_eos=True, S1a/S2a/S3a use_eos=False

    if mode is None:
        depth_mode = "d"
    elif mode == "leq":
        depth_mode = "leqd"
    elif mode == "trulyleq":
        depth_mode = "trulyleqd"
    else:
        raise ValueError(f"Unknown mode in task name: {task_name}")
    
    dataset = family if use_eos else f"{family}_"
    length_variant = "shorter" if is_shorter else "original"

    return {
        "dataset": dataset,
        "supervision_level": supervision_level,
        "use_eos": use_eos,
        "depth_mode": depth_mode,
        "target_d": target_d,
        "length_variant": length_variant,
    }

def task_config_builder(task: str) -> TaskConfig:
    parsed = parse_task_name(task)
    
    variant_spec = LENGTH_VARIANT_SPECS[parsed["length_variant"]]
    root = variant_spec["root"]
    min_test_len = variant_spec["min_test_len"]
    max_test_len = variant_spec["max_test_len"]

    cfg = TaskConfig(        
        dataset=parsed["dataset"],
        
        in_vocab=make_in_vocab(parsed["use_eos"]),
        out_vocab=['a', 'r'],
        
        train_files=make_train_files(
            root=root, 
            target_d=parsed["target_d"], 
            depth_mode=parsed["depth_mode"]
        ),
        val_files=make_val_files(
            root=root, 
            target_d=parsed["target_d"]
        ),
        test_files=make_test_files(
            root=root, 
            target_d=parsed["target_d"], 
            min_test_len=min_test_len, 
            max_test_len=max_test_len
        ),
    )

    cfg = add_supervision_specific_configs(cfg, parsed["supervision_level"])
    return cfg