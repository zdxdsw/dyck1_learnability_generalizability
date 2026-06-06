import random, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from contextlib import redirect_stdout
import torch
from torch.utils.data import DataLoader
from configs import flat_everything_config_builder
from utils import sequence_collator_builder
from dataset import *


# Change this manually when testing.
TASK = "S3a_d_trulyleq5"
RANDOM_SEED = 42

def get_first_val_spec(config):
    """
    config.val_files entries are expected to look like:

        (path, name, specific_length)

    After JSON serialization they may become lists instead of tuples,
    so this function accepts both.
    """
    if not hasattr(config, "val_files"):
        raise AttributeError("config does not have field `val_files`.")

    if len(config.val_files) == 0:
        raise ValueError("config.val_files is empty.")

    first_val = config.val_files[0]

    if isinstance(first_val, str):
        return first_val, "", None

    if isinstance(first_val, (tuple, list)):
        if len(first_val) == 3:
            path, name, specific_length = first_val
            return path, name, specific_length

        if len(first_val) == 2:
            path, name = first_val
            return path, name, None

        if len(first_val) == 1:
            return first_val[0], "", None

    raise TypeError(
        f"Unexpected val_files entry: {first_val!r}. "
        f"Expected str or tuple/list of length 1, 2, or 3."
    )


def print_config_summary(config):
    print("=" * 80)
    print("Config summary")
    print("=" * 80)

    fields = [
        "task",
        "model",
        "dataset",
        "in_vocab",
        "out_vocab",
        "out_S2_vocab",
        "S2_loss_weight",
        "out_S3_vocab",
        "S3_loss_weight",
        "max_seq_len",
        "ignore_idx",
        "per_device_eval_batch_size",
        "pe_type",
        "pe_aug_type",
        "max_position_embeddings",
    ]

    for field in fields:
        if hasattr(config, field):
            print(f"{field:28s} = {getattr(config, field)}")

    print(f"{'num_train_files':28s} = {len(config.train_files)}")
    print(f"{'num_val_files':28s} = {len(config.val_files)}")
    print(f"{'num_test_files':28s} = {len(config.test_files)}")


def print_raw_item(item, idx: int):
    print("\n" + "=" * 80)
    print(f"Raw dataset item at index {idx}")
    print("=" * 80)

    for k, v in item.items():
        print(f"{k}: {v}")


def print_batch_shapes(batch):
    print("\n" + "=" * 80)
    print("Batch tensor shapes")
    print("=" * 80)

    for k, v in batch.items():
        if torch.is_tensor(v):
            print(
                f"{k:28s} "
                f"shape={tuple(v.shape)!s:20s} "
                f"dtype={str(v.dtype):15s} "
                f"device={v.device}"
            )
        else:
            print(f"{k:28s} type={type(v)} value={v}")


def print_one_batched_item(batch, item_idx: int):
    print("\n" + "=" * 80)
    print(f"Full tensors for randomly chosen batch item: {item_idx}")
    print("=" * 80)

    for k, v in batch.items():
        print("\n" + "-" * 80)
        print(k)
        print("-" * 80)

        if torch.is_tensor(v):
            if v.ndim == 0:
                print(v)
            elif v.shape[0] > item_idx:
                print(v[item_idx])
            else:
                print(
                    f"Cannot index item {item_idx}; "
                    f"tensor shape is {tuple(v.shape)}"
                )
        else:
            print(v)


def run_test():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    torch.set_printoptions(
        threshold=float("inf"),
        linewidth=240,
        edgeitems=100000,
    )

    config = flat_everything_config_builder(TASK)

    print_config_summary(config)

    val_file, val_name, specific_length = get_first_val_spec(config)

    print("\n" + "=" * 80)
    print("Validation file selected")
    print("=" * 80)
    print(f"val_file        = {val_file}")
    print(f"val_name        = {val_name}")
    print(f"specific_length = {specific_length}")

    dataset_cls = eval(f"{config.dataset}Dataset")

    print("\n" + "=" * 80)
    print("Dataset class")
    print("=" * 80)
    print(f"dataset class   = {dataset_cls.__name__}")

    dataset = dataset_cls(
        files=[val_file],
        config=config,
        name=val_name,
        specific_length=specific_length,
    )

    print(f"dataset size    = {len(dataset)}")

    if len(dataset) == 0:
        raise ValueError(
            f"Dataset is empty for val_file={val_file}, "
            f"specific_length={specific_length}"
        )

    raw_idx = random.randint(0, len(dataset) - 1)
    raw_item = dataset[raw_idx]
    print_raw_item(raw_item, raw_idx)

    collate_fn = sequence_collator_builder(config)

    print("\n" + "=" * 80)
    print("Collator")
    print("=" * 80)
    print(f"collator class  = {collate_fn.__class__.__name__}")

    loader = DataLoader(
        dataset,
        batch_size=config.per_device_eval_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    batch = next(iter(loader))

    print_batch_shapes(batch)

    actual_batch_size = None
    for v in batch.values():
        if torch.is_tensor(v) and v.ndim >= 1:
            actual_batch_size = v.shape[0]
            break

    if actual_batch_size is None:
        raise ValueError("Could not infer batch size from collated batch.")

    item_idx = random.randint(0, actual_batch_size - 1)
    print_one_batched_item(batch, item_idx)


def main():
    output_path = Path(__file__).with_name("test_sequence_collator_builder.txt")

    # Print to console.
    run_test()

    # Also save the same output to sanity/test_sequence_collator_builder.txt.
    with open(output_path, "w") as f:
        with redirect_stdout(f):
            run_test()

if __name__ == "__main__":
    main()