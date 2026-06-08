"""
Sanity-check data loading and sequence collation.

Run from PROJECT_ROOT, for example:

    python sanity/test_dataloader.py --task S2a_your_task_name

This writes an aligned, human-readable inspection sample to:

    sanity/test_dataloader.txt

The script intentionally mirrors the data-loading path in trainer.py:
    config -> Dataset(config.val_files, config) -> sequence_collator_builder(config) -> DataLoader
"""

import os, random, sys, argparse
from pprint import pformat

import numpy as np
import torch
from torch.utils.data import DataLoader


# Allow this file to live in PROJECT_ROOT/sanity/ while importing project modules.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs import flat_everything_config_builder  
from dataset import *  
from utils import sequence_collator_builder


def as_list(x):
    """Convert tensors / numpy arrays / tuples to a plain Python list."""
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)


# def set_seed(seed: int) -> None:
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)


# def get_first_seed(config) -> int:
#     seed = getattr(config, "seed", 0)
#     if isinstance(seed, (list, tuple)):
#         return int(seed[0])
#     return int(seed)


def select_val_files(config, val_index):
    """
    config.val_files follows trainer.py's convention:
        [(path, name), ...]
    Optionally inspect one split by index; otherwise combine all validation paths.
    """
    if not getattr(config, "val_files", None):
        raise ValueError("config.val_files is empty or missing.")

    if val_index is not None:
        entry = config.val_files[val_index]
        path, name = entry[0], entry[1] if len(entry) > 1 else f"val_{val_index}"
        return [path], name

    paths = [entry[0] for entry in config.val_files]
    names = [entry[1] if len(entry) > 1 else f"val_{i}" for i, entry in enumerate(config.val_files)]
    return paths, " + ".join(names)


def id_to_token(idx, vocab):
    if idx == -1:
        return "·"  # ignore_index / unsupervised label
    if idx is None:
        return "None"
    if 0 <= int(idx) < len(vocab):
        return str(vocab[int(idx)])
    return f"<bad:{idx}>"


def label_vocab_for_key(config, key):
    if key == "labels":
        return getattr(config, "out_vocab")
    # e.g. S2_labels -> out_S2_vocab; S3_labels -> out_S3_vocab
    tag = key.removesuffix("_labels")
    attr = f"out_{tag}_vocab"
    if hasattr(config, attr):
        return getattr(config, attr)
    return None


def decode_ids(ids, vocab):
    return [id_to_token(i, vocab) for i in ids]


def pad_to_len(xs, n, pad=""):
    return xs + [pad] * max(0, n - len(xs))


def aligned_block(rows):
    """
    rows: list[(name, values)] where values is list[str/int].
    Returns a monospaced block with one sequence position per aligned column.
    """
    str_rows = [(name, [str(v) for v in values]) for name, values in rows]
    max_len = max(len(values) for _, values in str_rows)
    str_rows = [(name, pad_to_len(values, max_len)) for name, values in str_rows]

    pos_values = [str(i) for i in range(max_len)]
    all_value_rows = [pos_values] + [values for _, values in str_rows]
    widths = [max(len(row[i]) for row in all_value_rows) for i in range(max_len)]
    name_width = max(len("pos"), *(len(name) for name, _ in str_rows))

    def fmt_line(name, values):
        cells = [str(v).rjust(widths[i]) for i, v in enumerate(values)]
        return f"{name:<{name_width}} : " + "  ".join(cells)

    lines = [fmt_line("pos", pos_values)]
    lines.extend(fmt_line(name, values) for name, values in str_rows)
    return "\n".join(lines)


def summarize_config(config, val_paths, val_name):
    keys = [
        "task",
        "dataset",
        "model",
        "max_seq_len",
        "per_device_eval_batch_size",
        "ignore_idx",
        "pe_aug_type",
        "max_position_embeddings",
    ]
    available = {k: getattr(config, k) for k in keys if hasattr(config, k)}
    return (
        "CONFIG SUMMARY\n"
        "==============\n"
        f"{pformat(available)}\n\n"
        f"Validation split inspected: {val_name}\n"
        f"Validation files:\n" + "".join(f"  - {p}\n" for p in val_paths)
    )
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    args = parser.parse_args()

    # ---------------- Inline sanity-test settings ----------------
    output = os.path.join(THIS_DIR, "test_dataloader.txt")

    # None means use config.per_device_eval_batch_size.
    batch_size_override = None

    # None means combine all config.val_files paths into one validation dataset.
    # Set to an int, e.g. 0, to inspect only config.val_files[0].
    val_index = None

    num_workers = 0

    # False means truncate printed sequences to each item's true input_length.
    # True means print the full padded config.max_seq_len.
    keep_padding = False
    # -------------------------------------------------------------

    config = flat_everything_config_builder(args.task)
    config.task = args.task

    val_paths, val_name = select_val_files(config, val_index)
    dataset_class = eval(f"{config.dataset}Dataset")
    val_dataset = dataset_class(val_paths, config, name=val_name)

    if len(val_dataset) == 0:
        raise ValueError(f"Validation dataset is empty for paths: {val_paths}")

    collator = sequence_collator_builder(config)

    batch_size = batch_size_override or getattr(config, "per_device_eval_batch_size", 1)
    batch_size = min(batch_size, len(val_dataset))

    dataloader = DataLoader(
        val_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collator,
        pin_memory=False,
        num_workers=num_workers,
    )

    batch = next(iter(dataloader))

    output_lines = []
    output_lines.append(summarize_config(config, val_paths, val_name))
    output_lines.append("\nDATASET / BATCH SUMMARY\n=======================")
    output_lines.append(f"num_val_examples: {len(val_dataset)}")
    output_lines.append(f"batch_size: {batch_size}")
    output_lines.append(f"batch_keys: {sorted(batch.keys())}")

    output_lines.append("\nALIGNED BATCH ITEMS\n===================")

    for batch_item_idx in range(batch["input_ids"].size(0)):
        input_length = int(batch["input_lengths"][batch_item_idx].item())
        stop = batch["input_ids"].size(1) if keep_padding else input_length

        input_ids = as_list(batch["input_ids"][batch_item_idx])[:stop]
        raw_tokens = decode_ids(input_ids, config.in_vocab)

        rows = [
            ("raw_tok", raw_tokens),
            ("input_id", input_ids),
        ]

        # Include all collated label sequences: labels, S2_labels, S3_labels, etc.
        for key in sorted(k for k in batch.keys() if "labels" in k):
            ids = as_list(batch[key][batch_item_idx])[:stop]
            vocab = label_vocab_for_key(config, key)

            if vocab is not None:
                rows.append((f"{key}_tok", decode_ids(ids, vocab)))

            rows.append((key, ids))

        # Transformer-specific debug fields, if present.
        for key in ["binary_mask", "position_ids"]:
            if key in batch:
                rows.append((key, as_list(batch[key][batch_item_idx])[:stop]))

        output_lines.append(f"\n--- batch item {batch_item_idx} ---")
        output_lines.append(f"input_length_before_padding: {input_length}")
        output_lines.append(f"printed_length: {stop}")
        output_lines.append(aligned_block(rows))

    text = "\n".join(output_lines) + "\n"

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        f.write(text)

    print(text)

if __name__ == "__main__":
    main()
