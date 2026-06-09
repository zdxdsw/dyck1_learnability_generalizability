"""
Sanity-check validation dataset generation and sequence collation.

Run from PROJECT_ROOT:

    python sanity/test_dataloader.py

This writes an aligned, human-readable inspection sample and per-file dataset
statistics to:

    sanity/test_dataloader.txt

This script intentionally hardcodes the validation roots and the task. It uses
the flat config only for convenience, mainly because the dataset class and
sequence collator need config fields such as vocabularies, max_seq_len, model,
and ignore_idx.
"""

import os
import re
import sys
from pprint import pformat

import numpy as np
import torch
from torch.utils.data import DataLoader


# Allow this file to live in PROJECT_ROOT/sanity/ while importing project modules.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs import flat_everything_config_builder  # noqa: E402
from dataset import *  # noqa: F401,F403,E402; needed for eval(f"{config.dataset}Dataset")
from utils import sequence_collator_builder  # noqa: E402


# ---------------- Inline sanity-test settings ----------------
TASK = "S1_d_trulyleq8"

VALIDATION_SOURCES = [
    {
        "variant": "ORIGINAL",
        "root": "/data/yingshac/tmlr/validation",
    },
    {
        "variant": "SHORTER",
        "root": "/data/yingshac/tmlr/shorter/validation",
    },
]

OUTPUT = os.path.join(THIS_DIR, "test_dataloader.txt")

# None means use config.per_device_eval_batch_size.
BATCH_SIZE_OVERRIDE = None

NUM_WORKERS = 0

# False means truncate printed sequences to each item's true input_length.
# True means print the full padded config.max_seq_len.
KEEP_PADDING = False

SPECIAL_RAW_TOKENS = {"<pad>", "<eos>"}
# -------------------------------------------------------------


def as_list(x):
    """Convert tensors / numpy arrays / tuples to a plain Python list."""
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)


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


def formatted_table_with_variant_divider(rows, columns, divider_after_variant="ORIGINAL"):
    """
    rows: list[dict]
    columns: list[str]

    Returns a simple monospaced table. Inserts an extra horizontal divider
    immediately after the last row whose variant == divider_after_variant.
    """
    str_rows = [
        {col: str(row.get(col, "")) for col in columns}
        for row in rows
    ]

    widths = {
        col: max(
            len(col),
            *(len(row[col]) for row in str_rows),
        )
        for col in columns
    }

    def fmt_row(row):
        return " | ".join(row[col].ljust(widths[col]) for col in columns)

    divider = "-+-".join("-" * widths[col] for col in columns)

    lines = []
    header = " | ".join(col.ljust(widths[col]) for col in columns)
    lines.append(header)
    lines.append(divider)

    last_variant = None
    for row in str_rows:
        current_variant = row.get("variant", "")

        if (
            last_variant == divider_after_variant
            and current_variant != divider_after_variant
        ):
            lines.append(divider)

        lines.append(fmt_row(row))
        last_variant = current_variant

    return "\n".join(lines)


def extract_fix_d(filename):
    """
    Extract d from filenames like:
        fix_d_5.json
        fix_d_12.json

    Used for numerical sorting within each validation root.
    """
    match = re.search(r"fix_d_(\d+)\.json$", filename)
    if match is None:
        raise ValueError(f"Could not extract d from validation filename: {filename}")
    return int(match.group(1))


def discover_validation_files(validation_sources):
    """
    Return existing files under each validation root.

    Files are grouped by source, and within each source sorted numerically by d
    extracted from filenames like fix_d_5.json.

    Returns tuples:
        (variant, root, filename, path)
    """
    discovered = []

    for source in validation_sources:
        variant = source["variant"]
        root = source["root"]

        if not os.path.isdir(root):
            print(f"Warning: validation root does not exist or is not a directory: {root}")
            continue

        filenames = [
            filename
            for filename in os.listdir(root)
            if os.path.isfile(os.path.join(root, filename))
        ]
        filenames = sorted(filenames, key=extract_fix_d)

        for filename in filenames:
            path = os.path.join(root, filename)
            discovered.append((variant, root, filename, path))

    return discovered


def summarize_config(config, variant, val_path, val_name):
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
        f"Validation variant: {variant}\n"
        f"Validation files:\n  - {val_path}\n"
    )


def batch_item_rows(batch, batch_item_idx, config, keep_padding=False):
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

    return rows, input_length, stop


def content_raw_tokens_from_item(item, config):
    """
    Decode item['input'] and remove special tokens for structural statistics.

    This is what the requested even-length / boundary-symbol conditions should
    usually refer to. If we kept <eos>, every S1-style sequence would end with
    <eos>, so the "does not end with '('" condition would not test the generated
    parenthesis string itself.
    """
    raw_tokens = decode_ids(as_list(item["input"]), config.in_vocab)
    return [tok for tok in raw_tokens if tok not in SPECIAL_RAW_TOKENS]


def raw_sequence_key_from_item(item, config):
    """
    Return a hashable representation of the raw input token sequence after
    removing <eos> and <pad>.
    """
    return tuple(content_raw_tokens_from_item(item, config))


def label_tokens_from_item(item, config):
    label_ids = as_list(item["labels"])
    return decode_ids(label_ids, config.out_vocab)


def is_negative_item(item, config):
    label_tokens = label_tokens_from_item(item, config)
    supervised_label_tokens = [tok for tok in label_tokens if tok != "·"]
    return len(supervised_label_tokens) > 0 and supervised_label_tokens[-1] == "r"


def matches_target_instance_type(item, config):
    """
    Target type:
      (a) negative instance: last supervised labels_tok is "r"
      (b) even structural raw token length
      (c) structural raw_tok does not begin with ")" and does not end with "("
    """
    raw_tokens = content_raw_tokens_from_item(item, config)

    if len(raw_tokens) == 0:
        return False

    return (
        is_negative_item(item, config)
        and len(raw_tokens) % 2 == 0
        and raw_tokens[0] != ")"
        and raw_tokens[-1] != "("
    )


def compute_target_stats(dataset, config):
    total = len(dataset)
    count = sum(1 for item in dataset if matches_target_instance_type(item, config))
    unique_raw_sequences = {
        raw_sequence_key_from_item(item, config)
        for item in dataset
    }

    proportion = count / total if total > 0 else 0.0
    return {
        "target_count": count,
        "total_count": total,
        "target_proportion": proportion,
        "num_unique_raw_sequences": len(unique_raw_sequences),
    }


def main():
    config = flat_everything_config_builder(TASK)
    config.task = TASK

    validation_files = discover_validation_files(VALIDATION_SOURCES)
    if len(validation_files) == 0:
        raise ValueError(f"No validation files found under VALIDATION_SOURCES: {VALIDATION_SOURCES}")

    dataset_class = eval(f"{config.dataset}Dataset")
    collator = sequence_collator_builder(config)

    output_lines = []
    output_lines.append("DATALOADER SANITY CHECK")
    output_lines.append("=======================")
    output_lines.append(f"Hardcoded task: {TASK}")
    output_lines.append("Validation sources:")
    for source in VALIDATION_SOURCES:
        output_lines.append(f"  - {source['variant']}: {source['root']}")
    output_lines.append("")

    per_file_stat_rows = []

    aggregate_target_count = 0
    aggregate_total_count = 0
    aggregate_unique_raw_sequences = set()

    for file_idx, (variant, root, filename, val_path) in enumerate(validation_files):
        val_name = f"{variant} / {filename}"
        val_dataset = dataset_class([val_path], config, name=val_name)

        if len(val_dataset) == 0:
            output_lines.append(f"\nFILE {file_idx}: {val_name}")
            output_lines.append("=" * (len(f"FILE {file_idx}: {val_name}")))
            output_lines.append(f"Validation variant: {variant}")
            output_lines.append(f"Validation files:\n  - {val_path}")
            output_lines.append("Dataset is empty. Skipping batch preview and statistics.")
            continue

        batch_size = BATCH_SIZE_OVERRIDE or getattr(config, "per_device_eval_batch_size", 1)
        batch_size = min(batch_size, len(val_dataset))

        dataloader = DataLoader(
            val_dataset,
            shuffle=True,
            batch_size=batch_size,
            collate_fn=collator,
            pin_memory=False,
            num_workers=NUM_WORKERS,
        )

        batch = next(iter(dataloader))
        stats = compute_target_stats(val_dataset, config)

        aggregate_target_count += stats["target_count"]
        aggregate_total_count += stats["total_count"]
        aggregate_unique_raw_sequences.update(
            raw_sequence_key_from_item(item, config)
            for item in val_dataset
        )

        per_file_stat_rows.append({
            "variant": variant,
            "filename": filename,
            "target_proportion": f"{stats['target_proportion']:.6f}",
            "num_unique_raw_sequences": stats["num_unique_raw_sequences"],
        })

        header = f"FILE {file_idx}: {val_name}"
        output_lines.append(f"\n{header}")
        output_lines.append("=" * len(header))
        output_lines.append(summarize_config(config, variant, val_path, val_name))

        output_lines.append("DATASET / BATCH SUMMARY")
        output_lines.append("=======================")
        output_lines.append(f"variant: {variant}")
        output_lines.append(f"root: {root}")
        output_lines.append(f"filename: {filename}")
        output_lines.append(f"num_val_examples: {len(val_dataset)}")
        output_lines.append(f"batch_size: {batch_size}")
        output_lines.append(f"batch_keys: {sorted(batch.keys())}")

        output_lines.append("\nALIGNED BATCH ITEMS")
        output_lines.append("===================")

        for batch_item_idx in range(batch["input_ids"].size(0)):
            rows, input_length, stop = batch_item_rows(
                batch=batch,
                batch_item_idx=batch_item_idx,
                config=config,
                keep_padding=KEEP_PADDING,
            )

            output_lines.append(f"\n--- batch item {batch_item_idx} ---")
            output_lines.append(f"input_length_before_padding: {input_length}")
            output_lines.append(f"printed_length: {stop}")
            output_lines.append(aligned_block(rows))

        output_lines.append("\nREQUESTED INSTANCE-TYPE STATISTICS")
        output_lines.append("==================================")
        output_lines.append(f"variant: {variant}")
        output_lines.append(f"filename: {filename}")
        output_lines.append(f"root: {root}")
        output_lines.append(
            "Target type: negative label, even raw-token length; "
            "does not begin with ')' and does not end with '('."
        )
        output_lines.append("Note: structural raw-token length excludes <eos> and <pad>.")
        output_lines.append(f"target_count: {stats['target_count']}")
        output_lines.append(f"total_count: {stats['total_count']}")
        output_lines.append(f"target_proportion: {stats['target_proportion']:.6f}")
        output_lines.append(
            f"num_unique_raw_sequences: {stats['num_unique_raw_sequences']}"
        )

    aggregate_proportion = (
        aggregate_target_count / aggregate_total_count
        if aggregate_total_count > 0
        else 0.0
    )

    output_lines.append("\nAGGREGATE REQUESTED INSTANCE-TYPE STATISTICS")
    output_lines.append("============================================")
    output_lines.append(f"aggregate_target_count: {aggregate_target_count}")
    output_lines.append(f"aggregate_total_count: {aggregate_total_count}")
    output_lines.append(f"aggregate_target_proportion: {aggregate_proportion:.6f}")
    output_lines.append(
        f"aggregate_num_unique_raw_sequences: {len(aggregate_unique_raw_sequences)}"
    )

    output_lines.append("\nPER-FILE TARGET PROPORTION TABLE")
    output_lines.append("================================")
    output_lines.append(
        formatted_table_with_variant_divider(
            per_file_stat_rows,
            columns=[
                "variant",
                "filename",
                "target_proportion",
                "num_unique_raw_sequences",
            ],
            divider_after_variant="ORIGINAL",
        )
    )

    text = "\n".join(output_lines) + "\n"

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        f.write(text)

    print(text)
    print(f"\nWrote dataloader sanity output to: {OUTPUT}")


if __name__ == "__main__":
    main()

