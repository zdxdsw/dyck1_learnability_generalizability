"""
Only work on negative samples for now.

Load all negative samples in the universe.
    "/data/yingshac/tmlr/negatives/fix_d_[2-8].json"
    "/data/yingshac/tmlr/d_1/negatives.json"
    "/data/yingshac/tmlr/lengthy_negatives/fix_d_[2-8].json"
    "/data/yingshac/tmlr/deeper_tmp/negatives/fix_d_[9-16].json"
    "/data/yingshac/tmlr/deeper_tmp/lengthy_negatives"/fix_d_[9-16].json"

Exclude training samples. Training negatives are combined from:
    "/data/yingshac/tmlr/train/fix_d_[1-8].json"
    "/data/yingshac/tmlr/shorter/train/fix_d_[1-6].json"
"""

"""
Structure of every json file in the universe:
{
    "l1": [str1, str2, ...],
    "l2": [str1, str2, ...],
    ...
}

Structure of every json file in the training set:
{
    "pos": {
        "l1": [s1, s2, ...],
        ...        },
    "neg": {
        "l1": [s1, s2, ...],
        ...        }
}
"""

DATA_ROOT = "/data/yingshac/tmlr/"
NEGATIVES_UNIVERSE_PATHS = [
    f"{DATA_ROOT}/negatives/fix_d_{d}.json" for d in range(2, 9)
] + [
    f"{DATA_ROOT}/d_1/negatives.json"
] + [
    f"{DATA_ROOT}/lengthy_negatives/fix_d_{d}.json" for d in range(2, 9)
] + [
    f"{DATA_ROOT}/deeper_tmp/negatives/fix_d_{d}.json" for d in range(9, 17)
] + [
    f"{DATA_ROOT}/deeper_tmp/lengthy_negatives/fix_d_{d}.json" for d in range(9, 17)
]

TRAIN_NEGATIVES_PATHS = [
    f"{DATA_ROOT}/train/fix_d_{d}.json" for d in range(1, 9)
] + [
    f"{DATA_ROOT}/shorter/train/fix_d_{d}.json" for d in range(1, 7)
]

import json, os
from tqdm import tqdm
import matplotlib.pyplot as plt 
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from os import cpu_count

def get_first_violation_statistics(s: str):
    """Return (P_FV, R_FV) for a bracket string."""
    if len(s) == 0: raise ValueError("Input string must be nonempty.")
    c_open, c_closed = 0, 0
    p_fv = len(s)
    r_fv = None

    for p, x in enumerate(s):
        if x == "(":
            c_open += 1
        elif x == ")":
            c_closed += 1
        else:
            raise ValueError(f"Unexpected character {x!r} in string {s!r}")

        if c_closed > c_open:
            p_fv = p
            r_fv = 2 * (c_open / (p + 1) - 0.5)
            break

    if r_fv is None: # This corresponds to the <eos> position case.
        r_fv = 2 * (c_open / len(s) - 0.5)

    return p_fv, r_fv, p_fv/len(s)

def collect_negatives_from_paths(paths):
    negatives = set()
    pbar = tqdm(paths, desc="Collecting negatives", total=len(paths))
    for path in paths:
        pbar.set_description(f"Collecting negatives from {path.replace(DATA_ROOT, '')}")
        with open(path, "r") as f:
            data = json.load(f)
            if "neg" in data: data = data["neg"] # For training negatives
            for key in data:
                negatives.update(data[key])
        pbar.update(1)
    return negatives

all_train_negatives = collect_negatives_from_paths(TRAIN_NEGATIVES_PATHS)
all_universe_negatives = collect_negatives_from_paths(NEGATIVES_UNIVERSE_PATHS)

# Exclude training negatives from universe negatives
available_negatives = all_universe_negatives - all_train_negatives
print(f"Total negatives in universe: {len(all_universe_negatives)}")
print(f"Total negatives in training: {len(all_train_negatives)}")
print(f"Total available negatives after exclusion: {len(available_negatives)}")

# Compute and plot the distribution of R_FV for the available negatives in the universe.
# def compute_r_fv(s): return get_first_violation_statistics(s)[1]
# r_fv_values = []
# with ProcessPoolExecutor(max_workers=48) as pool:
#     for r_fv in tqdm(
#         pool.map(compute_r_fv, available_negatives, chunksize=2_000),
#         total=len(available_negatives),
#         desc="Computing R_FV values",
#     ):
#         r_fv_values.append(r_fv)

p_fv_values, r_fv_values, ratio_p_values = [], [], []
with ProcessPoolExecutor(max_workers=48) as pool:
    for p_fv, r_fv, ratio_p in tqdm(
        pool.map(get_first_violation_statistics, available_negatives, chunksize=2_000),
        total=len(available_negatives),
        desc="Computing P_FV, R_FV values",
    ):
        p_fv_values.append(p_fv)
        r_fv_values.append(r_fv)
        ratio_p_values.append(ratio_p)

p_fv_values = np.asarray(p_fv_values)
r_fv_values = np.asarray(r_fv_values)
ratio_p_values = np.asarray(ratio_p_values)
axv = 1/39
fig, axes = plt.subplots(4, 1, figsize=(12, 16))

### 1. Global P_FV histogram
# bins = number of distinct values of P_FV, since P_FV are integers < 120.
distinct_p_fv = set(p_fv_values)
axes[0].hist(p_fv_values, bins=len(distinct_p_fv), density=True,)
axes[0].set_title("P_FV Distribution (Global)")
axes[0].set_xlabel("P_FV")
axes[0].set_ylabel("Density")
axes[0].set_xlim(-0.5, max(distinct_p_fv) + 0.5)

### 2. Global RATIO_P histogram
# filter ratio_p_values where 0 < x < 1, print filter rate
filtered_ratio_p_values = ratio_p_values[(ratio_p_values > 0) & (ratio_p_values < 1)]
print(f"Filtered ratio_p_values: {len(filtered_ratio_p_values)} / {len(ratio_p_values)} = {len(filtered_ratio_p_values)/len(ratio_p_values):.4f}")
axes[1].hist(filtered_ratio_p_values, bins=200, range=(0, 1), density=True,)
axes[1].set_title(f"RATIO_P Distribution (Filtered {100*len(filtered_ratio_p_values)/len(ratio_p_values):.2f}%)")
axes[1].set_xlabel("RATIO_P")
axes[1].set_ylabel("Density")
axes[1].set_xlim(0, 1)

### 3. Global R_FV histogram
axes[2].hist(r_fv_values, bins=100, range=(-1, 1), density=True,)
axes[2].axvline(-axv, color="black", linestyle="--", linewidth=1, alpha=0.5)
axes[2].axvline(axv, color="black", linestyle="--", linewidth=1, alpha=0.5)

axes[2].set_title("R_FV Distribution (Global)")
axes[2].set_xlabel("R_FV")
axes[2].set_ylabel("Density")
axes[2].set_xlim(-1, 1)

### 4. Zoomed R_FV histogram
zoom_left, zoom_right = -0.2, 0.05
# zoom_left = np.min(r_fv_values[r_fv_values > -1])  # min R_FV value that > -1
# zoom_right = np.max(r_fv_values[r_fv_values < 1])  # max R_FV value that < 1
print(f"R_FV zoom range: [{zoom_left:.4f}, {zoom_right:.4f}]")

axes[3].hist(r_fv_values, bins=250, range=(zoom_left, zoom_right), density=True)
axes[3].axvline(-axv, color="black", linestyle="--", linewidth=1, alpha=0.5)
axes[3].axvline(axv, color="black", linestyle="--", linewidth=1, alpha=0.5)

axes[3].set_title(f"R_FV Distribution (Zoomed: [{zoom_left:.4f}, {zoom_right:.4f}])")
axes[3].set_xlabel("R_FV")
axes[3].set_ylabel("Density")
axes[3].set_xlim(zoom_left, zoom_right)

fig.tight_layout()
fig.savefig("./results/r_and_p_fv_distribution.png", dpi=200)
plt.close(fig)