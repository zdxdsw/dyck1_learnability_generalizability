"""
Only work on positive samples for now.

Load all positive samples in the universe.
    "/data/yingshac/tmlr/positives/fix_d_[2-8].json"
    "/data/yingshac/tmlr/d_1/positives.json"
    "/data/yingshac/tmlr/lengthy_positives/fix_d_[2-8].json"
    "/data/yingshac/tmlr/deeper_tmp/positives/fix_d_[9-16].json"
    "/data/yingshac/tmlr/deeper_tmp/lengthy_positives/fix_d_[9-16].json"

Exclude training samples. Training positives are combined from:
    "/data/yingshac/tmlr/train/fix_d_[1-8].json"
    "/data/yingshac/tmlr/shorter/train/fix_d_[1-6].json"
"""

RGBA = (57, 132, 83, 0.5)

# ------------------------------------------------------------------------------------------
DATA_ROOT = "/data/yingshac/tmlr/"
POSITIVES_UNIVERSE_PATHS = [
    f"{DATA_ROOT}/positives/fix_d_{d}.json" for d in range(2, 9)
] + [
    f"{DATA_ROOT}/d_1/positives.json"
] + [
    f"{DATA_ROOT}/lengthy_positives/fix_d_{d}.json" for d in range(2, 9)
]# + [
#     f"{DATA_ROOT}/deeper_tmp/positives/fix_d_{d}.json" for d in range(9, 17)
# ] + [
#     f"{DATA_ROOT}/deeper_tmp/lengthy_positives/fix_d_{d}.json" for d in range(9, 17)
# ]

TRAIN_POSITIVES_PATHS = [
    f"{DATA_ROOT}/train/fix_d_{d}.json" for d in range(1, 9)
] + [
    f"{DATA_ROOT}/shorter/train/fix_d_{d}.json" for d in range(1, 7)
]

PICKLE_PATH = "/scratch/yingshac/tmlr/scripts/experiments/hardness_metric"

import json, os, pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
mcolors.get_named_colors_mapping()["mycolor"] = (RGBA[0]/255, RGBA[1]/255, RGBA[2]/255, RGBA[3])

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from os import cpu_count

def get_longest_RoD(s):
    """ RoD = Range of Dependency
        Iterate throught the string s and simulate a stack. 
        Push the bracket and the position index,
        so that when we pop the stack, we can compute the length of the dependency.
    """

    stack = []
    longest_RoD = 0
    for i, c in enumerate(s):
        if c == "(": stack.append(i)
        elif c == ")":
            if stack:
                start_idx = stack.pop()
                RoD = i - start_idx
                longest_RoD = max(longest_RoD, RoD)
            else: return -1
        else: raise ValueError(f"Unexpected character {c!r} in string {s!r}")

    if stack: return -1

    return longest_RoD

# ------------------------------------------------------------------------------------------

def collect_negatives_from_paths(paths):
    negatives = set()
    pbar = tqdm(paths, desc="Collecting negatives", total=len(paths))
    for path in paths:
        pbar.set_description(f"Collecting negatives from {path.replace(DATA_ROOT, '')}")
        with open(path, "r") as f:
            data = json.load(f)
            if "pos" in data: data = data["pos"] # For training positives
            for key in data:
                negatives.update(data[key])
        pbar.update(1)
    return negatives

all_train_positives = collect_negatives_from_paths(TRAIN_POSITIVES_PATHS)
all_universe_positives = collect_negatives_from_paths(POSITIVES_UNIVERSE_PATHS)

# Exclude training positives from universe positives
available_positives = all_universe_positives - all_train_positives
print(f"Total positives in universe: {len(all_universe_positives)}")
print(f"Total positives in training: {len(all_train_positives)}")
print(f"Total available positives after exclusion: {len(available_positives)}")

# ------------------------------------------------------------------------------------------

longest_rod_values, longest_rod_values_train = [], []
basic_counts = {
    "L <= 40, RoD <= 40": 0,
    "42 <= L <= 80, RoD <= 40": 0,
    "42 <= L <= 80, 40 < RoD <= 80": 0,
    "82 <= L <= 120, RoD <= 40": 0,
    "82 <= L <= 120, 40 < RoD <= 80": 0,
    "82 <= L <= 120, 80 < RoD <= 120": 0,
}

for s in tqdm(available_positives, desc="Computing longest RoD values for positives"):
    longest_rod = get_longest_RoD(s)
    longest_rod_values.append(longest_rod)

    L = len(s)
    if L <= 40 and longest_rod <= 40:
        basic_counts["L <= 40, RoD <= 40"] += 1
    elif 42 <= L <= 80 and longest_rod <= 40:
        basic_counts["42 <= L <= 80, RoD <= 40"] += 1
    elif 42 <= L <= 80 and 40 < longest_rod <= 80:
        basic_counts["42 <= L <= 80, 40 < RoD <= 80"] += 1
    elif 82 <= L <= 120 and longest_rod <= 40:
        basic_counts["82 <= L <= 120, RoD <= 40"] += 1
    elif 82 <= L <= 120 and 40 < longest_rod <= 80:
        basic_counts["82 <= L <= 120, 40 < RoD <= 80"] += 1
    elif 82 <= L <= 120 and 80 < longest_rod <= 120:
        basic_counts["82 <= L <= 120, 80 < RoD <= 120"] += 1
    else:
        raise ValueError(f"Unexpected length {L} and longest RoD {longest_rod} for string {s!r}")

for s in tqdm(all_train_positives, desc="Computing longest RoD values for training positives"):
    longest_rod = get_longest_RoD(s)
    longest_rod_values_train.append(longest_rod)

# pickle CATEGORY_TO_STATS so that I can play around with the plots without loading the negatives again
with open(os.path.join(PICKLE_PATH, "longest_rod_values_dleq8.pkl"), "wb") as f: 
    pickle.dump(longest_rod_values, f)
with open(os.path.join(PICKLE_PATH, "longest_rod_values_train.pkl"), "wb") as f: 
    pickle.dump(longest_rod_values_train, f)
with open("results/longest_rod_distribution_basic_counts_dleq8.json", "w") as f:
    json.dump(basic_counts, f, indent=4)
"""
# ------------------------------------------------------------------------------------------
# load CATEGORY_TO_STATS from pickle
with open(os.path.join(PICKLE_PATH, "longest_rod_values_dleq8.pkl"), "rb") as f: longest_rod_values = pickle.load(f)
with open(os.path.join(PICKLE_PATH, "longest_rod_values_train.pkl"), "rb") as f: longest_rod_values_train = pickle.load(f)
print("Loaded CATEGORY_TO_STATS from pickle.")
with open("results/longest_rod_distribution_basic_counts_dleq8.json", "r") as f: basic_counts = json.load(f)
"""
# ------------------------------------------------------------------------------------------

longest_rod_values = np.asarray(longest_rod_values)
longest_rod_values_train = np.asarray(longest_rod_values_train)
fig, axes = plt.subplots(2, 1, figsize=(6, 6))

# longest_rod values are all integers. One bin per integer
for ax, values, title in zip(axes, [longest_rod_values, longest_rod_values_train], ["Longest RoD Distribution", "Longest RoD Distribution in Training"]):
    bins = np.arange(longest_rod_values.min(), longest_rod_values.max() + 2) - 0.5
    ax.hist(values, bins=bins, density=True, color="mycolor")
    ax.set_title(title)
    ax.set_ylabel("Density")
    # xticks = min + [20, 40, ..., 100] + max
    xticks = [longest_rod_values.min()] + list(range(20, 101, 20)) + [longest_rod_values.max()]
    ax.set_xticks(xticks)
    ax.set_xlim(longest_rod_values.min() - 1.5, longest_rod_values.max() + 1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# annotate basic counts on axes[0]
print(basic_counts)
for key, value in basic_counts.items():
    axes[0].annotate(
        f"{key}: {value}",
        xy=(0.95, 0.95 - 0.08 * list(basic_counts.keys()).index(key)),
        xycoords="axes fraction",
        ha="right", va="top", fontsize=10,
    )

fig.tight_layout()
fig.savefig("./results/longest_rod_distribution_dleq8.png", dpi=200)
plt.close(fig)