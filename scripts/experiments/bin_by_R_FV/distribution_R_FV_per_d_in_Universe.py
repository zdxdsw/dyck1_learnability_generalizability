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

    return p_fv, r_fv


def compute_r_fv(s): return get_first_violation_statistics(s)[1]


def collect_negatives_from_paths(paths):
    negatives = set()
    # pbar = tqdm(paths, desc="Collecting negatives", total=len(paths))
    for path in paths:
        # pbar.set_description(f"Collecting negatives from {path.replace(DATA_ROOT, '')}")
        with open(path, "r") as f:
            data = json.load(f)
            if "neg" in data: data = data["neg"] # For training negatives
            for key in data:
                negatives.update(data[key])
        # pbar.update(1)
    return negatives


def collect_paths_per_d(paths, d):
    results = []
    
    for path in paths:
        if "/d_1/" in path: x = 1
        else: 
            # pattern: "fix_d_{x}.json" -> extract x and check if it matches d
            x = int(os.path.basename(path).split("fix_d_")[-1].split(".json")[0])

        if x == d: results.append(path)
    
    return results


def plot(r_fv_values, save_path):
    r_fv_values = np.asarray(r_fv_values)
    axv = 1/39
    ### 1. Global fixed-width histogram
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].hist(r_fv_values, bins=100, range=(-1, 1), density=True,)
    axes[0].axvline(-axv, color="black", linestyle="--", linewidth=1, alpha=0.5)
    axes[0].axvline(axv, color="black", linestyle="--", linewidth=1, alpha=0.5)

    axes[0].set_title("R_FV Distribution (Global)")
    axes[0].set_xlabel("R_FV")
    axes[0].set_ylabel("Density")
    axes[0].set_xlim(-1, 1)

    ### 2. Zoomed fixed-width histogram
    # zoom_left, zoom_right = -0.2, 0.05
    zoom_left = np.min(r_fv_values[r_fv_values > -1])  # min R_FV value that > -1
    zoom_right = np.max(r_fv_values[r_fv_values < 1])  # max R_FV value that < 1
    tqdm.write(f"R_FV zoom range: [{zoom_left:.4f}, {zoom_right:.4f}]")

    axes[1].hist(r_fv_values, bins=250, range=(zoom_left, zoom_right), density=True)
    axes[1].axvline(-axv, color="black", linestyle="--", linewidth=1, alpha=0.5)
    axes[1].axvline(axv, color="black", linestyle="--", linewidth=1, alpha=0.5)

    axes[1].set_title(f"R_FV Distribution (Zoomed: [{zoom_left:.4f}, {zoom_right:.4f}])")
    axes[1].set_xlabel("R_FV")
    axes[1].set_ylabel("Density")
    axes[1].set_xlim(zoom_left, zoom_right)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

def process_and_plot_one_d(d):
    tqdm.write("\n" + "=" * 20 + f" Working on d={d} " + "=" * 20)
    
    train_negative_paths = collect_paths_per_d(TRAIN_NEGATIVES_PATHS, d)
    universe_negative_paths = collect_paths_per_d(NEGATIVES_UNIVERSE_PATHS, d)
    tqdm.write(f"\nFound {len(train_negative_paths)} training paths, {len(universe_negative_paths)} universe paths for d={d}.")
    all_train_negatives = collect_negatives_from_paths(train_negative_paths)
    all_universe_negatives = collect_negatives_from_paths(universe_negative_paths)

    # Exclude training negatives from universe negatives
    available_negatives = all_universe_negatives - all_train_negatives
    tqdm.write(f"Total negatives in universe: {len(all_universe_negatives)}")
    tqdm.write(f"Total negatives in training: {len(all_train_negatives)}")
    tqdm.write(f"Total available negatives after exclusion: {len(available_negatives)}")

    # Compute and plot the distribution of R_FV for the available negatives in the universe.
    r_fv_values = []
    with ProcessPoolExecutor(max_workers=24) as pool:
        for r_fv in tqdm(
            pool.map(compute_r_fv, available_negatives, chunksize=50_000),
            total=len(available_negatives),
            desc="Computing R_FV values",
        ):
            r_fv_values.append(r_fv)

    save_path = f"./results/r_fv_distribution_d_{d}.png"
    plot(r_fv_values, save_path)

if __name__ == "__main__":
    pgbar = tqdm(range(1, 17), desc="Compute distribution of R_FV per d")
    for d in pgbar: 
        # Use tqdm.write() instead of print() for anything printed 
        # while a progress bar may still be active.
        pgbar.set_description(f"Compute distribution of R_FV per d (d={d})")
        process_and_plot_one_d(d)
        tqdm.write("\n")
        