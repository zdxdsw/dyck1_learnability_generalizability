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

ROW_WIDTH = 13.5
ROW_HEIGHT = 2.4
DPI = 200

COL_TITLES = [
    "L <= 40",
    "42 <= L <= 80",
    "82 <= L <= 120",
]

ROW_GROUP_TITLES = [
    "RoD <= 40",
    "40 < RoD <= 80",
    "80 < RoD <= 120",
]

COLORS = {
    "L": "mycolor",
    "RoD": "gold",
}

RGBA = (235,215,118, 1.0)
PICKLE_PATH = "/scratch/yingshac/tmlr/scripts/experiments/hardness_metric"

NUM_SAMPLES_PER_CATEGORY = 10000
SAVE_DIR = "/data/yingshac/tmlr/stress_test_positives/"
""" 
    <filename>.json = {
        "pos": {
            "l": [s1, s2, ...],
            ...  
        }
    }
    Include "pos" as a key to be consistent with the train/val/test format.
"""

GET_FILENAMES = {
    "L <= 40, RoD <= 40": "Lleq40_RoDleq40",
    "42 <= L <= 80, RoD <= 40": "L42to80_RoDleq40",
    "42 <= L <= 80, 40 < RoD <= 80": "L42to80_RoD41to80",
    "82 <= L <= 120, RoD <= 40": "L82to120_RoDleq40",
    "82 <= L <= 120, 40 < RoD <= 80": "L82to120_RoD41to80",
    "82 <= L <= 120, 80 < RoD <= 120": "L82to120_RoD81to120",
}

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

import json, os, random, pickle
rng = random.Random(31)
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
mcolors.get_named_colors_mapping()["mycolor"] = (RGBA[0]/255, RGBA[1]/255, RGBA[2]/255, RGBA[3])

import numpy as np
# from concurrent.futures import ProcessPoolExecutor
# from os import cpu_count

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

SAMPLES = {
    "L <= 40, RoD <= 40": {"data": {}, "L_values": [], "RoD_values": []},
    "42 <= L <= 80, RoD <= 40": {"data": {}, "L_values": [], "RoD_values": []},
    "42 <= L <= 80, 40 < RoD <= 80": {"data": {}, "L_values": [], "RoD_values": []},
    "82 <= L <= 120, RoD <= 40": {"data": {}, "L_values": [], "RoD_values": []},
    "82 <= L <= 120, 40 < RoD <= 80": {"data": {}, "L_values": [], "RoD_values": []},
    "82 <= L <= 120, 80 < RoD <= 120": {"data": {}, "L_values": [], "RoD_values": []},
}

for s in tqdm(
    rng.sample(list(available_positives), k=len(available_positives)), 
    desc="Computing longest RoD values for positives"
):
    # baseline delta: 0.8
    # Downsample RoD = 9, 11, 17 -> delta 0.4
    # Upsample RoD \in [67, 79]; RoD \in [111, 119] -> delta = 1.0
    longest_rod = get_longest_RoD(s)
    if longest_rod in [9, 11, 17]: delta = 0.4
    elif 67 <= longest_rod <= 79 or 111 <= longest_rod <= 119: delta = 1.0
    else: delta = 0.8
    if rng.random() > delta: continue # Downsample or upsample based on delta
    
    L = len(s)
    
    if L <= 40 and longest_rod <= 40:
        if len(SAMPLES["L <= 40, RoD <= 40"]["L_values"]) >= NUM_SAMPLES_PER_CATEGORY:
            continue
        if L not in SAMPLES["L <= 40, RoD <= 40"]["data"]:
            SAMPLES["L <= 40, RoD <= 40"]["data"][L] = []
        SAMPLES["L <= 40, RoD <= 40"]["data"][L].append(s)
        SAMPLES["L <= 40, RoD <= 40"]["L_values"].append(L)
        SAMPLES["L <= 40, RoD <= 40"]["RoD_values"].append(longest_rod)
    
    elif 42 <= L <= 80 and longest_rod <= 40:
        if len(SAMPLES["42 <= L <= 80, RoD <= 40"]["L_values"]) >= NUM_SAMPLES_PER_CATEGORY:
            continue
        if L not in SAMPLES["42 <= L <= 80, RoD <= 40"]["data"]:
            SAMPLES["42 <= L <= 80, RoD <= 40"]["data"][L] = []
        SAMPLES["42 <= L <= 80, RoD <= 40"]["data"][L].append(s)
        SAMPLES["42 <= L <= 80, RoD <= 40"]["L_values"].append(L)
        SAMPLES["42 <= L <= 80, RoD <= 40"]["RoD_values"].append(longest_rod)
    
    elif 42 <= L <= 80 and 40 < longest_rod <= 80:
        if len(SAMPLES["42 <= L <= 80, 40 < RoD <= 80"]["L_values"]) >= NUM_SAMPLES_PER_CATEGORY:
            continue
        if L not in SAMPLES["42 <= L <= 80, 40 < RoD <= 80"]["data"]:
            SAMPLES["42 <= L <= 80, 40 < RoD <= 80"]["data"][L] = []
        SAMPLES["42 <= L <= 80, 40 < RoD <= 80"]["data"][L].append(s)
        SAMPLES["42 <= L <= 80, 40 < RoD <= 80"]["L_values"].append(L)
        SAMPLES["42 <= L <= 80, 40 < RoD <= 80"]["RoD_values"].append(longest_rod)
    
    elif 82 <= L <= 120 and longest_rod <= 40:
        if len(SAMPLES["82 <= L <= 120, RoD <= 40"]["L_values"]) >= NUM_SAMPLES_PER_CATEGORY:
            continue
        if L not in SAMPLES["82 <= L <= 120, RoD <= 40"]["data"]:
            SAMPLES["82 <= L <= 120, RoD <= 40"]["data"][L] = []
        SAMPLES["82 <= L <= 120, RoD <= 40"]["data"][L].append(s)
        SAMPLES["82 <= L <= 120, RoD <= 40"]["L_values"].append(L)
        SAMPLES["82 <= L <= 120, RoD <= 40"]["RoD_values"].append(longest_rod)
    
    elif 82 <= L <= 120 and 40 < longest_rod <= 80:
        if len(SAMPLES["82 <= L <= 120, 40 < RoD <= 80"]["L_values"]) >= NUM_SAMPLES_PER_CATEGORY:
            continue
        if L not in SAMPLES["82 <= L <= 120, 40 < RoD <= 80"]["data"]:
            SAMPLES["82 <= L <= 120, 40 < RoD <= 80"]["data"][L] = []
        SAMPLES["82 <= L <= 120, 40 < RoD <= 80"]["data"][L].append(s)
        SAMPLES["82 <= L <= 120, 40 < RoD <= 80"]["L_values"].append(L)
        SAMPLES["82 <= L <= 120, 40 < RoD <= 80"]["RoD_values"].append(longest_rod)
    
    elif 82 <= L <= 120 and 80 < longest_rod <= 120:
        if len(SAMPLES["82 <= L <= 120, 80 < RoD <= 120"]["L_values"]) >= NUM_SAMPLES_PER_CATEGORY:
            continue
        if L not in SAMPLES["82 <= L <= 120, 80 < RoD <= 120"]["data"]:
            SAMPLES["82 <= L <= 120, 80 < RoD <= 120"]["data"][L] = []
        SAMPLES["82 <= L <= 120, 80 < RoD <= 120"]["data"][L].append(s)
        SAMPLES["82 <= L <= 120, 80 < RoD <= 120"]["L_values"].append(L)
        SAMPLES["82 <= L <= 120, 80 < RoD <= 120"]["RoD_values"].append(longest_rod)
    else:
        raise ValueError(f"Unexpected length {L} and longest RoD {longest_rod} for string {s!r}")

# Save data to SAVE_DIR
for key in SAMPLES:
    filename = GET_FILENAMES[key]
    save_path = os.path.join(SAVE_DIR, f"{filename}.json")
    with open(save_path, "w") as f:
        json.dump({
            "pos": SAMPLES[key]["data"],
            "neg": {}, # empty neg for consistency with train/val/test format
        }, f, indent=2)
    print(f"Saved {key} to {save_path}")

# Save stats for plotting distributions
for key in SAMPLES:
    SAMPLES[key].pop("data", None) # remove the "data" key to save space
with open(os.path.join(PICKLE_PATH, "stress_test_positives_stats.pkl"), "wb") as f:
    pickle.dump(SAMPLES, f)
"""
# ------------------------------------------------------------------------------------------
# Load stats from pickle for plotting distributions
with open(os.path.join(PICKLE_PATH, "stress_test_positives_stats.pkl"), "rb") as f:
    SAMPLES = pickle.load(f)
# ------------------------------------------------------------------------------------------
"""
### Plot distributions of L and RoD for each category
num_rows, num_cols = 6, 3
fig, axes = plt.subplots(
    num_rows, num_cols,
    figsize=(ROW_WIDTH, num_rows*ROW_HEIGHT),
    constrained_layout=True
)

for row_group_idx, RoD_handle in enumerate(
    ["RoD <= 40", "40 < RoD <= 80", "80 < RoD <= 120"]
):
    for col_idx, L_handle in enumerate(
        ["L <= 40", "42 <= L <= 80", "82 <= L <= 120"]
    ):
        key = f"{L_handle}, {RoD_handle}"
        for row_idx, variable in zip(
            [row_group_idx*2, row_group_idx*2 + 1], ["L_values", "RoD_values"]
        ):
            ax = axes[row_idx, col_idx]
            
            if not (row_idx < (col_idx + 1) * 2):
                ax.axis("off") # Only row_idx < (col_idx + 1) * 2 have subplots
                continue
            
            values = np.array(SAMPLES[key][variable])
            bins = np.arange(values.min(), values.max() + 2) - 0.5
            ax.hist(values, bins=bins, density=True, color=COLORS[variable.split("_")[0]], alpha=0.8)
            ax.set_ylabel(f"{variable} Density", labelpad=8)
            xticks = list(range((values.min()//10+1)*10, values.max() + 1, 10))
            if values.min not in xticks: xticks = [values.min()] + xticks
            if values.max not in xticks: xticks = xticks + [values.max()]
            ax.set_xticks(xticks)
            ax.set_xlim(xticks[0] - 1.5, xticks[-1] + 1.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # annotate counts on top right corner
            ax.annotate(
                f"Count: {len(values)}",
                xy=(0.95, 0.95),
                xycoords="axes fraction",
                ha="right", va="top", fontsize=12,
            )

# Col Titles and Row Titles --------------------------------------------------------------------------------
for ax, title in zip(axes[0], COL_TITLES):
    ax.text(
        0.5, 1.15,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=14,
        linespacing=1.75,
        fontweight="semibold",
    )

# ------------------------------------------------------------------------------------------
fig.set_constrained_layout_pads(hspace=0.15, wspace=0.1)
fig.canvas.draw()  # useful if using constrained_layout
x0, x1 = 0.0, 1.0

# Line on top of the first row
top_y = axes[0, 0].get_position().y1 + 0.01
fig.add_artist(Line2D(
    [x0, x1],
    [top_y, top_y],
    transform=fig.transFigure,
    linewidth=2,
    color="gold",
))

# Lines between rows
for r in range(1, 4, 2):  # between 5 rows
    upper_row_bottom = axes[r, 0].get_position().y0
    lower_row_top = axes[r + 1, 0].get_position().y1
    y = (upper_row_bottom + lower_row_top) / 2 - 0.005

    fig.add_artist(Line2D(
        [x0, x1],
        [y, y],
        transform=fig.transFigure,
        linewidth=2,
        color="gold",
    ))

# Annotate row group titles
for r, title in zip([0, 2, 4], ROW_GROUP_TITLES):
    upper_row_top = axes[r, 0].get_position().y1
    lower_row_bottom = axes[r + 1, 0].get_position().y0
    y = (upper_row_top + lower_row_bottom) / 2

    fig.text(
        -0.025, y,
        title,
        ha="center",
        va="center",
        fontsize=14,
        linespacing=1.75,
        fontweight="semibold",
        rotation=90,
    )

# ------------------------------------------------------------------------------------------
fig.savefig("./results/positives_stress_test_distribution.png", bbox_inches="tight", dpi=DPI)
plt.close(fig)