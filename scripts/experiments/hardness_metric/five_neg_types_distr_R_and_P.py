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

ROW_WIDTH = 16
ROW_HEIGHT = 3.2
DPI = 300

COL_TITLES = [
    "R_FV distribution",
    "P_FV distribution",
    "Ratio_P distribution",
]

ROW_TITLEs = [
    "Neg type one",
    "Neg type two",
    "Neg type three",
    "Neg type four",
    "Neg type five",
]

PICKLE_PATH = "/scratch/yingshac/tmlr/scripts/experiments/hardness_metric/CATETORY_TO_STATS.pkl"

RGBA = (57, 132, 83, 0.5)

# ------------------------------------------------------------------------------------------
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

import json, os, pickle
from tqdm import tqdm
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
mcolors.get_named_colors_mapping()["mycolor"] = (RGBA[0]/255, RGBA[1]/255, RGBA[2]/255, RGBA[3])

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from os import cpu_count

"""
    Categorize the negative samples in the universe into 5 types:

    General expression:  S1  |  ')'  |  S2  |  '('  |  S3 
    S1, S2, S3 can be either empty, or a valid string.
    Let's call ')' eo (erroneous open), and '(' ec (erroneous close).
        We can toggle their existence. They can't be both absent.
        

    1. Ratio_P == 0 (i.e. p_fv == 0)
        first token is ')'
        r_fv == -1 must be true
        S1 == ""
    
    2. Ratio_P == 1 (i.e. p_fv == len(s)), but last token is not '('
        r_fv > 0
        S1 != "", S3 == "", ec == "no"
    
    3. Ratio_P == 1 (i.e. p_fv == len(s)), and last token is '('
        r_fv > 0
        S1 != "", S3 == "", ec == "yes"
    
    4. 0 < Ratio_P < 1, and last token is '('
        r_fv < 0
        S1 != "", S3 != "", ec == "no"
    
    5. The rest
        0 < Ratio_P < 1
        r_fv < 0
        S1 != "", S3 != "", ec == "yes"

"""
def get_neg_type(ratio_p, r_fv, first_token, last_token):
    """
        r_fv and first_token are here to double check.
        neg_type is a function of ratio_p and last_token only.

        Return: one of ["one", "two", "three", "four", "five"]
    """
    if ratio_p == 0:
        assert r_fv == -1
        assert first_token == ")"
        return "one"
    elif ratio_p == 1 and last_token != "(":
        assert r_fv > 0
        return "two"
    elif ratio_p == 1 and last_token == "(":
        assert r_fv > 0
        return "three"
    elif last_token == "(":
        assert 0 < ratio_p < 1
        assert r_fv < 0
        return "four"
    else:
        assert 0 < ratio_p < 1
        assert r_fv < 0
        return "five"
    
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

    ratio_p = p_fv/len(s)
    neg_type = get_neg_type(ratio_p, r_fv, s[0], s[-1])

    return (r_fv, p_fv, ratio_p, neg_type)
"""
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
"""

"""
    For each neg type, collect a list of r_fv, p_fv and ratio_p values.
    
    Plot histogram for these values for each category.
    
    For some category, certain variables always have the same value, 
        so we skip plotting histogram, but print out the count.
        Type two and three only have distributions for p_fv and r_fv.
        Type four and five have distributions fo all. 
"""
CATETORY_TO_STATS = {
    "one": {"p_fv": [], "r_fv": [], "ratio_p": []},
    "two": {"p_fv": [], "r_fv": [], "ratio_p": []},
    "three": {"p_fv": [], "r_fv": [], "ratio_p": []},
    "four": {"p_fv": [], "r_fv": [], "ratio_p": []},
    "five": {"p_fv": [], "r_fv": [], "ratio_p": []},
}

# keep track of whether we plot historgram or just print out the count (ax.annotate)
HAS_PLOT = [
    ("two", "p_fv"), ("two", "r_fv"),
    ("three", "p_fv"), ("three", "r_fv"),
    ("four", "p_fv"), ("four", "r_fv"), ("four", "ratio_p"),
    ("five", "p_fv"), ("five", "r_fv"), ("five", "ratio_p"),
]

"""
# ------------------------------------------------------------------------------------------
with ProcessPoolExecutor(max_workers=48) as pool:
    for r_fv, p_fv, ratio_p, neg_type in tqdm(
        pool.map(get_first_violation_statistics, available_negatives, chunksize=2_000),
        total=len(available_negatives),
        desc="Computing P_FV, R_FV values",
    ):
        CATETORY_TO_STATS[neg_type]["r_fv"].append(r_fv)
        CATETORY_TO_STATS[neg_type]["p_fv"].append(p_fv)
        CATETORY_TO_STATS[neg_type]["ratio_p"].append(ratio_p)

# pickle CATETORY_TO_STATS so that I can play around with the plots without loading the negatives again
with open(PICKLE_PATH, "wb") as f: pickle.dump(CATETORY_TO_STATS, f)
"""

# ------------------------------------------------------------------------------------------
# load CATETORY_TO_STATS from pickle
with open(PICKLE_PATH, "rb") as f: CATETORY_TO_STATS = pickle.load(f)
print("Loaded CATETORY_TO_STATS from pickle.")
# ------------------------------------------------------------------------------------------
def plot_for_one_category(ax_r, ax_p, ax_ratio_p, neg_type, stats):
    p_fv_values = np.asarray(stats["p_fv"])
    r_fv_values = np.asarray(stats["r_fv"])
    ratio_p_values = np.asarray(stats["ratio_p"])
    assert len(p_fv_values) == len(r_fv_values) == len(ratio_p_values)

    for ax, values, variable in zip(
        [ax_r, ax_p, ax_ratio_p],
        [r_fv_values, p_fv_values, ratio_p_values],
        ["r_fv", "p_fv", "ratio_p"],
    ):  
        
        if (neg_type, variable) in HAS_PLOT:  
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)          
            if variable == "r_fv":
                ax.hist(values, bins=100, range=(values.min(), values.max()), density=True, color="mycolor")
            elif variable == "p_fv":
                # bins = max(p_fv) - min(p_fv) + 1
                bins = np.arange(p_fv_values.min(), p_fv_values.max() + 1) - 0.5
                ax.hist(values, bins=bins, density=True, color="mycolor")
            elif variable == "ratio_p":
                ax.hist(values, bins=200, range=(0, 1), density=True, color="mycolor")
            # annotate the min and max value on the top left corner
            ax.annotate(
                f"Min: {values.min():.4f}\nMax: {values.max():.4f}",
                xy=(0.1, 0.9), xycoords="axes fraction",
                ha="left", va="top", fontsize=14
            )
        else: 
            ax.axis("off")
            single_value = set(values)
            assert len(single_value) == 1
            ax.annotate(
                f"{variable} = {single_value.pop()}", 
                xy=(0.9, 0.8), xycoords="axes fraction", 
                ha="right", va="top", fontsize=14
            ) 
        # annotate count on top right corner
        ax.annotate(
            f"Count: {len(values)}", 
            xy=(0.9, 0.9), xycoords="axes fraction", 
            ha="right", va="top", fontsize=14
        ) 

# ------------------------------------------------------------------------------------------
fig, axes = plt.subplots(
    5, 3,
    figsize=(ROW_WIDTH, 5*ROW_HEIGHT),
    constrained_layout=True
)

for neg_type in ["one", "two", "three", "four", "five"]:
    stats = CATETORY_TO_STATS[neg_type]
    row_idx = ["one", "two", "three", "four", "five"].index(neg_type)
    ax_r, ax_p, ax_ratio_p = axes[row_idx]
    plot_for_one_category(ax_r, ax_p, ax_ratio_p, neg_type, stats)

# Col Titles and Row Titles --------------------------------------------------------------------------------
for ax, title in zip(axes[0], COL_TITLES):
    ax.text(
        0.5, 1.08,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=14,
        linespacing=1.75,
        fontweight="semibold",
    )
for ax, title in zip(axes[:, 0], ROW_TITLEs):
    ax.text(
        -0.15, 0.5,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=14,
        linespacing=1.75,
        fontweight="semibold",
        rotation=90,
    )

# ------------------------------------------------------------------------------------------
fig.set_constrained_layout_pads(hspace=0.15)
fig.canvas.draw()  # useful if using constrained_layout
x0, x1 = 0.0, 1.0

# Line on top of the first row
top_y = axes[0, 0].get_position().y1
fig.add_artist(Line2D(
    [x0, x1],
    [top_y, top_y],
    transform=fig.transFigure,
    linewidth=1,
    color="black",
))

# Lines between rows
for r in range(4):  # between 5 rows
    upper_row_bottom = axes[r, 0].get_position().y0
    lower_row_top = axes[r + 1, 0].get_position().y1
    y = (upper_row_bottom + lower_row_top) / 2 - 0.005

    fig.add_artist(Line2D(
        [x0, x1],
        [y, y],
        transform=fig.transFigure,
        linewidth=1,
        color="black",
    ))

save_path = "results/five_neg_types_distr_R_and_P.pdf"
fig.savefig(save_path, bbox_inches="tight", dpi=DPI)