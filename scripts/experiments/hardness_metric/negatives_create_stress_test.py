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

ROW_WIDTH = 20
ROW_HEIGHT = 2
DPI = 200

COL_GROUP_TITLES = [
    "R_FV",
    "P_FV",
    "RATIO_P",
]

ROW_GROUP_TITLES = [
    "Neg type one",
    "Neg type two",
    "Neg type three",
    "Neg type four",
    "Neg type five",
]

COLORS = {
    "L": "myyellow",
    "rfv": "mygreen",
    "pfv": "mygreen",
    "ratiop": "mygreen",
}

ALPHAS = {
    "L": 1.0,
    "rfv": 0.65,
    "pfv": 0.65,
    "ratiop": 0.65,
}

PICKLE_PATH = "/scratch/yingshac/tmlr/scripts/experiments/hardness_metric"

NUM_SAMPLES_PER_CATEGORY = 5000
SAVE_DIR = "/data/yingshac/tmlr/stress_test_negatives/"
""" 
    <filename>.json = {
        "neg": {
            "l": [s1, s2, ...],
            ...  
        },
        "pos": {}
    }
    Include "neg" as a key to be consistent with the train/val/test format.
"""

# keep track of whether we plot historgram or just print out the count (ax.annotate)
HAS_DISTR = [
    ("two", "p_fv"), ("two", "r_fv"),
    ("three", "p_fv"), ("three", "r_fv"),
    ("four", "p_fv"), ("four", "r_fv"), ("four", "ratio_p"),
    ("five", "p_fv"), ("five", "r_fv"), ("five", "ratio_p"),
]

# ------------------------------------------------------------------------------------------
DATA_ROOT = "/data/yingshac/tmlr/"
NEGATIVES_UNIVERSE_PATHS = [
    f"{DATA_ROOT}/negatives/fix_d_{d}.json" for d in range(2, 9)
] + [
    f"{DATA_ROOT}/d_1/negatives.json"
] + [
    f"{DATA_ROOT}/lengthy_negatives/fix_d_{d}.json" for d in range(2, 9)
] # + [
#     f"{DATA_ROOT}/deeper_tmp/negatives/fix_d_{d}.json" for d in range(9, 17)
# ] + [
#     f"{DATA_ROOT}/deeper_tmp/lengthy_negatives/fix_d_{d}.json" for d in range(9, 17)
# ]

TRAIN_NEGATIVES_PATHS = [
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
RGBA = (235,215,118, 1.0)
mcolors.get_named_colors_mapping()["myyellow"] = (RGBA[0]/255, RGBA[1]/255, RGBA[2]/255, RGBA[3])
RGBA = (57, 132, 83, 0.8)
mcolors.get_named_colors_mapping()["mygreen"] = (RGBA[0]/255, RGBA[1]/255, RGBA[2]/255, RGBA[3])

import numpy as np

# ------------------------------------------------------------------------------------------

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
    
def get_statistics(s):
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


# ------------------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------------------
with open("results/five_neg_types_distr_R_and_P_train_range.json", "r") as f: 
    CATEGORY_TO_RANGE_TRAIN = json.load(f) # load training negatives range from json

# Keys are cross products of 
# Neg Type: ["one", "two", "three", "four", "five"]
# Variable: ["rfv", "pfv", "ratiop"] <-- mapped from ["r_fv", "p_fv", "ratio_p"] by removing "_"
# Side: ["in", "out"] <- if (neg_type, variable) has distribution
SAMPLES = {}
for neg_type in ["one", "two", "three", "four", "five"]:
    for variable in ["r_fv", "p_fv", "ratio_p"]:
        if (neg_type, variable) not in HAS_DISTR: 
            key = f"{neg_type}_{variable.replace('_', '')}"
            SAMPLES[key] = {"data": {}, "L_values": [], f"{variable.replace('_', '')}_values": []}
        else:
            for side in ["in", "out"]:
                key = f"{neg_type}_{variable.replace('_', '')}_{side}"
                SAMPLES[key] = {"data": {}, "L_values": [], f"{variable.replace('_', '')}_values": []}

# For certain categories, force L > 40.
FORCE_L_GT_40_KEYS = [
    "one_rfv", "one_pfv", "one_ratiop",
    "two_ratiop", "three_ratiop", 
    "four_rfv_in", "four_pfv_in", "four_ratiop_in", 
    "five_rfv_in", "five_pfv_in", "five_ratiop_in"
]
# For certain categories, downsample even_L
baseline_delta = 1.0
DOWNSAMPLE_EVEN_L_KEY_TO_DELTA = {
    "one_rfv": 0.25, 
    "one_pfv": 0.25,
    "one_ratiop": 0.25,
    "five_rfv_in": 0.4,
    "five_pfv_in": 0.4,
    "five_ratiop_in": 0.5,
}
DOWNSAMPLE_ODD_L_KEY_TO_DELTA = {"five_ratiop_out": 0.2,}

DOWNSAMPLE_VARIABLE_KEY_TO_RANGE_AND_DELTA = {
    "five_rfv_out": [([-0.03, 0.017], 0.7)],
    "five_pfv_out": [([40, 60], 0.4), ([60, 70], 0.6), ([70, 90], 0.8)],
    "four_rfv_out": [([-0.03, 0.02], 0.4)],
}

for s in tqdm(
    rng.sample(list(available_negatives), k=len(available_negatives)), 
    desc="Computing hardness metrics for negatives"
):

    r_fv, p_fv, ratio_p, neg_type = get_statistics(s)
    L = len(s)

    for variable, value in zip(["r_fv", "p_fv", "ratio_p"], [r_fv, p_fv, ratio_p]):
        if (neg_type, variable) in HAS_DISTR:
            range_train = CATEGORY_TO_RANGE_TRAIN[neg_type][variable]
            if range_train[0] <= value <= range_train[1]: # in
                key = f"{neg_type}_{variable.replace('_', '')}_in"
            else: # out
                key = f"{neg_type}_{variable.replace('_', '')}_out"
        else:
            key = f"{neg_type}_{variable.replace('_', '')}"
        
        if len(SAMPLES[key]["L_values"]) >= NUM_SAMPLES_PER_CATEGORY: 
            continue
        if key in FORCE_L_GT_40_KEYS and L <= 40: continue # Force L > 40 for certain categories

        # Downsample or upsample based on delta
        delta = baseline_delta
        if L%2==0 and key in DOWNSAMPLE_EVEN_L_KEY_TO_DELTA: delta = DOWNSAMPLE_EVEN_L_KEY_TO_DELTA[key]
        if L%2==1 and key in DOWNSAMPLE_ODD_L_KEY_TO_DELTA: delta = DOWNSAMPLE_ODD_L_KEY_TO_DELTA[key]
        if rng.random() > delta: continue 

        drop = False
        if key in DOWNSAMPLE_VARIABLE_KEY_TO_RANGE_AND_DELTA:
            for range_, delta in DOWNSAMPLE_VARIABLE_KEY_TO_RANGE_AND_DELTA[key]:
                if range_[0] <= value <= range_[1] and rng.random() > delta: 
                    drop=True
        elif key == "four_pfv_out":
            if 40 <= value <= 50: delta = min(1.0, L/150)
            elif 50 < value <= 80: delta = min(1.0, L/120)
            elif 80 < value <= 90: delta = min(1.0, L/100)
            if rng.random() > delta: drop=True
        
        if drop: continue

        if L not in SAMPLES[key]["data"]: SAMPLES[key]["data"][L] = []
        SAMPLES[key]["data"][L].append(s)
        SAMPLES[key]["L_values"].append(L)
        SAMPLES[key][f"{variable.replace('_', '')}_values"].append(value)

# Save data to SAVE_DIR
for key in SAMPLES:
    save_path = os.path.join(SAVE_DIR, f"{key}.json")
    with open(save_path, "w") as f:
        json.dump({
            "neg": SAMPLES[key]["data"],
            "pos": {}, # empty pos for consistency with train/val/test format
        }, f, indent=2)
    print(f"Saved {key} to {save_path}")

# Save stats for plotting distributions
for key in SAMPLES:
    SAMPLES[key].pop("data", None) # remove the "data" key to save space
with open(os.path.join(PICKLE_PATH, "stress_test_negatives_stats.pkl"), "wb") as f:
    pickle.dump(SAMPLES, f)
"""
# ------------------------------------------------------------------------------------------
# Load stats from pickle for plotting distributions
with open(os.path.join(PICKLE_PATH, "stress_test_negatives_stats.pkl"), "rb") as f:
    SAMPLES = pickle.load(f)
# ------------------------------------------------------------------------------------------


def plot_for_one_ax(ax, values, key, side, bins):
    color = COLORS[key]
    alpha = ALPHAS[key]

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.hist(values, bins=bins, density=True, color=color, alpha=alpha)
    ax.set_ylabel(f"{side.capitalize()}{key} Density", labelpad=8)

    if not isinstance(bins, int):
        xticks = list(range((values.min()//10+1)*10, values.max() + 1, 10))
        if values.min not in xticks: xticks = [values.min()] + xticks
        if values.max not in xticks: xticks = xticks + [values.max()]
        ax.set_xticks(xticks)
        ax.set_xlim(xticks[0] - 1.5, xticks[-1] + 1.5)
    
    # annotate counts on top right corner
    ax.annotate(
        f"Count: {len(values)}",
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        ha="right", va="top", fontsize=12,
    )
    

# ------------------------------------------------------------------------------------------
num_rows, num_cols = 10, 6
fig, axes = plt.subplots(
    num_rows, num_cols,
    figsize=(ROW_WIDTH, num_rows*ROW_HEIGHT),
    constrained_layout=True
)

for neg_type in ["one", "two", "three", "four", "five"]:
    first_row_idx = ["one", "two", "three", "four", "five"].index(neg_type) * 2
    second_row_idx = first_row_idx + 1
    for variable in ["r_fv", "p_fv", "ratio_p"]:
        first_col_idx = ["r_fv", "p_fv", "ratio_p"].index(variable) * 2
        second_col_idx = first_col_idx + 1
        short_var = variable.replace("_", "")
        if (neg_type, variable) in HAS_DISTR: 
            # two rows, in (variable and L) and out (variable and L)
            for side, row_idx in zip(["in", "out"], [first_row_idx, second_row_idx]):
                for key, col_idx in zip([short_var, "L"], [first_col_idx, second_col_idx]):
                    values = np.array(SAMPLES[f"{neg_type}_{short_var}_{side}"][f"{key}_values"])
                    ax = axes[row_idx, col_idx]
                    bins = np.arange(values.min(), values.max() + 2) - 0.5 if key == "L" else 80
                    plot_for_one_ax(ax, values, key, side+"_", bins)
        else:
            row_idx, col_idx, key = first_row_idx, second_col_idx, "L"
            values = np.array(SAMPLES[f"{neg_type}_{short_var}"][f"{key}_values"])
            ax = axes[row_idx, col_idx]
            bins = np.arange(values.min(), values.max() + 2) - 0.5 if key == "L" else 80
            plot_for_one_ax(ax, values, key, "", bins)
            axes[first_row_idx, first_col_idx].axis("off")
            axes[second_row_idx, first_col_idx].axis("off")
            axes[second_row_idx, second_col_idx].axis("off") 
        

# Fig-level Annotations --------------------------------------------------------------------------------

fig.set_constrained_layout_pads(hspace=0.2, wspace=0.1)
fig.canvas.draw()  # useful if using constrained_layout
x0, x1 = 0.0, 1.0

# Line on top of the first row
top_y = axes[0, 0].get_position().y1 + 0.01
fig.add_artist(Line2D(
    [x0, x1],
    [top_y, top_y],
    transform=fig.transFigure,
    linewidth=2,
    color="mygreen",
))

# Lines between rows
for r in range(1, 8, 2):  # between 5 rows
    upper_row_bottom = axes[r, 0].get_position().y0
    lower_row_top = axes[r + 1, 0].get_position().y1
    y = (upper_row_bottom + lower_row_top) / 2 - 0.005

    fig.add_artist(Line2D(
        [x0, x1],
        [y, y],
        transform=fig.transFigure,
        linewidth=2,
        color="mygreen",
    ))

# Annotate col group titles
for group_idx, group_title in enumerate(COL_GROUP_TITLES):
    
    left_ax = axes[0][2 * group_idx]
    right_ax = axes[0][2 * group_idx + 1]
    left_pos = left_ax.get_position()
    right_pos = right_ax.get_position()

    x_center = (left_pos.x0 + right_pos.x1) / 2
    y_top = max(left_pos.y1, right_pos.y1)

    fig.text(
        x_center,
        y_top + 0.02,
        group_title, ha="center", va="bottom",
        fontsize=14, linespacing=1.75, fontweight="semibold",
    )

# Annotate row group titles
for r, title in zip([0, 2, 4, 6, 8], ROW_GROUP_TITLES):
    upper_row_top = axes[r, 0].get_position().y1
    lower_row_bottom = axes[r + 1, 0].get_position().y0
    y = (upper_row_top + lower_row_bottom) / 2

    fig.text(
        -0.02, y,
        title,
        ha="center", va="center",
        fontsize=14, linespacing=1.75,
        fontweight="semibold", rotation=90,
    )

# ------------------------------------------------------------------------------------------
fig.savefig("./results/negatives_stress_test_distribution.pdf", bbox_inches="tight", dpi=DPI)
plt.close(fig)