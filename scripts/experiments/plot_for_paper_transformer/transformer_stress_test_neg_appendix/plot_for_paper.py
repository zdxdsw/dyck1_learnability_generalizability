from typing import Literal
import os, json, argparse, re
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

PROJECT_ROOT = "/home/yingshac/workspace/tmlr/scripts/experiments"

PLOT_STATS_PATHS = [ # three columns
    "single_metric_plot_transformer/stress_test_neg_singletask/S1_d_trulyleq[2-8]_2ABCDEFG/S1_d_trulyleq8_2ABCDEFG_acc.json",
    # "single_metric_plot_transformer/stress_test_neg_singletask/S3_d_trulyleq[2-8]_2ABCDEFG/S3_d_trulyleq8_2ABCDEFG_acc.json",
]

COL_TITLES = [
    "In/out according to RFV",
    "In/out according to PFV",
    "In/out according to RatioP",
]

ROW_TITLES = [
    r"$\ \ \ $Supervision: SL1",
    # "Supervision: SL1 + SL3",
]

YLIM = (0.32, 1.05)
ROW_WIDTH = 12
ROW_HEIGHT = 2.8
DPI = 300


# Constants --------------------------------------------------------------------------------

NOISE_STD = 0.1 # for jittering x values for better visualization

def SPLITNAME_TO_XTICKLABEL(split_name: str) -> str:
    x = split_name.split("_")
    
    if split_name.startswith("one_"):
        return "Type1"
    
    elif split_name.startswith("two_"):
        if len(x) == 3:
            side = x[-1]
            return "Type2-" + side.capitalize()
        else: return "Type2"
    elif split_name.startswith("three_"):
        if len(x) == 3:
            side = x[-1]
            return "Type3-" + side.capitalize()
        else: return "Type3"
    elif split_name.startswith("four_"):
        if len(x) == 3:
            side = x[-1]
            return "Type4-" + side.capitalize()
        else: return "Type4"
    elif split_name.startswith("five_"):
        if len(x) == 3:
            side = x[-1]
            return "Type5-" + side.capitalize()
        else: return "Type5"
    
    else: raise ValueError(f"Unexpected split_name: {split_name}")

SPLITNAME_TO_X = {
    "one_rfv": 0.5,
    "two_rfv_in": 2,
    "two_rfv_out": 3,
    "three_rfv_in": 4,
    "three_rfv_out": 5,
    "four_rfv_in": 6,
    "four_rfv_out": 7,
    "five_rfv_in": 8,
    "five_rfv_out": 9,
    "one_pfv": 0.5,
    "two_pfv_in": 2,
    "two_pfv_out": 3,
    "three_pfv_in": 4,
    "three_pfv_out": 5,
    "four_pfv_in": 6,
    "four_pfv_out": 7,
    "five_pfv_in": 8,
    "five_pfv_out": 9,
    "one_ratiop": 1.5,
    "two_ratiop": 3,
    "three_ratiop": 4.5,
    "four_ratiop_in": 6,
    "four_ratiop_out": 7,
    "five_ratiop_in": 8,
    "five_ratiop_out": 9,
}

SHORT_VAR_TO_ROW_IDX = {
    "rfv": 0,
    "pfv": 1,
    "ratiop": 2,
}

PE_CODE_TO_NAME = {
	"A": "rotary",
	"B": "scaler",
	"C": "sinusoidal",
	"D": "absolute",
	"E": "alibi",
	"F": "relative",
	"G": "none",
    "H": "rotary",
	"I": "scaler",
	"J": "sinusoidal",
	"K": "absolute",
	"L": "alibi",
	"M": "relative",
    "O": "rotary",
	"P": "scaler",
	"Q": "sinusoidal",
	"R": "absolute",
	"S": "alibi",
	"T": "relative",
	"U": "none",
}

PE_NAME_TO_COLOR = {
	"rotary": "firebrick",
	"scaler": "blue",
	"sinusoidal": "limegreen",
	"absolute": "darkorchid",
	"alibi": "deepskyblue",
	"relative": "goldenrod",
	"none": "hotpink"
}

PE_NAME_TO_PLOT_ORDER = {
	"rotary": 0,
	"scaler": 1,
	"sinusoidal": 2,
	"absolute": 3,
	"alibi": 4,
	"relative": 5,
	"none": 6
}


def plot_for_one_row(row_idx, row_axes, plot_stats):
    for pe_name in plot_stats:
        X, Y, Y_err, Xticklabels = [[], [], []], [[], [], []], [[], [], []], [[], [], []] # three cols
        for split_name in plot_stats[pe_name]:
            short_var = split_name.split("_")[1] # e.g. "rfv", "pfv", "ratiop"
            col_idx = SHORT_VAR_TO_ROW_IDX[short_var]
            x = SPLITNAME_TO_X[split_name]
            X[col_idx].append(x)
            Y[col_idx].append(plot_stats[pe_name][split_name]["mean"])
            Y_err[col_idx].append(plot_stats[pe_name][split_name]["std"])
            Xticklabels[col_idx].append(SPLITNAME_TO_XTICKLABEL(split_name))
        for x, y, yerr, ax in zip(X, Y, Y_err, row_axes):
            # add noise to x via np.random.normal(0, args.noise_std)
            x = np.asarray(x, dtype=np.float64)
            x += np.random.normal(0, NOISE_STD, size=x.shape)
            y = np.asarray(y)
            yerr = np.asarray(yerr)

            # clip y values to [0, 1] for better visualization
            lower = np.clip(y - yerr, YLIM[0]+0.01, YLIM[1])
            upper = np.clip(y + yerr, YLIM[0]+0.01, YLIM[1])
            yerr_clipped = np.vstack([
                y - lower,   # distance from marker to lower cap
                upper - y,   # distance from marker to upper cap
            ])
            # markers only: more opaque
            ax.scatter(
                x, y,
                facecolors='none', edgecolors=PE_NAME_TO_COLOR[pe_name],
                marker='s', linewidths=1.2, s=50, alpha=0.65,
                label=pe_name + r"$\qquad$" if pe_name == "none" else pe_name,
            )
            ax.errorbar(
                x, y,
                yerr=yerr_clipped,       # vertical error bar size
                fmt='none', ecolor=PE_NAME_TO_COLOR[pe_name], # error bar color
                capsize=4, capthick=1.2,
                elinewidth=1.2, linestyle='none', alpha=0.15
            )

        if pe_name == "none":
            for col_idx, ax, x, xticklabels in zip(range(len(row_axes)), row_axes, X, Xticklabels):
                # xlim: min(x)-1, max(x)+1
                ax.set_xlim(min(x)-1, max(x)+1)
                ax.set_ylim(YLIM[0], YLIM[1])
                if row_idx == 0:
                    ax.set_xticks(x)
                    ax.set_xticklabels(xticklabels, rotation=35, ha='right', fontsize=11)
                else: # no xticks
                    ax.set_xticks([])
                    ax.set_xticklabels([])
                
                if col_idx == 0: ax.set_ylabel("n_acc", labelpad=8, fontweight="semibold", fontsize=12)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

# ------------------------------------------------------------------------------------------

num_rows, num_cols = 1, 3
fig, axes = plt.subplots(
    num_rows, num_cols,
    figsize=(ROW_WIDTH, num_rows*ROW_HEIGHT),
    constrained_layout=True,
)

# fig.set_constrained_layout_pads(wspace=0.05)
row_idx, row_axes = 0, axes
path = PLOT_STATS_PATHS[row_idx]
with open(os.path.join(PROJECT_ROOT, path), "r") as f: 
    plot_stats = json.load(f)
print(f"Processing {os.path.basename(path)} ...")
plot_for_one_row(row_idx, row_axes, plot_stats)


# Col Titles and Row Titles --------------------------------------------------------------------------------
for ax, title in zip(axes, COL_TITLES):
    ax.text(
        0.47, 1.1,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        linespacing=1.6,
        # fontweight="semibold",
    )

# Fig-level legend
handles, labels = axes.flat[0].get_legend_handles_labels()
title_handle = Line2D([], [], linestyle="none", marker=None)
fig.legend(
    [title_handle] + handles, 
    ["Hidden Size:"] + labels, # "fake" legend title on the left
    ncol=8, fontsize=12,
    loc="lower center", bbox_to_anchor=(0.5, -0.2),
    columnspacing=1.2, handletextpad=0.05,
)

# ------------------------------------------------------------------------------------------
save_path = os.path.join(os.path.basename(os.getcwd()) + ".pdf") # figure name = basename of cwd
fig.suptitle(r"Supervision: SL1; Eval on: SL1.$\ \ \ $Cumu. $d^{tr}_{\max}=8.\ \ \ \ $Stress-Testing Negatives by Type.", fontsize=12, fontweight="semibold", y=1.12)
fig.savefig(save_path, bbox_inches="tight", dpi=DPI)