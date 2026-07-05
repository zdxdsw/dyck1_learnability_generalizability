import os, json
from matplotlib import pyplot as plt
import numpy as np


PLOT_STATS = [
    [ # first row
        "dg_larger(S2_acc)/S2_d_trulyleq[2-8]_2ABCDEFG_dg_larger",
        "lg(S2_acc)/S2_d_trulyleq[2-8]_2ABCDEFG_lg",
    ], 
    [ # second row
        "dg_larger(S2_p_acc)/S2_d_trulyleq[2-8]_2ABCDEFG_dg_larger",
        "lg(S2_p_acc)/S2_d_trulyleq[2-8]_2ABCDEFG_lg",
    ], 
    [ # third row
        "dg_larger(S2_n_acc)/S2_d_trulyleq[2-8]_2ABCDEFG_dg_larger",
        "lg(S2_n_acc)/S2_d_trulyleq[2-8]_2ABCDEFG_lg",
    ],
]

ROW_TITLEs = [
    "",
    "p_acc",
    "n_acc",
]

MODEL = "transformer"
PLOT_STATS_DIR = f"/home/yingshac/workspace/tmlr/scripts/experiments/single_metric_plot_{MODEL}"
YLIM = (0.58, 1.0)
ROW_WIDTH = 7
ROW_HEIGHT = 2.6
DPI = 300

# Constants --------------------------------------------------------------------------------

DMIN, DMAX = 2, 8

METRIC_SHORTHAND_TO_FULL = {
    "cvg": "Convergence", 
    "pg": "Permutation Generalization", 
    "dg_larger": "Depth Generalization",
    "dg_smaller": "Depth Generalization (towards smaller)", # only available for non-cumu
    "lg": "Length Generalization",
}

RNN_LEGEND_NAME_TO_COLOR = {
	"hs=64": "firebrick",
	"hs=32": "blue",
	"hs=16": "darkorchid",
	"hs=8": "deepskyblue",
	"hs=6": "goldenrod",
    "hs=4": "hotpink",
}

RNN_LEGEND_NAME_TO_PLOT_ORDER = {
	"hs=64": 0,
	"hs=32": 1,
	"hs=16": 2,
	"hs=8": 3,
	"hs=6": 4,
	"hs=4": 5,
}

TRANSFORMER_LEGEND_NAME_TO_COLOR = {
	"rotary": "firebrick",
	"scaler": "blue",
	"sinusoidal": "limegreen",
	"absolute": "darkorchid",
	"alibi": "deepskyblue",
	"relative": "goldenrod",
	"none": "hotpink"
}

TRANSFORMER_LEGEND_NAME_TO_PLOT_ORDER = {
	"rotary": 0,
	"scaler": 1,
	"sinusoidal": 2,
	"absolute": 3,
	"alibi": 4,
	"relative": 5,
	"none": 6
}

# ------------------------------------------------------------------------------------------

LEGEND_NAME_TO_COLOR = RNN_LEGEND_NAME_TO_COLOR if MODEL == "rnn" else TRANSFORMER_LEGEND_NAME_TO_COLOR
LEGEND_NAME_TO_PLOT_ORDER = RNN_LEGEND_NAME_TO_PLOT_ORDER if MODEL == "rnn" else TRANSFORMER_LEGEND_NAME_TO_PLOT_ORDER

# delete empty rows in PLOT_STATS
PLOT_STATS = [row for row in PLOT_STATS if len(row) > 0]

# pass ax by reference to plot on the same figure
def plot_one_ax(plot_stats_filename, ax, row_idx, col_idx):
    
    with open(os.path.join(PLOT_STATS_DIR, plot_stats_filename + ".json"), "r") as f: 
        plot_stats = json.load(f)
    
    ### parse filename
    mode = "cumu" if "trulyleq" in plot_stats_filename else "non-cumu"
    metric = "_".join(plot_stats_filename.split("/")[-1].split("_")[4:])

    x_label = r"$d^{tr}_{\max}$ (cumu)" if mode == "cumu" else r"$d^{tr}$ (non-cumu)"
    y_label = METRIC_SHORTHAND_TO_FULL[metric]

    # plot error band
    for legend_name in plot_stats:
        x = []
        y = []
        y_err = []
        for train_d in sorted(plot_stats[legend_name].keys()):
            stats = plot_stats[legend_name][train_d]
            x.append(int(train_d))
            y.append(stats["mean"])
            y_err.append(stats["std"])
        ax.plot(x, y, label=legend_name, color=LEGEND_NAME_TO_COLOR[legend_name], marker='.', alpha=0.5, linewidth=1.8)
        upper = np.minimum(np.array(y)+np.array(y_err), YLIM[1]) # clip upper bound to YLIM[1]
        lower = np.maximum(np.array(y)-np.array(y_err), YLIM[0]) # clip lower bound to YLIM[0]
        ax.fill_between(x, lower, upper, color=LEGEND_NAME_TO_COLOR[legend_name], alpha=0.1)

    ax.tick_params(axis="both", labelsize=10)
    if row_idx == num_rows - 1: ax.set_xlabel(x_label, fontsize=12, labelpad=6)
    ax.set_ylabel(y_label, fontsize=12, labelpad=6)
    
    ax.set_xticks(list(range(DMIN, DMAX+1)))

    if col_idx == 0:
        ax.set_ylim(YLIM[0]-0.01, YLIM[1]+0.01)
        # yticks: ymax, ymin, step=-0.2, then reverse the list to get ascending order
        ax.set_yticks(list(np.arange(YLIM[1], YLIM[0]-0.01, step=-0.20))[::-1])
    else:
        ax.set_ylim(0.84, 1.01)
        ax.set_yticks(list(np.arange(1.0, 0.84, step=-0.05))[::-1])
    if col_idx == row_idx == 0: 
        ax.legend(loc="lower right", framealpha=0.4, title="Positional Encoding")
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ------------------------------------------------------------------------------------------
num_rows = len(PLOT_STATS)
num_cols = len(PLOT_STATS[0])
for row in PLOT_STATS: # check all rows in PLOT_STATS have the same number of columns
    assert len(row) == num_cols, "All rows in PLOT_STATS must have the same number of columns"

save_path = os.path.join(os.path.basename(os.getcwd()) + ".pdf") # figure name = basename of cwd

fig, axes = plt.subplots(
    num_rows, num_cols, 
    figsize=(ROW_WIDTH, num_rows*ROW_HEIGHT),
    constrained_layout=True
)
fig.set_constrained_layout_pads(
    w_pad=0.04167,
    h_pad=0.04167,
    wspace=0.08,
    hspace=0.02,
)
for row in range(num_rows):
    for col in range(num_cols):
        plot_stats_filename = PLOT_STATS[row][col]
        ax = axes[row, col] if num_rows > 1 else axes[col] # if only one row, axes is 1D array
        if not plot_stats_filename: 
            ax.axis("off")
            continue # skip empty cells
        plot_one_ax(os.path.join(PLOT_STATS_DIR, plot_stats_filename), ax, row_idx=row, col_idx=col)

# Col Titles and Row Titles --------------------------------------------------------------------------------
for ax, title in zip(axes[:, 0], ROW_TITLEs):
    ax.text(
        -0.3, 0.5,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        linespacing=1.75,
        fontweight="semibold",
        rotation=90,
    )

# set fig super title: "Supervision: S1 Eval on: S1"
fig.suptitle(r"Supervision: S1$\qquad\quad\ $; Eval on: $\quad\ \ $.$\ \ \ $2-Layer; Causal.", fontsize=12, fontweight="bold", y=1.04)

fig.text(
    0.36, 1.04,
    "+ S2",
    color="red",
    ha="left",
    va="top",
    fontsize=12,
    fontweight="semibold",
)
fig.text(
    0.57, 1.04,
    "S2",
    color="red",
    ha="left",
    va="top",
    fontsize=12,
    fontweight="semibold",
)

fig.savefig(save_path, bbox_inches="tight", dpi=DPI)