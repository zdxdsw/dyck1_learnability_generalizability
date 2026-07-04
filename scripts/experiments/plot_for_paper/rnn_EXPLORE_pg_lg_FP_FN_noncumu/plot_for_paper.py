import os, json
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

PLOT_STATS = [
    [ # first row
        "pg(p_acc)/S1_d_[2-8]_ABCDEF_pg",
        "pg(p_acc)/S2_d_[2-8]_ABCDEF_pg",
        "pg(p_acc)/S3_d_[2-8]_ABCDEF_pg",
        "pg(S2_p_acc)/S2_d_[2-8]_ABCDEF_pg",
    ],
    [ # second row
        "pg(n_acc)/S1_d_[2-8]_ABCDEF_pg",
        "pg(n_acc)/S2_d_[2-8]_ABCDEF_pg",
        "pg(n_acc)/S3_d_[2-8]_ABCDEF_pg",
        "pg(S2_n_acc)/S2_d_[2-8]_ABCDEF_pg",
    ],
    [ # third row
        "lg(p_acc)/S1_d_[2-8]_ABCDEF_lg",
        "lg(p_acc)/S2_d_[2-8]_ABCDEF_lg",
        "lg(p_acc)/S3_d_[2-8]_ABCDEF_lg",
        "lg(S2_p_acc)/S2_d_[2-8]_ABCDEF_lg",
    ],
    [ # fourth row
        "lg(n_acc)/S1_d_[2-8]_ABCDEF_lg",
        "lg(n_acc)/S2_d_[2-8]_ABCDEF_lg",
        "lg(n_acc)/S3_d_[2-8]_ABCDEF_lg",
        "lg(S2_n_acc)/S2_d_[2-8]_ABCDEF_lg",

    ]
]

COL_TITLES = [
    "Supervision: S1\nEval on: S1",
    "Supervision: S1 + S2\nEval on: S1",
    "Supervision: S1 + S3\nEval on: S1", 
    "Supervision: S1 + S2\nEval on: ",
]

ROW_TITLEs = [
    "p_acc",
    "n_acc",
    "p_acc",
    "n_acc",
]

MODEL = "rnn"
PLOT_STATS_DIR = f"/home/yingshac/workspace/tmlr/scripts/experiments/single_metric_plot_{MODEL}"
YLIM = (0.05, 1.0)
ROW_WIDTH = 14
ROW_HEIGHT = 2.8
DPI = 300

# Constants --------------------------------------------------------------------------------

DMIN, DMAX = 2, 8

METRIC_SHORTHAND_TO_FULL = {
    "cvg": "Convergence", 
    "pg": "Permutation Generalization", 
    "dg_larger": "Depth Generalization (towards larger)",
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
    metric = plot_stats_filename.split("_")[-1]

    # title = f"{args.supervision} L={args.num_hidden_layers}"
    x_label = r"$d^{tr}_{\max}$ (cumu)" if mode == "cumu" else r"$d^{tr}$ (non-cumu)"
    y_label = METRIC_SHORTHAND_TO_FULL[metric]
    # if ROW_YLABEL_PREFIXES:
    #     y_label = ROW_YLABEL_PREFIXES[row_idx] + y_label

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

    if row_idx == num_rows - 1: # only last row has x label
        ax.set_xlabel(x_label, fontsize=12)
    if col_idx == 0: # only first column has y label
        ax.set_ylabel(y_label, fontsize=12, labelpad=6)
    ax.set_ylim(YLIM[0]-0.05, YLIM[1]+0.05)
    ax.set_xticks(list(range(DMIN, DMAX+1)))
    # yticks: ymax, ymin, step=-0.2, then reverse the list to get ascending order
    ax.set_yticks(list(np.arange(YLIM[1], YLIM[0]-0.01, step=-0.2))[::-1])
    if col_idx == 0: # only first column has legend
        ax.legend(loc="lower right", framealpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)



num_rows = len(PLOT_STATS)
num_cols = len(PLOT_STATS[0])
for row in PLOT_STATS: # check all rows in PLOT_STATS have the same number of columns
    assert len(row) == num_cols, "All rows in PLOT_STATS must have the same number of columns"

save_path = os.path.join(os.path.basename(os.getcwd()) + ".pdf") # figure name = basename of cwd

fig, axes = plt.subplots(
    num_rows, num_cols, 
    figsize=(ROW_WIDTH, num_rows*ROW_HEIGHT),
    # sharex=True,
    # sharey=True,
    constrained_layout=True
)

for row in range(num_rows):
    for col in range(num_cols):
        plot_stats_filename = PLOT_STATS[row][col]
        ax = axes[row, col] if num_rows > 1 else axes[col] # if only one row, axes is 1D array
        plot_one_ax(os.path.join(PLOT_STATS_DIR, plot_stats_filename), ax, row_idx=row, col_idx=col)

for ax, title in zip(axes[0], COL_TITLES):
    ax.text(
        0.5, 1.03,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        linespacing=1.75,
        fontweight="semibold",
    )
# separately add a red S2 for the last column's title
ax = axes[0][-1]
ax.text(
    0.67, 1.03,
    "S2",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=12,
    color="red",
    fontweight="semibold",
)

for ax, title in zip(axes[:, 0], ROW_TITLEs):
    ax.text(
        -0.23, 0.5,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        linespacing=1.75,
        fontweight="semibold",
        rotation=90,
    )

# fig.tight_layout()
fig.savefig(save_path, bbox_inches="tight", dpi=DPI)