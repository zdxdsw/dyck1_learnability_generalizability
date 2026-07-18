import os, json
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

PLOT_STATS = [
    [ # first row
        "lg(acc)/S1_d_trulyleq[2-8]_ABCDEF_lg",
        "lg(acc)/S2_d_trulyleq[2-8]_ABCDEF_lg",
        "lg(acc)/S3_d_trulyleq[2-8]_ABCDEF_lg",
    ],
    [ # second row
        "lg(acc)/S1_d_[2-8]_ABCDEF_lg",
        "lg(acc)/S2_d_[2-8]_ABCDEF_lg",
        "lg(acc)/S3_d_[2-8]_ABCDEF_lg",
    ],
    [ # third row
        "dg_smaller(acc)/S1_d_[2-8]_ABCDEF_dg_smaller",
        "dg_smaller(acc)/S2_d_[2-8]_ABCDEF_dg_smaller",
        "dg_smaller(acc)/S3_d_[2-8]_ABCDEF_dg_smaller",
    ],
]

COL_TITLES = [
    "Supervision: SL1\nEval on: SL1",
    "Supervision: SL1 + SL2\nEval on: SL1",
    "Supervision: SL1 + SL3\nEval on: SL1", 
]

num_rows = len(PLOT_STATS)
num_cols = len(PLOT_STATS[0])
for row in PLOT_STATS: # check all rows in PLOT_STATS have the same number of columns
    assert len(row) == num_cols, "All rows in PLOT_STATS must have the same number of columns"

ROW_TITLEs = [
    "Cumu",
    "Non-Cumu",
    "Non-Cumu",
]

MODEL = "rnn"
PLOT_STATS_DIR = f"/home/yingshac/workspace/tmlr/scripts/experiments/single_metric_plot_{MODEL}"
YLIM = (0.42, 1.0)
ROW_WIDTH = 10
ROW_HEIGHT = 2.8
DPI = 300

# Constants --------------------------------------------------------------------------------

DMIN, DMAX = 2, 8

METRIC_SHORTHAND_TO_FULL = {
    "cvg": "Convergence", 
    "pg": "Permutation Generalization", 
    "dg_larger": "Depth Generalization (towards larger)",
    "dg_smaller": r"Depth Gen $\mathbf{dg\_smaller}$", # only available for non-cumu
    "lg": "Length Generalization",
    "dg_+1": "Depth +1",
    "dg_+2": "Depth +2",
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
        lower = np.maximum(np.array(y)-np.array(y_err), YLIM[0]-0.05) # clip lower bound to YLIM[0]
        ax.fill_between(x, lower, upper, color=LEGEND_NAME_TO_COLOR[legend_name], alpha=0.1)

    ax.tick_params(axis="both", labelsize=10)

    if row_idx ==2: ax.set_xlabel(x_label, fontsize=12, labelpad=6)
    if col_idx == 0: # only first column has y label
        ax.set_ylabel(y_label, fontsize=12, labelpad=6)
    ax.set_ylim(YLIM[0]-0.05, YLIM[1]+0.05)
    ax.set_xticks(list(range(DMIN, DMAX+1)))
    # yticks: ymax, ymin, step=-0.2, then reverse the list to get ascending order
    ax.set_yticks(list(np.arange(YLIM[1], YLIM[0]-0.01, step=-0.2))[::-1])
    if col_idx == row_idx == 0: # only first column has legend
        ax.legend(loc="lower left", framealpha=0.4, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ------------------------------------------------------------------------------------------
save_path = os.path.join(os.path.basename(os.getcwd()) + ".pdf") # figure name = basename of cwd

fig, axes = plt.subplots(
    num_rows, num_cols, 
    figsize=(ROW_WIDTH, num_rows*ROW_HEIGHT),
    constrained_layout=True
)

for ax in axes.flat: ax.set_facecolor("none") # all subplots have transparent background
fig.set_constrained_layout_pads(hspace=0.08)

for row in range(num_rows):
    for col in range(num_cols):
        plot_stats_filename = PLOT_STATS[row][col]
        ax = axes[row, col] if num_rows > 1 else axes[col] # if only one row, axes is 1D array
        plot_one_ax(os.path.join(PLOT_STATS_DIR, plot_stats_filename), ax, row_idx=row, col_idx=col)

# Col Titles and Row Titles --------------------------------------------------------------------------------
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

first_col_axes = axes[:, 0] if num_rows > 1 else axes
for ax, title in zip(first_col_axes, ROW_TITLEs):
    ax.text(
        -0.27, 0.5,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        linespacing=1.75,
        fontweight="semibold",
        rotation=90,
    )

# Shades --------------------------------------------------------------------------------
# shade the first row
fig.canvas.draw()
row_axes = axes[0, :]

x0 = 0.023
x1 = max(ax.get_position().x1 for ax in row_axes)
y0 = min(ax.get_position().y0 for ax in row_axes)-0.04
y1 = 1.01

pad_x = 0.0
pad_y = 0.0

rect = Rectangle(
    (x0 - pad_x, y0 - pad_y),
    (x1 - x0) + 2 * pad_x,
    (y1 - y0) + 2 * pad_y,
    transform=fig.transFigure,
    facecolor="0.95",
    edgecolor="none",
    zorder=-1,
)

fig.add_artist(rect)


# fig.tight_layout()
fig.savefig(save_path, bbox_inches="tight", dpi=DPI)