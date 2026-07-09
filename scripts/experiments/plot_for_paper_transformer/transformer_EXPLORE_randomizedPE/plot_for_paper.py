import os, json
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

PLOT_STATS = [
    [ # first row
        "pg(acc)/S1_d_trulyleq[2-8]_2abcdefg_pg",
        "pg(acc)/S1_d_trulyleq[2-8]_2ghijklm_pg",
        "pg(acc)/S1_d_trulyleq[2-8]_1ABCDEFG_pg",
        "pg(acc)/S1_d_trulyleq[2-8]_1GHIJKLM_pg", 
    ], 
    [ # second row
        "pg(p_acc)/S1_d_trulyleq[2-8]_2abcdefg_pg",
        "pg(p_acc)/S1_d_trulyleq[2-8]_2ghijklm_pg",
        "pg(p_acc)/S1_d_trulyleq[2-8]_1ABCDEFG_pg",
        "pg(p_acc)/S1_d_trulyleq[2-8]_1GHIJKLM_pg", 
    ],
    [ # third row
        "pg(n_acc)/S1_d_trulyleq[2-8]_2abcdefg_pg",
        "pg(n_acc)/S1_d_trulyleq[2-8]_2ghijklm_pg",
        "pg(n_acc)/S1_d_trulyleq[2-8]_1ABCDEFG_pg",
        "pg(n_acc)/S1_d_trulyleq[2-8]_1GHIJKLM_pg", 
    ],
    [ # fourth row
        "dg_larger(acc)/S1_d_trulyleq[2-8]_2abcdefg_dg_larger",
        "dg_larger(acc)/S1_d_trulyleq[2-8]_2ghijklm_dg_larger",
        "dg_larger(acc)/S1_d_trulyleq[2-8]_1ABCDEFG_dg_larger",
        "dg_larger(acc)/S1_d_trulyleq[2-8]_1GHIJKLM_dg_larger",
    ],
    [ # fifth row
        "dg_larger(p_acc)/S1_d_trulyleq[2-8]_2abcdefg_dg_larger",
        "dg_larger(p_acc)/S1_d_trulyleq[2-8]_2ghijklm_dg_larger",
        "dg_larger(p_acc)/S1_d_trulyleq[2-8]_1ABCDEFG_dg_larger",
        "dg_larger(p_acc)/S1_d_trulyleq[2-8]_1GHIJKLM_dg_larger",
    ],
    [ # sixth row
        "dg_larger(n_acc)/S1_d_trulyleq[2-8]_2abcdefg_dg_larger",
        "dg_larger(n_acc)/S1_d_trulyleq[2-8]_2ghijklm_dg_larger",
        "dg_larger(n_acc)/S1_d_trulyleq[2-8]_1ABCDEFG_dg_larger",
        "dg_larger(n_acc)/S1_d_trulyleq[2-8]_1GHIJKLM_dg_larger",
    ],
    [ # seventh row
        "lg(acc)/S1_d_trulyleq[2-8]_2abcdefg_lg",
        "lg(acc)/S1_d_trulyleq[2-8]_2ghijklm_lg",
        "lg(acc)/S1_d_trulyleq[2-8]_1ABCDEFG_lg",
        "lg(acc)/S1_d_trulyleq[2-8]_1GHIJKLM_lg",
    ],
    [ # eighth row
        "lg(p_acc)/S1_d_trulyleq[2-8]_2abcdefg_lg",
        "lg(p_acc)/S1_d_trulyleq[2-8]_2ghijklm_lg",
        "lg(p_acc)/S1_d_trulyleq[2-8]_1ABCDEFG_lg",
        "lg(p_acc)/S1_d_trulyleq[2-8]_1GHIJKLM_lg",
    ],
    [ # ninth row
        "lg(n_acc)/S1_d_trulyleq[2-8]_2abcdefg_lg",
        "lg(n_acc)/S1_d_trulyleq[2-8]_2ghijklm_lg",
        "lg(n_acc)/S1_d_trulyleq[2-8]_1ABCDEFG_lg",
        "lg(n_acc)/S1_d_trulyleq[2-8]_1GHIJKLM_lg",
    ]
]

COL_TITLES = [
    "Shifted PE-Aug\n2-Layer; Non-Causal.",
    "\n2-Layer; Non-Causal.",
    "Shifted PE-Aug\n1-Layer; Causal.",
    "\n1-Layer; Causal.",
]

ROW_TITLES = [
    "",
    "p_acc",
    "n_acc",
    "",
    "p_acc",
    "n_acc",
    "",
    "p_acc",
    "n_acc",
    "",
    "p_acc",
    "n_acc",
]

MODEL = "transformer"
PLOT_STATS_DIR = f"/home/yingshac/workspace/tmlr/scripts/experiments/single_metric_plot_{MODEL}"
YLIM = (0.0, 1.0)
ROW_WIDTH = 13
ROW_HEIGHT = 2.6
DPI = 300

# Constants --------------------------------------------------------------------------------

DMIN, DMAX = 2, 8

METRIC_SHORTHAND_TO_FULL = {
    "cvg": "Convergence", 
    "pg": "Permutation Generalization", 
    "dg_larger": "Depth Generalization",
    "dg_smaller": "Depth Generalization (toward smaller)", # only available for non-cumu
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

    # title = f"{args.supervision} L={args.num_hidden_layers}"
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
    if row_idx == num_rows - 1: ax.set_xlabel(x_label, fontsize=12)
    if col_idx == 0: ax.set_ylabel(y_label, fontsize=12, labelpad=10)
    
    ax.set_xticks(list(range(DMIN, DMAX+1)))
    if row_idx > 0: 
        ax.set_ylim(YLIM[0]-0.05, YLIM[1]+0.05)
        ax.set_yticks(list(np.arange(YLIM[1], YLIM[0]-0.01, step=-0.2))[::-1])
    else: 
        ax.set_ylim(0.45, 1.05)
        ax.set_yticks(list(np.arange(1.0, 0.45-0.01, step=-0.2))[::-1])    
    
    if col_idx == row_idx == 0: 
        ax.legend(loc="lower right", framealpha=0.6, 
                  fontsize=11, title="Positional Encoding")
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

for ax in axes.flat: ax.set_facecolor("none") # all subplots have transparent background

for row in range(num_rows):
    for col in range(num_cols):
        plot_stats_filename = PLOT_STATS[row][col]
        ax = axes[row, col] if num_rows > 1 else axes[col] # if only one row, axes is 1D array
        plot_one_ax(os.path.join(PLOT_STATS_DIR, plot_stats_filename), ax, row_idx=row, col_idx=col)

# Col Titles and Row Titles --------------------------------------------------------------------------------
for ax, title in zip(axes[0], COL_TITLES):
    ax.text(
        0.5, 1.02,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        linespacing=1.75,
        fontweight="semibold",
    )

## separately add a red S2 for the last column's title
ax = axes[0][1]
ax.text(
    0.5, 1.13,
    "Randomized PE-Aug",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=12,
    color="red",
    fontweight="semibold",
)
ax = axes[0][-1]
ax.text(
    0.5, 1.13,
    "Randomized PE-Aug",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=12,
    color="red",
    fontweight="semibold",
)

for ax, title in zip(axes[:, 0], ROW_TITLES):
    ax.text(
        -0.35, 0.5,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        linespacing=1.75,
        rotation=90,
        fontweight="semibold",
    )

# Shades ------------------------------------------------------------------------------------------
fig.canvas.draw()
for c in [0, 2]:
    col_axes = axes[:, c]

    x0 = min(ax.get_position().x0 for ax in col_axes)-0.027
    x1 = max(ax.get_position().x1 for ax in col_axes)+0.002
    y0 = 0.0
    y1 = 1.005

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

fig.suptitle(r"Supervision: SL1; Eval on: SL1.", fontsize=12, fontweight="bold", y=1.03)

fig.savefig(save_path, bbox_inches="tight", dpi=DPI)