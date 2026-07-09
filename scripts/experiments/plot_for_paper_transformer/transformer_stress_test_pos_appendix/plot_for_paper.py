import os, json
from matplotlib import pyplot as plt
import numpy as np
# from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.patches import Rectangle


PROJECT_ROOT = "/home/yingshac/workspace/tmlr/scripts/experiments"

PLOT_STATS_PATHS = [ # four columns
    "single_metric_plot_transformer/stress_test_pos_singletask/S1_d_trulyleq[2-8]_2ABCDEFG/S1_d_trulyleq8_2ABCDEFG_acc.json",
    "single_metric_plot_transformer/stress_test_pos_singletask/S1_d_trulyleq[2-8]_2abcdefg/S1_d_trulyleq8_2abcdefg_acc.json",
    "single_metric_plot_transformer/stress_test_pos_singletask/S1_d_trulyleq[2-8]_1ABCDEFG/S1_d_trulyleq8_1ABCDEFG_acc.json",
    "single_metric_plot_transformer/stress_test_pos_singletask/S1_d_trulyleq[2-8]_1abcdefg/S1_d_trulyleq8_1abcdefg_acc.json"
]

ROW_IDX_TO_PE_NAMES = [
    "rotary",
    "scaler",
	"sinusoidal",
	"absolute",
	"alibi",
	"relative",
	"none",
]

COL_TITLES = [
    "2-Layer, Causal",
    "2-Layer, Non-Causal",
    "1-Layer, Causal",
    "1-Layer, Non-Causal",
]

PE_NAME_COLORBAR_HELPERS = { # min_alpha, max_alpha,
    "rotary": (0.1, 0.85),
    "scaler": (0.05, 0.75),
    "sinusoidal": (0.15, 0.9),
    "absolute": (0.1, 0.85),
    "alibi": (0.1, 1.0),
    "relative": (0.15, 1.0),
    "none": (0.1, 0.9),
}

ROW_TITLES = list(PE_NAME_COLORBAR_HELPERS.keys())

HATCH_FACECOLOR = tuple([0.95]*3)
HATCH_EDGECOLOR = tuple([0.75]*3)

ROW_WIDTH = 11
ROW_HEIGHT = 2.8
DPI = 300

# Constants --------------------------------------------------------------------------------

X_TICKLABELS = [
    r"$[2, 40]$",
    r"$[42,80]$",
    r"$[82,120]$"
]

Y_TICKLABELS = [
    r"$[2, 40]$",
    r"$(40,80]$",
    r"$(80,120]$"
]

SPLITNAME_TO_SUBPLOT_COORD = {
    "Lleq40_RoDleq40": (0, 0),
    "L42to80_RoDleq40": (0, 1),
    "L82to120_RoDleq40": (0, 2),
    "L42to80_RoD41to80": (1, 1),
    "L82to120_RoD41to80": (1, 2),
    "L82to120_RoD81to120": (2, 2),
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


def plot_for_one_ax(ax, row_idx, col_idx, data, colorname, min_alpha, max_alpha, x_ticklabels, y_ticklabels):
    heatmap_height, heatmap_width = data.shape

    # Build a colormap from color with varying alpha
    base = np.array(to_rgba(colorname))
    colors = [(*base[:3], min_alpha), (*base[:3], max_alpha)]
    cmap = LinearSegmentedColormap.from_list(f"{colorname}_alpha", colors)

    # You can set vmin/vmax manually if you want a fixed scale across plots
    norm = Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))
    im = ax.imshow(data, cmap=cmap, norm=norm)

    # Lower-triangular non-applicable cells: i > j
    for i in range(heatmap_height):
        for j in range(heatmap_width):
            if i > j:
                rect = Rectangle(
                    (j - 0.5, i - 0.5),
                    1, 1,
                    facecolor=HATCH_FACECOLOR, edgecolor=HATCH_EDGECOLOR,
                    hatch="\\\\", linewidth=0.8,
                )
                ax.add_patch(rect)
            else:
                ax.text(
                    j, i,
                    f"{data[i, j]:.3f}",
                    ha="center", va="center",
                    fontsize=11, color="black",
                )

    # Ticks / labels
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(heatmap_width))
    ax.set_xticklabels(x_ticklabels)
    if row_idx == 0:
        ax.set_xlabel(r"$\ell$ Bucket", fontsize=13, labelpad=12)
        ax.xaxis.set_label_position("top")

    ax.set_yticks(np.arange(heatmap_height))
    ax.set_yticklabels(y_ticklabels, rotation=90, va="center")
    if col_idx == 0:
        ax.set_ylabel(r"LRoD Bucket", fontsize=13, labelpad=10)
    
    # Cell boundaries
    ax.set_xticks(np.arange(-0.5, heatmap_width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, heatmap_height, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # ax.set_title(ax_title, y=-0.2, x=0.43, fontsize=12, fontweight="semibold")


# ------------------------------------------------------------------------------------------
num_rows, num_cols = 7, 4

fig, axes = plt.subplots(
    num_rows, num_cols,
    figsize=(ROW_WIDTH, num_rows*ROW_HEIGHT),
    constrained_layout=True,
)

fig.set_constrained_layout_pads(hspace=0.0, wspace=0.1)

for col_idx, path in enumerate(PLOT_STATS_PATHS):
    with open(os.path.join(PROJECT_ROOT, path), "r") as f: J = json.load(f)
    print(f"Processing {os.path.basename(path)} ...")
    
    for row_idx, pe_name in enumerate(ROW_IDX_TO_PE_NAMES):
        ax = axes[row_idx, col_idx]
        min_alpha, max_alpha = PE_NAME_COLORBAR_HELPERS[pe_name]
        plot_stats = J[pe_name]

        data = np.full((3, 3), np.nan)
        for split_name in plot_stats:
            i, j = SPLITNAME_TO_SUBPLOT_COORD[split_name]
            data[i, j] = plot_stats[split_name]["mean"]

        # def plot_for_one_ax(ax, data, colorname, min_alpha, max_alpha, x_labels, y_labels, ax_title):
        plot_for_one_ax(
            ax, row_idx, col_idx, data,
            PE_NAME_TO_COLOR[pe_name], min_alpha, max_alpha,
            x_ticklabels=X_TICKLABELS,
            y_ticklabels=Y_TICKLABELS,
        )

# Col Titles and Row Titles --------------------------------------------------------------------------------
for ax, title in zip(axes[0], COL_TITLES):
    ax.text(
        0.45, 1.35,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        linespacing=1.6,
        fontweight="semibold",
    )

for ax, title in zip(axes[:, 0], ROW_TITLES):
    ax.text(
        -0.38, 0.5,
        title.capitalize(),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=13,
        linespacing=1.75,
        fontweight="semibold",
        rotation=90,
    )

# ------------------------------------------------------------------------------------------
save_path = os.path.join(os.path.basename(os.getcwd()) + ".pdf") # figure name = basename of cwd
fig.suptitle(r"Supervision: SL1; Eval on: SL1.$\ \ \ \ $Acc on stress testing sets of positives", fontsize=14, fontweight="bold", y=1.015)
# Stress Test: Positives Bucketed by 'Longest Range-of-Dependency' (LRoD) and by 'Length'
fig.savefig(save_path, bbox_inches="tight", dpi=DPI)