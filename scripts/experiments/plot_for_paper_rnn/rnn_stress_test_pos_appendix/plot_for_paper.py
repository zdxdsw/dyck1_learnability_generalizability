import os, json
from matplotlib import pyplot as plt
import numpy as np
# from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.patches import Rectangle


PROJECT_ROOT = "/home/yingshac/workspace/tmlr/scripts/experiments"

PLOT_STATS_PATHS = [ # three columns
    # "single_metric_plot_rnn/stress_test_pos_singletask/S1_d_trulyleq[2-8]_ABCDEF/S1_d_trulyleq8_ABCDEF_acc.json",
    # "single_metric_plot_rnn/stress_test_pos_singletask/S2_d_trulyleq[2-8]_ABCDEF/S2_d_trulyleq8_ABCDEF_acc.json",
    # "single_metric_plot_rnn/stress_test_pos_singletask/S3_d_trulyleq[2-8]_ABCDEF/S3_d_trulyleq8_ABCDEF_acc.json",
    "single_metric_plot_rnn/stress_test_pos_singletask/S1_d_[2-8]_ABCDEF/S1_d_8_ABCDEF_acc.json",
    "single_metric_plot_rnn/stress_test_pos_singletask/S2_d_[2-8]_ABCDEF/S2_d_8_ABCDEF_acc.json",
    "single_metric_plot_rnn/stress_test_pos_singletask/S3_d_[2-8]_ABCDEF/S3_d_8_ABCDEF_acc.json",
]

ROW_IDX_TO_LEGEND_NAMES = [
    "hs=64",
    "hs=32",
	"hs=16",
	"hs=8",
	"hs=6",
	"hs=4",
]

COL_TITLES = [
    "Supervision: SL1",
    "Supervision: SL1 + SL2",
    "Supervision: SL1 + SL3",
]

LEGEND_NAME_COLORBAR_HELPERS = { # min_alpha, max_alpha,
    "hs=64": (0.1, 0.85),
    "hs=32": (0.1, 0.75),
    "hs=16": (0.1, 0.85),
    "hs=8": (0.1, 1.0),
    "hs=6": (0.15, 1.0),
    "hs=4": (0.1, 0.9),
}

ROW_TITLES = list(LEGEND_NAME_COLORBAR_HELPERS.keys())

HATCH_FACECOLOR = tuple([0.95]*3)
HATCH_EDGECOLOR = tuple([0.75]*3)

ROW_WIDTH = 8.5
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

CODE_TO_LEGEND_NAME = {
	"A": "hs=64",
	"B": "hs=32",
	"C": "hs=16",
	"D": "hs=8",
	"E": "hs=6",
	"F": "hs=4",
}

LEGEND_NAME_TO_COLOR = {
	"hs=64": "firebrick",
	"hs=32": "blue",
	"hs=16": "darkorchid",
	"hs=8": "deepskyblue",
	"hs=6": "goldenrod",
    "hs=4": "hotpink",
}


def plot_for_one_ax(ax, row_idx, col_idx, data, colorname, min_alpha, max_alpha, x_ticklabels, y_ticklabels):
    heatmap_height, heatmap_width = data.shape

    # Build a colormap from color with varying alpha
    base = np.array(to_rgba(colorname))
    colors = [(*base[:3], min_alpha), (*base[:3], max_alpha)]
    cmap = LinearSegmentedColormap.from_list(f"{colorname}_alpha", colors)
    
    # You can set vmin/vmax manually if you want a fixed scale across plots
    # normalization tricks
    # print(f"Before normalization: {data}")
    # x1 = data[0][0]
    # cap = max([data[i][j] for i in range(heatmap_height) for j in range(1, heatmap_width) if i <= j]) # max entries other than the first cell
    # gap = x1 - cap
    # print(x1, cap, gap)
    # if gap > 0.5: data[0][0] -= gap/2 # reduce the first cell to make the color scale more visible

    # bottom = np.nanmin(data) # nanmin means it will ignore NaN values when computing the minimum
    # if bottom > 0.95: data[-1][0] = bottom*0.99
    # elif bottom > 0.7: data[-1][0] = bottom*0.95

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
num_rows, num_cols = 6, 3

fig, axes = plt.subplots(
    num_rows, num_cols,
    figsize=(ROW_WIDTH, num_rows*ROW_HEIGHT),
    constrained_layout=True,
)

fig.set_constrained_layout_pads(hspace=0.0, wspace=0.1)

for col_idx, path in enumerate(PLOT_STATS_PATHS):
    with open(os.path.join(PROJECT_ROOT, path), "r") as f: J = json.load(f)
    print(f"Processing {os.path.basename(path)} ...")
    
    for row_idx, legend_name in enumerate(ROW_IDX_TO_LEGEND_NAMES):
        ax = axes[row_idx, col_idx]
        min_alpha, max_alpha = LEGEND_NAME_COLORBAR_HELPERS[legend_name]
        plot_stats = J[legend_name]

        # initialize 3x3 heatmap filled with nan
        data = np.full((3, 3), np.nan)
        for split_name in plot_stats:
            i, j = SPLITNAME_TO_SUBPLOT_COORD[split_name]
            data[i, j] = plot_stats[split_name]["mean"]

        # def plot_for_one_ax(ax, data, colorname, min_alpha, max_alpha, x_labels, y_labels, ax_title):
        plot_for_one_ax(
            ax, row_idx, col_idx, data,
            LEGEND_NAME_TO_COLOR[legend_name], min_alpha, max_alpha,
            x_ticklabels=X_TICKLABELS,
            y_ticklabels=Y_TICKLABELS,
        )

# Col Titles and Row Titles --------------------------------------------------------------------------------
for ax, title in zip(axes[0], COL_TITLES):
    ax.text(
        0.47, 1.3,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
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
fig.suptitle(r"Eval on: SL1. Non-Cumu. $d^{tr}=8\ \ \ \ $Stress-Testing Positives (p_acc)", fontsize=14, fontweight="bold", y=1.015)
# Stress Test: Positives Bucketed by 'Longest Range-of-Dependency' (LRoD) and by 'Length'
fig.savefig(save_path, bbox_inches="tight", dpi=DPI)