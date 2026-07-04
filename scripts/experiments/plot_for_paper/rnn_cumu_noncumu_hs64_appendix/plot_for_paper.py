import os, json
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

TASK_SUITES = [
    [ # first row
        ("S1_d_trulyleq[2-8]_A", "acc"),
        ("S2_d_trulyleq[2-8]_A", "acc"),
        ("S3_d_trulyleq[2-8]_A", "acc"),
        ("S2_d_trulyleq[2-8]_A", "S2_acc"),
    ],
    [ # second row
        ("S1_d_[2-8]_A", "acc"),
        ("S2_d_[2-8]_A", "acc"),
        ("S3_d_[2-8]_A", "acc"),
        ("S2_d_[2-8]_A", "S2_acc"),
    ]
]

COL_TITLES = [
    "Supervision: S1\nEval on: S1",
    "Supervision: S1 + S2\nEval on: S1",
    "Supervision: S1 + S3\nEval on: S1",
    "Supervision: S1 + S2\nEval on: ",
]

num_rows = len(TASK_SUITES)
num_cols = len(TASK_SUITES[0])
for row in TASK_SUITES: # check all rows in TASK_SUITES have the same number of columns
    assert len(row) == num_cols, "All rows in TASK_SUITES must have the same number of columns"

LEGEND_PREFIX = [
    [r"$d^{tr}_{\max}$"] * num_cols,
    [r"$d^{tr}$ "] * num_cols,
]

ROW_TITLEs = [
    "Cumu",
    "Non-Cumu",
]

MODEL = "rnn"
OUTPUT_ROOT = f"/home/yingshac/workspace/tmlr/scripts/experiments/output_{MODEL}/"
YLIM = (0.43, 1.0)
ROW_WIDTH = 14
ROW_HEIGHT = 2.8
DPI = 300

# Constants --------------------------------------------------------------------------------

DMIN, DMAX = 2, 8

"""
    Process each terminal.txt file in <output_dir>/<task_suite>/<handle>/
    Extract lines that include "Epoch <epoch> Step <step> Metrics on [Validation|Testing] Sets:"

    Validation Metrics in terminal.txt look like:
    
        Epoch 14 Step 14070 Metrics on Validation Sets:

        Validation set: d_2
        {'num_samples': 10000, 'loss': 0.0, 'acc': 1.0}
        ...
        ...
    
    val_d_to_<metric> = {
        <train_d>:{
            2: [a1, a2, a3, a4, a5, a6, a7, a8, a9], # 9 seeds
            ...
        },
        ...
    }
"""

def extract_metrics_for_one_ax(output_dir, metric):
    val_d_to_metric = {d: {} for d in range(DMIN, DMAX+1)}

    task_suite_prefix = os.listdir(f"{output_dir}/")[0][:-1]
    for d in range(DMIN, DMAX+1):
        
        if not os.path.exists(f"{output_dir}/{task_suite_prefix}{d}"): 
            print(f"Directory {output_dir}/{task_suite_prefix}{d} does not exist, skip...")
            continue
        
        for handle in os.listdir(f"{output_dir}/{task_suite_prefix}{d}"):
            
            handle_path = f"{output_dir}/{task_suite_prefix}{d}/{handle}"
            
            # if "validation_metrics.txt" exists then use it, otherwise use terminal.txt
            if os.path.exists(f"{handle_path}/validation_metrics.txt"):
                with open(f"{handle_path}/validation_metrics.txt", "r") as f:
                    lines = f.readlines()
            else:
                with open(f"{output_dir}/{task_suite_prefix}{d}/{handle}/terminal.txt", "r") as f:
                    lines = f.readlines()
            if not "Finish!!!" in lines[-1]:
                print(f"Job {handle_path} is not finished, skip...")
                continue
            
            best_epoch_line_idx = -1
            smallest_same_d_val_loss = float("inf")
            for i in range(len(lines)):
                line = lines[i]
                if line.startswith("Epoch") and "Metrics on Validation Sets" in line:
                    epoch_line_idx = i
                    # extract the metrics for each validation set
                    for j in range(epoch_line_idx+1, len(lines)):
                        line = lines[j]
                        if line.startswith("Epoch") and "Metrics on Validation Sets" in line:
                            break
                        if line.startswith("Validation set: d_"):
                            val_d = int(line.strip().split("Validation set: d_")[1])
                            if val_d != d: continue
                            metrics = eval(lines[j+1].strip())
                            if metrics["loss"] <= smallest_same_d_val_loss:
                                smallest_same_d_val_loss = metrics["loss"]
                                best_epoch_line_idx = epoch_line_idx

            if best_epoch_line_idx == -1:
                print(f"Cannot find Validation Metrics in {handle_path}")
                continue
            
            # extract the metrics for each validation set
            for j in range(best_epoch_line_idx+1, len(lines)):
                line = lines[j]
                if line.startswith("Epoch") and "Metrics on Testing Sets" in line:
                    break
                if line.startswith("Validation set: d_"):
                    val_d = int(line.strip().split("Validation set: d_")[1])
                    metrics = eval(lines[j+1].strip())
                    if val_d not in val_d_to_metric[d]: val_d_to_metric[d][val_d] = []
                    val_d_to_metric[d][val_d].append(metrics[metric])
    return val_d_to_metric

"""
    Plot Validation Metrics 
    i2-i1+1 lines, each line corresponds to train d in RANGE -> legend
    x-axis: test d in RANGE
    y-axis: one plot for acc and another for loss
    error bar: std across 9 seeds (shaded)
    color: sample dmax-dmin+1 colors from 'winter' colormap
"""

def plot_one_ax(val_d_to_acc, ax, row_idx, col_idx):
    colors = plt.cm.winter(np.linspace(0, 1, DMAX-DMIN+1))
    for d in range(DMIN, DMAX+1):
        if not val_d_to_acc[d]: continue
        x = []
        y = []
        y_err = []
        VAL_D_RANGE = (min(val_d_to_acc[d].keys()), max(val_d_to_acc[d].keys())+1)
        for val_d in range(*VAL_D_RANGE):
            x.append(val_d)
            y.append(np.mean(val_d_to_acc[d][val_d]))
            y_err.append(np.std(val_d_to_acc[d][val_d]))
        ax.plot(x, y, label=LEGEND_PREFIX[row_idx][col_idx]+f"={d}", color=colors[d-DMIN], alpha=0.75, linewidth=1.8)
        upper = np.minimum(np.array(y)+np.array(y_err), YLIM[1]) # clip upper bound to YLIM[1]
        lower = np.maximum(np.array(y)-np.array(y_err), YLIM[0]) # clip lower bound to YLIM[0]
        ax.fill_between(x, lower, upper, color=colors[d-DMIN], alpha=0.1)
        ax.scatter(d, np.mean(val_d_to_acc[d][d]), color=colors[d-DMIN], s=20, zorder=5) # draw a dot on val_d = train_d
    
    ax.tick_params(axis="both", labelsize=10)
    if row_idx == num_rows - 1: ax.set_xlabel(r"$d^{test}$", fontsize=12, labelpad=10)

    y_label = r"acc at $d^{test}$"
    if col_idx == 0: ax.set_ylabel(y_label, fontsize=12, labelpad=6)
    
    ax.set_ylim(YLIM[0]-0.05, YLIM[1]+0.05)
    #ax.set_xticks(list(range(1, 17)))
    ax.set_xticks([1,2,3,4,5,6,7,8,9,10,12,14,16])
    ax.set_yticks(list(np.arange(YLIM[1], YLIM[0]-0.01, step=-0.2))[::-1])

    if col_idx == 1: ax.legend(loc="upper right", framealpha=0.6)
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

for row in range(num_rows):
    for col in range(num_cols):
        task_suite, metric = TASK_SUITES[row][col]
        output_dir = os.path.join(OUTPUT_ROOT, task_suite)
        val_d_to_acc = extract_metrics_for_one_ax(output_dir, metric)
        ax = axes[row, col] if num_rows > 1 else axes[col] # if only one row, axes is 1D array
        plot_one_ax(val_d_to_acc, ax, row_idx=row, col_idx=col)

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

# separately add a red S2 for the last column's title
ax = axes[0][-1]
ax.text(
    0.68, 1.03,
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
        -0.25, 0.5,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        linespacing=1.75,
        fontweight="semibold",
        rotation=90,
    )

# Shades ------------------------------------------------------------------------------------------
fig.canvas.draw()
col_axes = axes[:, 0]

x0 = min(ax.get_position().x0 for ax in col_axes)-0.065
x1 = max(ax.get_position().x1 for ax in col_axes)+0.003
# y0 = min(ax.get_position().y0 for ax in col_axes)
# y1 = max(ax.get_position().y1 for ax in col_axes)
y0 = -0.01
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


fig.savefig(save_path, bbox_inches="tight", dpi=DPI)
