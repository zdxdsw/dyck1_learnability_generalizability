import os, json
from matplotlib import pyplot as plt
import numpy as np
# from matplotlib.patches import Rectangle

TASK_SUITES = [
    [ # first row
        ("S1_d_trulyleq[2-8]_2C", "acc"),
        ("S1_d_trulyleq[2-8]_2C", "acc"),
        ("S1_d_trulyleq[2-8]_2C", "p_acc"),
        ("S1_d_trulyleq[2-8]_2C", "n_acc"),
    ],
    [ # second row
        ("S1_d_trulyleq[2-8]_2A", "acc"),
        ("S1_d_trulyleq[2-8]_2A", "acc"),
        ("S1_d_trulyleq[2-8]_2A", "p_acc"),
        ("S1_d_trulyleq[2-8]_2A", "n_acc"),
    ]
]

COL_TITLES = [
    "acc (smoothed)",
    "acc (non-smoothed)",
    "p_acc (smoothed)",
    "n_acc (smoothed)"
]

ROW_TITLEs = [
    "Sinusoidal PE",
    "Rotary PE",
]

num_rows = len(TASK_SUITES)
num_cols = len(TASK_SUITES[0])
for row in TASK_SUITES: # check all rows in TASK_SUITES have the same number of columns
    assert len(row) == num_cols, "All rows in TASK_SUITES must have the same number of columns"

LEGEND_PREFIX = [
    [r"$d^{tr}_{\max}$"] * num_cols,
    [r"$d^{tr}_{\max}$"] * num_cols,
]

MODEL = "transformer"
OUTPUT_ROOT = f"/home/yingshac/workspace/tmlr/scripts/experiments/output_{MODEL}/"
YLIM = (0.0, 1.0)
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
    
    Testing Metrics in terminal.txt look like:

        Epoch 14 Step 14070 Metrics on Testing Sets:

        Testing set: d_2_l_41
        {'num_samples': 50, 'loss': 0.0, 'acc': 1.0}
        ...
        ...

    test_l_to_acc = {
        <train_d>:{
            41: [a1, a2, a3, a4, a5, a6, a7, a8, a9], # 9 seeds
            ...
        },
        ...
    }
"""

def extract_metrics_for_one_ax(output_dir, metric):
    test_l_to_metric = {d: {} for d in range(DMIN, DMAX+1)}

    task_suite_prefix = os.listdir(f"{output_dir}/")[0][:-1]
    for d in range(DMIN, DMAX+1):
        
        if not os.path.exists(f"{output_dir}/{task_suite_prefix}{d}"): 
            print(f"Directory {output_dir}/{task_suite_prefix}{d} does not exist, skip...")
            continue
        
        for handle in os.listdir(f"{output_dir}/{task_suite_prefix}{d}"):
            
            handle_path = f"{output_dir}/{task_suite_prefix}{d}/{handle}"
            
            # if "testing_metrics.txt" exists then use it, otherwise use terminal.txt
            FIND = False
            if os.path.exists(f"{handle_path}/testing_metrics.txt"):
                with open(f"{handle_path}/testing_metrics.txt", "r") as f:
                    lines = f.readlines()
                best_test_epoch_line_idx = 5
            else:
                with open(f"{output_dir}/{task_suite_prefix}{d}/{handle}/terminal.txt", "r") as f:
                    lines = f.readlines()
                FIND = True
            if not "Finish!!!" in lines[-1]:
                print(f"Job {handle_path} is not finished, skip...")
                continue
            
            if FIND:
                best_epoch_id = epoch_id = -1
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
                                    best_epoch_id = epoch_id
            
                # extract lengthy test metrics for Epoch = best_epoch_id
                best_test_epoch_line_idx = -1
                for i in range(len(lines)):
                    line = lines[i]
                    if line.startswith("Epoch") and "Metrics on Testing Sets" in line:
                        tmp = int(line.strip().split("Epoch ")[1].split(" Step")[0])
                        if tmp == best_epoch_id:
                            best_test_epoch_line_idx = i
                            break
                if best_test_epoch_line_idx == -1:
                    print(f"Cannot find Testing Metrics in {task_suite}{task_suite_prefix}{d}/{handle}")
                    continue
            
            
            # extract the metrics for each testing set
            for j in range(best_test_epoch_line_idx+1, len(lines)):
                line = lines[j]
                if line.startswith("Overall Testing Accuracy:"): 
                    break
                if line.startswith("Testing set: d_"):
                    test_l = int(line.strip().split("Testing set: d_")[1].split("_l_")[1])
                    metrics = eval(lines[j+1].strip())
                    if test_l not in test_l_to_metric[d]: test_l_to_metric[d][test_l] = []
                    if metric == "p_acc":
                        p_acc = metrics["TP"] / (metrics["TP"] + metrics["FN"]) if (metrics["TP"] + metrics["FN"]) > 0 else 0.0
                        test_l_to_metric[d][test_l].append(p_acc)
                    elif metric == "n_acc":
                        n_acc = metrics["TN"] / (metrics["TN"] + metrics["FP"]) if (metrics["TN"] + metrics["FP"]) > 0 else 0.0
                        test_l_to_metric[d][test_l].append(n_acc)
                    else:
                        test_l_to_metric[d][test_l].append(metrics[metric])
    
    return test_l_to_metric

"""
    Plot Testing Metrics 
    i2-i1+1 lines, each line corresponds to train d in RANGE -> legend
    x-axis: test d in RANGE
    y-axis: one plot for acc and another for loss
    error bar: std across 9 seeds (shaded)
    color: sample dmax-dmin+1 colors from 'autumn' colormap
"""

def plot_one_ax(test_l_to_metric, ax, row_idx, col_idx, smooth=False):
    colors = plt.cm.autumn(np.linspace(0, 1, DMAX-DMIN+1))
    for d in range(DMIN, DMAX+1):
        if not test_l_to_metric[d]: continue
        TEST_L_RANGE = (min(test_l_to_metric[d].keys()), max(test_l_to_metric[d].keys())+1)

        x = []
        y = []
        y_err = []
        
        if not smooth:
            for test_l in range(*TEST_L_RANGE):
                x.append(test_l)
                y.append(np.mean(test_l_to_metric[d][test_l]))
                y_err.append(np.std(test_l_to_metric[d][test_l]))
        
        else:
            BIN = 10
            start = TEST_L_RANGE[0]
            while start < TEST_L_RANGE[1]:
                end = start + BIN - 1
                y_values = []
                for test_l in range(start, end + 1):
                    if test_l in test_l_to_metric[d]:
                        y_values.extend(test_l_to_metric[d][test_l])
                if len(y_values) > 0:
                    x.append((start + end) / 2)       # bin center
                    y.append(np.mean(y_values))
                    y_err.append(np.std(y_values))

                start += BIN
        
        x = np.array(x)
        y = np.array(y)
        y_err = np.array(y_err)

        lineweight = 1.8 if col_idx != 1 else 1.2
        ax.plot(x, y, label=LEGEND_PREFIX[row_idx][col_idx]+f"={d}", color=colors[d-DMIN], alpha=0.75, linewidth=lineweight)
        upper = np.minimum(np.array(y)+np.array(y_err), YLIM[1]) # clip upper bound to YLIM[1]
        lower = np.maximum(np.array(y)-np.array(y_err), YLIM[0]) # clip lower bound to YLIM[0]
        ax.fill_between(x, lower, upper, color=colors[d-DMIN], alpha=0.1)
    
    ax.tick_params(axis="both", labelsize=10)
    if row_idx == num_rows - 1: ax.set_xlabel(r"$\ell^{test}$", fontsize=12, labelpad=10)

    y_label = r"acc at $\ell^{test}$"
    if col_idx == 0: ax.set_ylabel(y_label, fontsize=12, labelpad=6)
    
    # x ticks: TEST_L_RANGE, step = 20 + [TEST_L_RANGE[1]-1] if (TEST_L_RANGE[1]-1) % 20 != TEST_L_RANGE[0] % 20
    x_ticks = list(np.arange(TEST_L_RANGE[0], TEST_L_RANGE[1], step=20))
    if (TEST_L_RANGE[1]-1) % 20 != TEST_L_RANGE[0] % 20: x_ticks.append(TEST_L_RANGE[1]-1)
    ax.set_xticks(x_ticks)

    ax.set_ylim(YLIM[0]-0.05, YLIM[1]+0.05)
    ax.set_yticks(list(np.arange(YLIM[1], YLIM[0]-0.01, step=-0.2))[::-1])

    if col_idx == 0: ax.legend(loc="lower left", framealpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ------------------------------------------------------------------------------------------
save_path = os.path.join(os.path.basename(os.getcwd()) + ".pdf") # figure name = basename of cwd

fig, axes = plt.subplots(
    num_rows, num_cols, 
    figsize=(ROW_WIDTH, num_rows*ROW_HEIGHT),
    constrained_layout=True
)

# for ax in axes.flat: ax.set_facecolor("none") # all subplots have transparent background

for row in range(num_rows):
    for col in range(num_cols):
        task_suite, metric = TASK_SUITES[row][col]
        output_dir = os.path.join(OUTPUT_ROOT, task_suite)
        test_l_to_metric = extract_metrics_for_one_ax(output_dir, metric)
        ax = axes[row, col] if num_rows > 1 else axes[col] # if only one row, axes is 1D array
        smooth = col != 1
        plot_one_ax(test_l_to_metric, ax, row_idx=row, col_idx=col, smooth=smooth)

# Col Titles and Row Titles --------------------------------------------------------------------------------
for ax, title in zip(axes[0], COL_TITLES):
    ax.text(
        0.5, 1.04,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        linespacing=1.75,
        # fontweight="semibold",
    )

for ax, title in zip(axes[:, 0], ROW_TITLEs):
    ax.text(
        -0.28, 0.5,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        linespacing=1.75,
        fontweight="semibold",
        rotation=90,
    )

fig.suptitle(r"Length Generalization.$\ \ \ $Supervision: S1; Eval on: S1.$\ \ \ $2-Layer; Causal.", fontsize=12, fontweight="bold", y=1.05)

fig.savefig(save_path, bbox_inches="tight", dpi=DPI)
