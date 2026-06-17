import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import numpy as np
from collections import defaultdict
from tqdm import tqdm

TASK_SUITE = [
    "S1_d_[2-8]_2A",
    "S1_d_[2-8]_2B",
    "S1_d_[2-8]_2C",
    "S1_d_[2-8]_2D",
    "S1_d_[2-8]_2E",
    "S1_d_[2-8]_2F",
    "S1_d_[2-8]_2G",
    "S1_d_[2-8]_4A",
    "S1_d_[2-8]_4B",
    "S1_d_[2-8]_4C",
    "S1_d_[2-8]_4D",
    "S1_d_[2-8]_4E",
    "S1_d_[2-8]_4F",
    "S1_d_[2-8]_4G",
    "S1_d_trulyleq[2-8]_2A",
    "S1_d_trulyleq[2-8]_2B",
    "S1_d_trulyleq[2-8]_2C",
    "S1_d_trulyleq[2-8]_2D",
    "S1_d_trulyleq[2-8]_2E",
    "S1_d_trulyleq[2-8]_2F",
    "S1_d_trulyleq[2-8]_2G",
    "S1_d_trulyleq[2-8]_4A",
    "S1_d_trulyleq[2-8]_4B",
    "S1_d_trulyleq[2-8]_4C",
    "S1_d_trulyleq[2-8]_4D",
    "S1_d_trulyleq[2-8]_4E",
    "S1_d_trulyleq[2-8]_4F",
    "S1_d_trulyleq[2-8]_4G",
    "S1_d_trulyleq[2-8]_4a",
    "S1_d_trulyleq[2-8]_4b",
    "S1_d_trulyleq[2-8]_4c",
    "S1_d_trulyleq[2-8]_4d",
    "S1_d_trulyleq[2-8]_4e",
    "S1_d_trulyleq[2-8]_4f",
    "S1_d_trulyleq[2-8]_4g",
]

MODEL = "Transformer"  # "RNN" or "Transformer"
OUTPUT_DIR = f"/home/yingshac/workspace/tmlr/scripts/experiments/output_{MODEL.lower()}"
FIGURE_DIR = f"/home/yingshac/workspace/tmlr/scripts/experiments/results_{MODEL.lower()}"
"""
========== training_metrics.txt looks like: ==========

Epoch 9 Metrics on Training Sets:

Training set: d_1
{'num_samples': 16720, 'loss': 0.0, 'acc': 1.0}
Training set: d_2
{'num_samples': 30000, 'loss': 0.0, 'acc': 1.0}

Finish!!! S1_d_trulyleq[2-8]_4F/S1_d_trulyleq2/0607_114340
"""

"""
1. Iterate through OUTPUT_DIR: task_suite/task/handle
    create a plot for each task_suite.
    plot_stats = {
        <train_d>: {
            "1": {"mean_loss": float, "std_loss": float, "mean_acc": float, "std_acc": float},
            "2": {"mean_loss": float, "std_loss": float, "mean_acc": float, "std_acc": float},
            ...,
        }
    }
2. For each task:
    train_d = int(task[-1])
    x_to_metrics = {
        1: {"loss": [], "acc": []},
        2: {"loss": [], "acc": []},
        ...,
    }

    3. For each handle, read training_metrics.txt.
        Locate line_idx containing "Epoch <epoch> Metrics on Training Sets:" -> critical_line_idx

        starting from critical_line_idx+2, read two lines at a time, until the next empty line.
        line1: Training set: d_<d> -> x = int(line1.split(" ")[-1].split("_")[-1])
        line2: m = eval(line2.strip()) -> loss = m["loss"], acc = m["acc"]
        Append loss and acc to x_to_metrics[x]["loss"] and x_to_metrics[x]["acc"]
    
    Sort x_to_metrics by x.
    Aggregate x_to_metrics for plot_states[train_d].

4. Plotting
    Sort plot_states by train_d.
    colors = plt.cm.winter(np.linspace(0, 1, len(plot_stats)))

    Acc and Loss plots side by side
    title: Acc/Loss on training depth
    xlabel: train d
    ylabel: acc/loss
    legend: max_train_d
    save_path: os.path.join(FIGURE_DIR, task_suite, "bve_training_depth.png")
"""

def get_plot_stats(task_suite):
    plot_stats = defaultdict(lambda: defaultdict(lambda: {
        "mean_loss": 0.0, 
        "std_loss": 0.0, 
        "mean_acc": 0.0, 
        "std_acc": 0.0
    }))
    
    for task in os.listdir(os.path.join(OUTPUT_DIR, task_suite)):
        x_to_metrics = defaultdict(lambda: {"loss": [], "acc": []})
        train_d = int(task[-1])

        ### Collect metrics across 9 seeds for the same task
        for handle in os.listdir(os.path.join(OUTPUT_DIR, task_suite, task)):
            training_metrics_path = os.path.join(OUTPUT_DIR, task_suite, task, handle, "training_metrics.txt")
            if not os.path.exists(training_metrics_path):
                print(f"Warning: {training_metrics_path} is missing. Skip ...")
                continue
            
            with open(training_metrics_path, "r") as f:
                lines = f.readlines()
            
            critical_line_idx = None
            for idx, line in enumerate(lines):
                if line.startswith("Epoch") and "Metrics on Training Sets" in line:
                    critical_line_idx = idx
                    break
            
            if critical_line_idx is None:
                print(f"Warning: Critical line not found in {training_metrics_path}. Skip ...")
                continue
            
            idx = critical_line_idx + 2
            while idx < len(lines) and lines[idx].strip() != "":
                line1 = lines[idx].strip()
                line2 = lines[idx + 1].strip()
                
                x = int(line1.split(" ")[-1].split("_")[-1])
                m = eval(line2)
                loss = m["loss"]
                acc = m["acc"]
                
                x_to_metrics[x]["loss"].append(loss)
                x_to_metrics[x]["acc"].append(acc)
                
                idx += 2
            
        ### Sort x_to_metrics by x. Aggregate x_to_metrics for plot_states[train_d].  
        for x in sorted(x_to_metrics.keys()):
            plot_stats[train_d][x]["mean_loss"] = np.mean(x_to_metrics[x]["loss"])
            plot_stats[train_d][x]["std_loss"] = np.std(x_to_metrics[x]["loss"])
            plot_stats[train_d][x]["mean_acc"] = np.mean(x_to_metrics[x]["acc"])
            plot_stats[train_d][x]["std_acc"] = np.std(x_to_metrics[x]["acc"])
    
    plot_stats = dict(sorted(plot_stats.items(), key=lambda item: item[0], reverse=True), ) # Sort plot_states by train_d, large first
    return plot_stats

def create_plot(plot_stats, save_path):
    colors = plt.cm.winter(np.linspace(0, 1, len(plot_stats)))
    colors = colors[::-1] # reverse the color order, so that larger train_d has darker color

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    handles = []
    for idx, (train_d, x_to_metrics) in enumerate(plot_stats.items()):
        x = sorted(x_to_metrics.keys())

        y1 = [x_to_metrics[xi]["mean_acc"] for xi in x]
        y1_err = [x_to_metrics[xi]["std_acc"] for xi in x]
        y2 = [x_to_metrics[xi]["mean_loss"] for xi in x]
        y2_err = [x_to_metrics[xi]["std_loss"] for xi in x]

        if len(x) > 1: # line plot with error band
            axes[0].plot(x, y1, label=f"max_train_d={train_d}", marker='.', color=colors[idx])
            upper = np.minimum(np.array(y1)+np.array(y1_err), 1.0)
            axes[0].fill_between(x, np.array(y1)-np.array(y1_err), upper, color=colors[idx], alpha=0.1)

            axes[1].plot(x, y2, label=f"max_train_d={train_d}", marker='.', color=colors[idx])
            lower = np.maximum(np.array(y2)-np.array(y2_err), 0.0)
            axes[1].fill_between(x, lower, np.array(y2)+np.array(y2_err), color=colors[idx], alpha=0.1)
        else: # scatter plot with error bar (on each dot)
            axes[0].errorbar(x, y1, yerr=y1_err, label=None, color=colors[idx], ecolor=mcolors.to_rgba(colors[idx], 0.2),fmt='o', capsize=5)
            axes[1].errorbar(x, y2, yerr=y2_err, label=None, color=colors[idx], ecolor=mcolors.to_rgba(colors[idx], 0.2), fmt='o', capsize=5)

            # give the legend a separate Line2D handle that only draws the marker
            handles.append(Line2D([0], [0], marker='o', linestyle='None', color=colors[idx], label=f"max_train_d={train_d}"))

        
    axes[0].set_title("Acc on training depth")
    axes[0].set_xlabel("training d")
    axes[0].set_ylabel("acc")
    
    axes[1].set_title("Loss on training depth")
    axes[1].set_xlabel("training d")
    axes[1].set_ylabel("loss")

    if handles:
        axes[0].legend(handles=handles[::-1])
        axes[1].legend(handles=handles[::-1])
    else:
        # retrieve axes legend handles and display legend in reverse order
        handles0, labels0 = axes[0].get_legend_handles_labels()
        handles1, labels1 = axes[1].get_legend_handles_labels()
        axes[0].legend(handles=handles0[::-1], labels=labels0[::-1])
        axes[1].legend(handles=handles1[::-1], labels=labels1[::-1])


    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    for task_suite in tqdm(TASK_SUITE):
        plot_stats = get_plot_stats(task_suite)
        save_path = os.path.join(FIGURE_DIR, task_suite, "bve_training_depth.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        create_plot(plot_stats, save_path)
    print("\nDone!\n")