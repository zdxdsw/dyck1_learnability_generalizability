import random, os, pytz, json
from tqdm import tqdm, trange
from datetime import datetime
timezone = pytz.timezone('America/New_York') 
from matplotlib import pyplot as plt
import numpy as np

output_dir = "/home/yingshac/workspace/tmlr/scripts/rnn/output/S1a_d_leq[2-8]_B"
figure_dir = "./S1a_d_leq[2-8]_B/"
# create figure_dir if not exist
os.makedirs(figure_dir, exist_ok=True)
"""
    Process each terminal.txt file in <output_dir>/<task_suite>/<handle>/
    The terminal.txt file contains the training log, so ignore them. 
    Extract lines that include "Epoch <epoch> Step <step> Metrics on [Validation|Testing] Sets:"
    If metrics for multiple epochs are printed, keep the one with the last epoch.
"""

"""
    Validation Metrics in terminal.txt look like:
    
        Epoch 14 Step 14070 Metrics on Validation Sets:

        Validation set: d_2
        {'num_samples': 10000, 'loss': 0.0, 'acc': 1.0}
        ...
        ...
        Validation set: d_8
        {'num_samples': 10000, 'loss': 3.39, 'acc': 0.48}

    Create dictionaries: val_d_to_acc, val_d_to_loss.

    val_d_to_acc = {
        <train_d>:{
            2: [a1, a2, a3, a4, a5], # 5 seeds
            ...
            8: [a1, a2, a3, a4, a5]
        },
        ...
    }
    val_d_to_loss = {
        <train_d>:{
            2: [l1, l2, l3, l4, l5], # 5 seeds
            ...
            8: [l1, l2, l3, l4, l5]
        },
        ...
    }
"""
val_d_to_acc = {d: {i: [] for i in range(2, 9)} for d in range(2, 9)}
val_d_to_loss = {d: {i: [] for i in range(2, 9)} for d in range(2, 9)}

"""
    Testing Metrics in terminal.txt look like:

        Epoch 14 Step 14070 Metrics on Testing Sets:

        Testing set: d_2_l_41
        {'num_samples': 50, 'loss': 0.0, 'acc': 1.0}
        ...
        ...
        Testing set: d_2_l_120
        {'num_samples': 350, 'loss': 0.0, 'acc': 1.0}

    Create dictionaries: test_l_to_acc, test_l_to_loss.

    test_l_to_acc = {
        <train_d>:{
            41: [a1, a2, a3, a4, a5], # 5 seeds
            ...
            120: [a1, a2, a3, a4, a5]
        },
        ...
    }
    test_l_to_loss = {
        <train_d>:{
            41: [l1, l2, l3, l4, l5], # 5 seeds
            ...
            120: [l1, l2, l3, l4, l5]
        },
        ...
    }
"""
test_l_to_acc = {d: {i: [] for i in range(41, 121)} for d in range(2, 9)}
test_l_to_loss = {d: {i: [] for i in range(41, 121)} for d in range(2, 9)}

print("Extracting metrics from terminal.txt files...")
task_suite_prefix = os.listdir(f"{output_dir}/")[0][:-1]
for d in trange(2, 9):
    for handle in os.listdir(f"{output_dir}/{task_suite_prefix}{d}"):
        with open(f"{output_dir}/{task_suite_prefix}{d}/{handle}/terminal.txt", "r") as f:
            lines = f.readlines()
        
        # iterate from the first line; go through every line that satisfies: 
        # line.startswith("Epoch") and "Metrics on Validation Sets" in line -> temporarily store line idx as epoch_line_idx
        # for every Epoch,
        # extract the metrics written in the next line of "Validation set: d_{d}", which looks like:
        # {'num_samples': 10000, 'loss': 0.0, 'acc': 1.0}"
        # create a map from epoch_line_idx to 'loss'
        # eventually obtain the epoch_line_idx with smallest loss; if there is a tie, choose the one with the largest epoch_lind_idx
        best_epoch_line_idx, best_epoch_id = -1, -1
        smallest_same_d_val_loss = float("inf")
        for i in range(len(lines)):
            line = lines[i]
            if line.startswith("Epoch") and "Metrics on Validation Sets" in line:
                epoch_line_idx = i
                epoch_id = int(line.strip().split("Epoch ")[1].split(" Step")[0])
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
                            best_epoch_id = epoch_id

        if best_epoch_line_idx == -1:
            print(f"Cannot find Validation Metrics in {task_suite_prefix}{d}/{handle}")
            continue
        # extract the metrics for each validation set
        for j in range(best_epoch_line_idx+1, len(lines)):
            line = lines[j]
            if line.startswith("Epoch") and "Metrics on Testing Sets" in line:
                break
            if line.startswith("Validation set: d_"):
                val_d = int(line.strip().split("Validation set: d_")[1])
                metrics = eval(lines[j+1].strip())
                val_d_to_acc[d][val_d].append(metrics["acc"])
                val_d_to_loss[d][val_d].append(metrics["loss"])

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
            print(f"Cannot find Testing Metrics in {task_suite_prefix}{d}/{handle}")
            continue
        # extract the metrics for each testing set
        for j in range(best_test_epoch_line_idx+1, len(lines)):
            line = lines[j]
            if line.startswith("Overall Testing Accuracy:"): 
                break
            if line.startswith("Testing set: d_"):
                test_l = int(line.strip().split("Testing set: d_")[1].split("_l_")[1])
                metrics = eval(lines[j+1].strip())
                test_l_to_acc[d][test_l].append(metrics["acc"])
                test_l_to_loss[d][test_l].append(metrics["loss"])

"""
    Plot Validation Metrics 
    7 lines, each line corresponds to train d in [2, 8] -> legend
    x-axis: test d in [2, 8]
    y-axis: one plot for acc and another for loss
    error bar: std across 5 seeds (shaded)
    title: "Acc/Loss on greater depth, dim=64"
    color: sample 7 colors from 'winter' colormap
"""
print("Plot Validation Metrics...")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
colors = plt.cm.winter(np.linspace(0, 1, 7))
for d in range(2, 9):
    x = []
    y = []
    y_err = []
    for val_d in range(2, 9):
        x.append(val_d)
        y.append(np.mean(val_d_to_acc[d][val_d]))
        y_err.append(np.std(val_d_to_acc[d][val_d]))
    plt.plot(x, y, label=f"train d={d}", color=colors[d-2])
    plt.fill_between(x, np.array(y)-np.array(y_err), np.array(y)+np.array(y_err), color=colors[d-2], alpha=0.2)
plt.xlabel("test d")
plt.ylabel("acc")
plt.title("Acc on greater depth")
plt.legend()

plt.subplot(1, 2, 2)
for d in range(2, 9):
    x = []
    y = []
    y_err = []
    for val_d in range(2, 9):
        x.append(val_d)
        y.append(np.mean(val_d_to_loss[d][val_d]))
        y_err.append(np.std(val_d_to_loss[d][val_d]))
    plt.plot(x, y, label=f"train d={d}", color=colors[d-2])
    plt.fill_between(x, np.array(y)-np.array(y_err), np.array(y)+np.array(y_err), color=colors[d-2], alpha=0.2)
plt.xlabel("test d")
plt.ylabel("loss")
plt.title("Loss on greater depth")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(figure_dir, "bve_greater_depth.png"), dpi=250, bbox_inches='tight') # bve = Best same d Val Epoch
plt.close()

"""
    Plot Testing Metrics 
    7 lines, each line corresponds to train d in [2, 8] -> legend
    x-axis: test l in [41, 120]
    y-axis: one plot for acc and another for loss
    error bar: std across 5 seeds (shaded)
    title: "Acc/Loss on greater length, dim=64"
    color: sample 7 colors from 'autumn' colormap
"""
print("Plot Testing Metrics...")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
colors = plt.cm.autumn(np.linspace(0, 1, 7))
for d in range(2, 9):
    x = []
    y = []
    y_err = []
    for test_l in range(41, 121):
        x.append(test_l)
        y.append(np.mean(test_l_to_acc[d][test_l]))
        y_err.append(np.std(test_l_to_acc[d][test_l]))
    plt.plot(x, y, label=f"train d={d}", color=colors[d-2])
    plt.fill_between(x, np.array(y)-np.array(y_err), np.array(y)+np.array(y_err), color=colors[d-2], alpha=0.2)
plt.xlabel("test l")
plt.ylabel("acc")
plt.title("Acc on greater length")
plt.legend()

plt.subplot(1, 2, 2)
for d in range(2, 9):
    x = []
    y = []
    y_err = []
    for test_l in range(41, 121):
        x.append(test_l)
        y.append(np.mean(test_l_to_loss[d][test_l]))
        y_err.append(np.std(test_l_to_loss[d][test_l]))
    plt.plot(x, y, label=f"train d={d}", color=colors[d-2])
    plt.fill_between(x, np.array(y)-np.array(y_err), np.array(y)+np.array(y_err), color=colors[d-2], alpha=0.2)
plt.xlabel("test l")
plt.ylabel("loss")
plt.title("Loss on greater length")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(figure_dir, "bve_greater_length.png"), dpi=250, bbox_inches='tight') # bve = Best same d Val Epoch
plt.close()