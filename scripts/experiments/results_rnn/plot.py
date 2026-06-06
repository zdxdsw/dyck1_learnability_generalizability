import random, os, pytz, json, re, argparse
from tqdm import tqdm, trange
from datetime import datetime
timezone = pytz.timezone('America/New_York') 
from matplotlib import pyplot as plt
import numpy as np

args = argparse.ArgumentParser()
args.add_argument("--task_suite", type=str)
args.add_argument("--output_dir", type=str)
args.add_argument("--figure_dir", type=str)
args = args.parse_args()

task_suite = args.task_suite
output_dir = args.output_dir
figure_dir = args.figure_dir

# extract range of d as a tuple (i1, i2) from output_dir name: matching "[\d-\d]_"
match = re.search(r"(\d)-(\d)", output_dir)
i1, i2 = int(match.group(1)), int(match.group(2))
RANGE = (i1, i2+1)
print(f"range of d: [{i1}, {i2}]")

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
# assume we don't know the range of val_d in advance, but we know the range of train_d is RANGE
val_d_to_acc = {d: {} for d in range(*RANGE)}
val_d_to_loss = {d: {} for d in range(*RANGE)}

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
# assume we do not know the range of test_l in advance
test_l_to_acc = {d: {} for d in range(*RANGE)}
test_l_to_loss = {d: {} for d in range(*RANGE)}

print("Extracting metrics from terminal.txt files...")
task_suite_prefix = os.listdir(f"{output_dir}/")[0][:-1]
for d in trange(*RANGE):
    for handle in os.listdir(f"{output_dir}/{task_suite_prefix}{d}"):
        with open(f"{output_dir}/{task_suite_prefix}{d}/{handle}/terminal.txt", "r") as f:
            lines = f.readlines()
        
        # iterate from the last line and go backwards, 
        # find the last line idx that satisfies:         
        # line.startswith("Epoch") and "Metrics on Validation Sets" in line
        last_val_line_idx = -1
        for i in range(len(lines)-1, -1, -1):
            line = lines[i]
            if line.startswith("Epoch") and "Metrics on Validation Sets" in line:
                last_val_line_idx = i
                break
        if last_val_line_idx == -1:
            print(f"Cannot find Validation Metrics in {task_suite_prefix}{d}/{handle}")
            continue
        # extract the metrics for each validation set
        for j in range(last_val_line_idx+1, len(lines)):
            line = lines[j]
            if line.startswith("Epoch") and "Metrics on Testing Sets" in line:
                break
            if line.startswith("Validation set: d_"):
                val_d = int(line.strip().split("Validation set: d_")[1])
                metrics = eval(lines[j+1].strip())
                if val_d not in val_d_to_acc[d]: val_d_to_acc[d][val_d] = []
                if val_d not in val_d_to_loss[d]: val_d_to_loss[d][val_d] = []
                val_d_to_acc[d][val_d].append(metrics["acc"])
                val_d_to_loss[d][val_d].append(metrics["loss"])

        # iterate from the last line and go backwards, 
        # find the last line idx that satisfies:         
        # line.startswith("Epoch") and "Metrics on Testing Sets" in line
        last_test_line_idx = -1
        for i in range(len(lines)-1, -1, -1):
            line = lines[i]
            if line.startswith("Epoch") and "Metrics on Testing Sets" in line:
                last_test_line_idx = i
                break
        if last_test_line_idx == -1:
            print(f"Cannot find Testing Metrics in {task_suite_prefix}{d}/{handle}")
            continue
        # extract the metrics for each testing set
        for j in range(last_test_line_idx+1, len(lines)):
            line = lines[j]
            if line.startswith("Testing set: d_"):
                test_l = int(line.strip().split("Testing set: d_")[1].split("_l_")[1])
                metrics = eval(lines[j+1].strip())
                if test_l not in test_l_to_acc[d]: test_l_to_acc[d][test_l] = []
                if test_l not in test_l_to_loss[d]: test_l_to_loss[d][test_l] = []
                test_l_to_acc[d][test_l].append(metrics["acc"])
                test_l_to_loss[d][test_l].append(metrics["loss"])

"""
    Plot Validation Metrics 
    i2-i1+1 lines, each line corresponds to train d in RANGE -> legend
    x-axis: test d in RANGE
    y-axis: one plot for acc and another for loss
    error bar: std across 9 seeds (shaded)
    title: "Acc/Loss on greater depth, dim=64"
    color: sample i2-i1+1 colors from 'winter' colormap
"""
print("Plot Validation Metrics...")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
colors = plt.cm.winter(np.linspace(0, 1, i2-i1+1))
for d in range(*RANGE):
    x = []
    y = []
    y_err = []
    VAL_D_RANGE = (min(val_d_to_acc[d].keys()), max(val_d_to_acc[d].keys())+1)
    for val_d in range(*VAL_D_RANGE):
        x.append(val_d)
        y.append(np.mean(val_d_to_acc[d][val_d]))
        y_err.append(np.std(val_d_to_acc[d][val_d]))
    plt.plot(x, y, label=f"train d={d}", color=colors[d-2])
    upper = np.minimum(np.array(y)+np.array(y_err), 1.0)
    plt.fill_between(x, np.array(y)-np.array(y_err), upper, color=colors[d-2], alpha=0.2)
    plt.scatter(d, np.mean(val_d_to_acc[d][d]), color=colors[d-2], s=20, zorder=5) # draw a dot on val_d = train_d
plt.xlabel("test d")
plt.ylabel("acc")
plt.title("Acc on greater depth")
plt.legend()

plt.subplot(1, 2, 2)
for d in range(*RANGE):
    x = []
    y = []
    y_err = []
    VAL_D_RANGE = (min(val_d_to_loss[d].keys()), max(val_d_to_loss[d].keys())+1)
    for val_d in range(*VAL_D_RANGE):
        x.append(val_d)
        y.append(np.mean(val_d_to_loss[d][val_d]))
        y_err.append(np.std(val_d_to_loss[d][val_d]))
    plt.plot(x, y, label=f"train d={d}", color=colors[d-2])
    lower = np.maximum(np.array(y)-np.array(y_err), 0.0)
    plt.fill_between(x, lower, np.array(y)+np.array(y_err), color=colors[d-2], alpha=0.2)
    plt.scatter(d, np.mean(val_d_to_loss[d][d]), color=colors[d-2], s=20, zorder=5) # draw a dot on val_d = train_d
plt.xlabel("test d")
plt.ylabel("loss")
plt.title("Loss on greater depth")
plt.legend()

plt.tight_layout()
plt.savefig(f"{figure_dir}greater_depth.png", dpi=250, bbox_inches='tight')
plt.close()

"""
    Plot Testing Metrics 
    i2-i1+1 lines, each line corresponds to train d in RANGE -> legend
    x-axis: test l in TEST_L_RANGE
    y-axis: one plot for acc and another for loss
    error bar: std across 9 seeds (shaded)
    title: "Acc/Loss on greater length, dim=64"
    color: sample i2-i1+1 colors from 'autumn' colormap
"""
print("Plot Testing Metrics...")
plt.figure(figsize=(14, 12))
plt.subplot(2, 2, 1)
colors = plt.cm.autumn(np.linspace(0, 1, i2-i1+1))
for d in range(*RANGE):
    x = []
    y = []
    y_err = []
    TEST_L_RANGE = (min(test_l_to_acc[d].keys()), max(test_l_to_acc[d].keys())+1)
    for test_l in range(*TEST_L_RANGE):
        x.append(test_l)
        y.append(np.mean(test_l_to_acc[d][test_l]))
        y_err.append(np.std(test_l_to_acc[d][test_l]))
    plt.plot(x, y, label=f"train d={d}", color=colors[d-2])
    upper = np.minimum(np.array(y)+np.array(y_err), 1.0)
    plt.fill_between(x, np.array(y)-np.array(y_err), upper, color=colors[d-2], alpha=0.2)
plt.xlabel("test l")
plt.ylabel("acc")
plt.title("Acc on greater length")
plt.legend()

plt.subplot(2, 2, 2)
for d in range(*RANGE):
    x = []
    y = []
    y_err = []
    TEST_L_RANGE = (min(test_l_to_loss[d].keys()), max(test_l_to_loss[d].keys())+1)
    for test_l in range(*TEST_L_RANGE):
        x.append(test_l)
        y.append(np.mean(test_l_to_loss[d][test_l]))
        y_err.append(np.std(test_l_to_loss[d][test_l]))
    plt.plot(x, y, label=f"train d={d}", color=colors[d-2])
    lower = np.maximum(np.array(y)-np.array(y_err), 0.0)
    plt.fill_between(x, lower, np.array(y)+np.array(y_err), color=colors[d-2], alpha=0.2)
plt.xlabel("test l")
plt.ylabel("loss")
plt.title("Loss on greater length")
plt.legend()

# Plot Testing Metrics (smoothed) ...
BIN = 10
plt.subplot(2, 2, 3)
colors = plt.cm.autumn(np.linspace(0, 1, i2-i1+1))
for d in range(*RANGE):
    x = []
    y = []
    y_err = []
    TEST_L_RANGE = (min(test_l_to_acc[d].keys()), max(test_l_to_acc[d].keys())+1)
    start = TEST_L_RANGE[0]
    while start < TEST_L_RANGE[1]:
        end = start + BIN - 1
        y_values = []
        for test_l in range(start, end + 1):
            if test_l in test_l_to_acc[d]:
                y_values.extend(test_l_to_acc[d][test_l])
        if len(y_values) > 0:
            x.append((start + end) / 2)       # bin center
            y.append(np.mean(y_values))
            y_err.append(np.std(y_values))

        start += BIN

    x = np.array(x)
    y = np.array(y)
    y_err = np.array(y_err)
    upper = np.minimum(y + y_err, 1.0)

    plt.plot(x, y, label=f"train d={d}", color=colors[d - 2])
    plt.fill_between(x, y - y_err, upper, color=colors[d - 2], alpha=0.1)
plt.xlabel("test l")
plt.xlim(TEST_L_RANGE[0]-1, TEST_L_RANGE[1])
plt.ylabel("acc")
plt.title("Acc on greater length (smoothed)")
plt.legend()

plt.subplot(2, 2, 4)
for d in range(*RANGE):
    x = []
    y = []
    y_err = []
    TEST_L_RANGE = (min(test_l_to_loss[d].keys()), max(test_l_to_loss[d].keys())+1)
    start = TEST_L_RANGE[0]
    while start < TEST_L_RANGE[1]:
        end = start + BIN - 1
        y_values = []
        for test_l in range(start, end + 1):
            if test_l in test_l_to_loss[d]:
                y_values.extend(test_l_to_loss[d][test_l])
        if len(y_values) > 0:
            x.append((start + end) / 2)       # bin center
            y.append(np.mean(y_values))
            y_err.append(np.std(y_values))

        start += BIN

    x = np.array(x)
    y = np.array(y)
    y_err = np.array(y_err)
    lower = np.maximum(y - y_err, 0.0) 

    plt.plot(x, y, label=f"train d={d}", color=colors[d - 2])
    plt.fill_between(x, lower, y + y_err, color=colors[d - 2], alpha=0.1)
plt.xlabel("test l")
plt.xlim(TEST_L_RANGE[0]-1, TEST_L_RANGE[1])
plt.ylabel("loss")
plt.title("Loss on greater length (smoothed)")
plt.legend()

plt.tight_layout()
plt.savefig(f"{figure_dir}greater_length.png", dpi=250, bbox_inches='tight')
plt.close()