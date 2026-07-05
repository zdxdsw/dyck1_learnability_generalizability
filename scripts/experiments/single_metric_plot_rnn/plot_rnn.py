from typing import Literal
import os, json, argparse, re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

MODEL = "RNN"
OUTPUT_ROOT = f"/home/yingshac/workspace/tmlr/scripts/experiments/"

SUPERVISION = "S3" # ["S1", "S2", "S3", "S1a", "S2a", "S3a"]
MODE = "non-cumu" # ["non-cumu", "cumu"] will map to ["d", "trulyleqd"]
NUM_HIDDEN_LAYERS = 1 
LEGEND_CODE_RANGE =  "ABCDEF"

METRIC = ["dg_+1"] # ["cvg", "pg", "dg_larger", "dg_smaller", "lg", "dg_+1", "dg_+2"]
LOSS_OR_ACC = ["p_acc", "n_acc"] 
"""
["loss", "acc", "S2_loss", "S2_acc", "S2_p_acc", "S2_n_acc", "S3_loss", "S3_acc", "p_acc", "n_acc"]

    p_acc = TP / (TP + FN)  i.e. acc on positive samples
    n_acc = TN / (TN + FP)  i.e. acc on negative samples
"""

# ------------------------------------------------------------------
# Useful mappings for and plotting
# ------------------------------------------------------------------
METRIC_SHORTHAND_TO_FULL = {
    "cvg": "Convergence", 
    "pg": "Permutation Generalization", 
    "dg_larger": "Depth Generalization (towards larger)",
    "dg_smaller": "Depth Generalization (towards smaller)", # only available for non-cumu
    "lg": "Length Generalization",
    "dg_+1": "Depth Generalization (+1)",
    "dg_+2": "Depth Generalization (+2)",
}

# When value is a list, 
# if the 1st file exists, then use 1st file; otherwise use 2nd file.
METRIC_SHORTHAND_TO_LOGFILE = { 
    "cvg": "training_metrics.txt",
    "pg": ["validation_metrics.txt", "terminal.txt"],
    "dg_larger": ["validation_metrics.txt", "terminal.txt"],
    "dg_+1": ["validation_metrics.txt", "terminal.txt"],
    "dg_+2": ["validation_metrics.txt", "terminal.txt"],
    "dg_smaller": ["validation_metrics.txt", "terminal.txt"], # only available for non-cumu
    "lg": ["testing_metrics.txt", "terminal.txt"],
}
METRIC_SHORTHAND_TO_SPLIT = {
    "cvg": "Training",
    "pg": "Validation",
    "dg_larger": "Validation",
    "dg_+1": "Validation",
    "dg_+2": "Validation",
    "dg_smaller": "Validation", # only available for non-cumu
    "lg": "Testing",
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

LEGEND_NAME_TO_PLOT_ORDER = {
	"hs=64": 0,
	"hs=32": 1,
	"hs=16": 2,
	"hs=8": 3,
	"hs=6": 4,
	"hs=4": 5,
}

def Print(s=""): print(s, flush=True)

"""
# ------------------------------------------------------------------
1. Get Task Suites
# ------------------------------------------------------------------
    Use supervision, depth_mode, and legend_code_range to determine task_suites
    E.g. S1_d_trulyleq[2-8]_A, S1_d_trulyleq[2-8]_C, ... S1_d_trulyleq[2-8]_F
    task_suites = get_task_suites() <- args
        for task_suite in os.listdir(output_dir):
            find the ones that match the pattern:
            {supervision}_d_{depth_mode[:-1]}[<digit>-<digit>]_<legend_code>
    legend_codes = [task_suite[-1] for task_suite in task_suites]
    task_suite_full = task_suites[0][:-1] + "".join(legend_codes)
"""
def get_task_suites(output_dir, supervision, depth_mode, num_hidden_layers, legend_code_range):
    
    # since this is a regex, use a raw f-string for readability:
    regex_pattern = (
        rf"{supervision}_d_{depth_mode[:-1]}"
        rf"\[\d+-\d+\]_"
        rf"[{legend_code_range}]"
    )

    task_suites = []
    for task_suite in os.listdir(output_dir):
        if re.match(regex_pattern, task_suite):
            task_suites.append(task_suite)
    
    Print("="*14 + " Found task_suites: " + "="*14 + f"\n{task_suites}\n")
    
    return task_suites

"""
# ------------------------------------------------------------------
2. Infer range of train_d -> d_min, d_max
# ------------------------------------------------------------------
    infer from the first task_suite (inside brackets), 
    which should be the same for all 7 task_suites. 
"""
def infer_train_d_range(task_suite):
    match = re.search(r"\[(\d+)-(\d+)\]", task_suite)
    if match:
        d_min = int(match.group(1))
        d_max = int(match.group(2))
        Print("="*8 + f" Inferred train_d range: [{d_min}, {d_max}] " + "="*8)
        return d_min, d_max
    else:
        raise ValueError(f"Cannot infer train_d range from task_suite: {task_suite}")

"""
# ------------------------------------------------------------------
3. Get Plot Stats
# ------------------------------------------------------------------
    plot_stats = {
        "<legend_name>": {
            "<train_d>": {
                "mean": <float>,
                "std": <float>
            },
            ...
        }
    }
    For each task_suite:
        legend_code = task_suite[-1]
        legend_name = LEGEND_CODE_TO_NAME[legend_code.upper()]
        
        For each task (i.e. train_d, 7 in total):
            train_d = task[-1]

            For each handle (i.e. random seed, 9 in total):
                
                best_epoch_id = get_best_epoch_id(output_dir, task_suite, task, handle, train_d)
                read_logfile() 
                    -> os.path.join(output_dir, task_suite, task, handle, logfile)
                    -> lines
                best_epoch_lineidx = locate_best_epoch_lineidx()
                    -> startswith "Epoch <best_epoch_id>" and contains "Metrics on <split> Sets:"
                keep reading from best_epoch_lineidx+1, read two at a time, until an empty line.
                    -> line1: startswith "<split> set: d_{x}" or "<split> set: d_8_l_{x}" -> extract x
                    -> line2: metrics = eval(line2.strip()) -> y = metrics[args.loss_or_acc]
                now, we have a list of (x, y) per handle
                single_metric = get_single_metric() <- list of (x, y)
            
            compute mean and std of single_metric across 9 handles
            write to plot_stats[legend_name][train_d]: {"mean": mean, "std": std}
"""
def read_logfile(output_dir, task_suite, task, handle, logfile):
    if isinstance(logfile, list):
        log_path = None
        for f in logfile:
            log_path = os.path.join(output_dir, task_suite, task, handle, f)
            if os.path.exists(log_path):
                break
    else:
        log_path = os.path.join(output_dir, task_suite, task, handle, logfile)
    
    if log_path is None or (not os.path.exists(log_path)):
        Print(f"Func read_logfile(): [skip] Missing logfile: {task_suite}/{task}/{handle}: ", logfile)
        return None
    
    with open(log_path, "r") as f: lines = f.readlines()
    return lines


def get_best_epoch_id(output_dir, task_suite, task, handle, same_d):
    """
    Read terminal.txt and select the epoch with smallest same-d validation loss.
    Tie-breaking: If multiple epochs have the same loss, choose the largest epoch_id.
    Returns: int epoch_id, or None if the run should be skipped.
    """
    terminal_path = os.path.join(output_dir, task_suite, task, handle, "terminal.txt")

    if not os.path.exists(terminal_path):
        Print(f"Func get_best_epoch_id(): [skip] Missing terminal.txt: {terminal_path}")
        return None

    with open(terminal_path, "r") as f: lines = f.readlines()
    # the last non-empty line should print f"Finish!!! {hangle}", so we can check if the job is finished
    if not lines[-1].startswith(f"Finish!!! {handle}"):
        Print(f"Func get_best_epoch_id(): [skip] Job {task_suite}/{task}/{handle} is not finished.")
        return None
    
    best_epoch_id = -1
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
                    if val_d != same_d: continue
                    metrics = eval(lines[j+1].strip())
                    if metrics["loss"] <= smallest_same_d_val_loss:
                        smallest_same_d_val_loss = metrics["loss"]
                        best_epoch_id = epoch_id

    if best_epoch_id == -1:
        Print(f"Func get_best_epoch_id(): [skip] Cannot find best_val_loss Metrics in {task_suite}/{task}/{handle}")
        return None

    return best_epoch_id


def locate_best_epoch_lineidx(lines, best_epoch_id, split):
    for i in range(len(lines)):
        line = lines[i]
        if line.startswith(f"Epoch {best_epoch_id}") and f"Metrics on {split} Sets:" in line:
            return i
    Print(f"Func locate_best_epoch_lineidx(): [skip] Cannot find best epoch line for epoch {best_epoch_id} and split {split}")
    return None


def get_loss_or_acc(metrics, loss_or_acc):
    """
    metrics: a dict recovered from log
    
    e.g. {
        'num_samples': 50, 
        'S2_loss': 0.23, 
        'S2_acc': 0.97, 
        'S2_numel': 409260, 
        'S2_FP': 14100, 
        'S2_TP': 310570, 
        'S2_FN': 0, 
        'S2_TN': 84590, 
        'S3_loss': 0.53, 
        'S3_acc': 0.87, 
        'S3_numel': 4550, 
        'FP': 0.06, 
        'TP': 0.0, 
        'FN': 0.0, 
        'TN': 0.94, 
        'loss': 0.45, 
        'acc': 0.94
    }
    """
    
    if loss_or_acc == "p_acc":
        return metrics["TP"] / (metrics["TP"] + metrics["FN"]) if (metrics["TP"] + metrics["FN"]) > 0 else 0.0
    if loss_or_acc == "n_acc":
        return metrics["TN"] / (metrics["TN"] + metrics["FP"]) if (metrics["TN"] + metrics["FP"]) > 0 else 0.0
    if loss_or_acc == "S2_p_acc":
        return metrics["S2_TP"] / (metrics["S2_TP"] + metrics["S2_FN"]) if (metrics["S2_TP"] + metrics["S2_FN"]) > 0 else 0.0
    if loss_or_acc == "S2_n_acc":
        return metrics["S2_TN"] / (metrics["S2_TN"] + metrics["S2_FP"]) if (metrics["S2_TN"] + metrics["S2_FP"]) > 0 else 0.0
    elif loss_or_acc in metrics:
        return metrics[loss_or_acc]
    else:
        Print(f"Func get_loss_or_acc(): [skip] Cannot find {loss_or_acc} in metrics: {metrics}")
        return None

"""
# ------------------------------------------------------------------
4. Flesh Out get_single_metric() 
# ------------------------------------------------------------------
 
    Convergence + non-cumu: 
        single_metric = y at x = train_d
    Convergence + cumu: 
        single_metric = AUC if we plot (x, y)
    
    Permutation Generalization + non-cumu: 
        single_metric = y at x = train_d
    Permutation Generalization + cumu: 
        single_metric = AUC if we plot (x, y)
    
    Depth Generalization (towards smaller) + (must be) non-cumu: 
        single_metric = AUC if we plot (x, y) for d_min <= x <= train_d - 1
    Depth Generalization (towards larger) + cumu/non-cumu:
        single_metric = AUC if we plot (x, y) for train_d + 1 <= x <= d_max

    Length Generalization + cumu/non-cumu: 
        single_metric = AUC if we plot (x, y)

"""
def compute_auc(x_y_pairs, x_min, x_max):
    x_y_pairs = sorted(x_y_pairs, key=lambda pair: pair[0]) # sort by x
    if x_min is None: x_min = x_y_pairs[0][0]
    if x_max is None: x_max = x_y_pairs[-1][0]

    # filter pairs with x_min <= x <= x_max
    filtered_pairs = [pair for pair in x_y_pairs if x_min <= pair[0] <= x_max]
    if len(filtered_pairs) < 1:
        Print(f"Func compute_auc(): [skip] Not enough points to compute AUC.")
        return None
    elif len(filtered_pairs) == 1:
        return filtered_pairs[0][1]
    
    # compute AUC using trapezoidal rule
    auc = 0.0
    for i in range(len(filtered_pairs)-1):
        x1, y1 = filtered_pairs[i]
        x2, y2 = filtered_pairs[i+1]
        auc += (x2 - x1) * (y1 + y2) / 2
    
    return auc / (x_max - x_min) # normalize AUC by the range of x

def get_single_metric(x_y_pairs, metric, mode, train_d):
    if metric in ["cvg", "pg"]:
        if mode == "non-cumu":
            for x, y in x_y_pairs:
                if x == train_d:
                    return y
            Print(f"Func get_single_metric(): [skip] Cannot find y for x=train_d={train_d}")
            return None
        elif mode == "cumu":
            auc = compute_auc(x_y_pairs, None, train_d)
            return auc
        else:
            Print(f"Func get_single_metric(): [skip] Unknown mode: {mode}")
            return None
    
    elif metric == "dg_smaller": # only for non-cumu
        if mode != "non-cumu":
            Print(f"Func get_single_metric(): [skip] dg_smaller metric is only applicable for non-cumu mode.")
            return None
        auc = compute_auc(x_y_pairs, None, train_d - 1)
        return auc
    
    elif metric == "dg_larger":
        auc = compute_auc(x_y_pairs, train_d + 1, None)
        return auc
    
    elif metric == "dg_+1":
        result = [y for x, y in x_y_pairs if x == train_d + 1]
        return result[0] if result else None
    
    elif metric == "dg_+2":
        result = [y for x, y in x_y_pairs if x == train_d + 2]
        return result[0] if result else None
    
    elif metric == "lg":
        auc = compute_auc(x_y_pairs, None, None)
        return auc
    
    else:
        Print(f"Func get_single_metric(): [skip] Unknown single_metric_name: {metric}")
        return None

"""
# ------------------------------------------------------------------
5. Plotting
# ------------------------------------------------------------------
    title: f"{supervision} L={num_hidden_layers}"
    x label: f"train_d ({cumu/non-cumu})"
    y label: f"{metric_fullname}"
    X: train_d
    Y: mean (fill_between +- std if error_band=True)
    x ticks: each int in [d_min, d_max]
    legend: legend_name
    save_path: f"../{METRIC}({LOSS_OR_ACC})/{task_suite_full}_{METRIC}.png" <- create dir if not exist
"""

def plot(plot_stats, args, d_min, d_max, figure_name):
    title = f"{args.supervision} L={args.num_hidden_layers}"
    x_label = f"train_d ({args.mode})"
    y_label = METRIC_SHORTHAND_TO_FULL[args.metric]
    save_dir = f"./{args.metric}({args.loss_or_acc})"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, figure_name)

    # plot one without error band and another with error band side by side
    plt.figure(figsize=(12, 5))
    ax = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    for legend_name in plot_stats:
        x = []
        y = []
        y_err = []
        for train_d in sorted(plot_stats[legend_name].keys()):
            stats = plot_stats[legend_name][train_d]
            x.append(train_d)
            y.append(stats["mean"])
            y_err.append(stats["std"])
        ax.plot(x, y, label=legend_name, color=LEGEND_NAME_TO_COLOR[legend_name], marker='.', alpha=0.5)
        ax2.plot(x, y, label=legend_name, color=LEGEND_NAME_TO_COLOR[legend_name], marker='.', alpha=0.5)
        upper = np.minimum(np.array(y)+np.array(y_err), 1.0)
        ax2.fill_between(x, np.array(y)-np.array(y_err), upper, color=LEGEND_NAME_TO_COLOR[legend_name], alpha=0.1)

    plt.suptitle(title, fontsize=14)
    for a in [ax, ax2]:
        a.set_xlabel(x_label, fontsize=12)
        a.set_ylabel(y_label, fontsize=12)
        a.set_xticks(list(range(d_min, d_max+1)))
        a.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=args.dpi)

    # save plot_stats to the same dir
    save_stats_path = save_path.replace(".png", ".json")
    with open(save_stats_path, "w") as f: json.dump(plot_stats, f, indent=4)
    
    Print("\n" + "="*19 + "Saving ..." + "="*19)
    Print(f"Plot destination: {save_path}\nStats destination: {save_stats_path}\n")


# Example of key contents in log files
"""
--------------------------------------------------
Epoch <best_epoch_id> Metrics on Training Sets:

Training set: d_1
{'num_samples': 16720, 'loss': 0.46, 'acc': 0.75}
Training set: d_2
{'num_samples': 30000, 'loss': 0.59, 'acc': 0.63}
...
...

--------------------------------------------------
Epoch <best_epoch_id> Step <some_number> Metrics on Validation Sets:

Validation set: d_8
{'num_samples': 10000, 'loss': 0.59, 'acc': 0.63}
Validation set: d_1
{'num_samples': 38, 'loss': 0.52, 'acc': 0.68}
Validation set: d_2
{'num_samples': 10000, 'loss': 0.59, 'acc': 0.63}
...
...
Validation set: d_16
{'num_samples': 320, 'loss': 0.6, 'acc': 0.62}

--------------------------------------------------
Epoch <best_epoch_id> Step <some_number> Metrics on Testing Sets:

Testing set: d_8_l_41
{'num_samples': 50, 'loss': 0.76, 'acc': 0.0}
Testing set: d_8_l_42
{'num_samples': 350, 'loss': 0.69, 'acc': 0.57}
...
...

--------------------------------------------------
"""


def run_all(args, task_suites = None):
    ### 1. Get Task Suites (if task_suites not provided)
    if task_suites is None:
        task_suites = get_task_suites(
            args.output_dir, 
            args.supervision, 
            args.depth_mode, 
            args.num_hidden_layers, 
            args.legend_code_range.upper()
        )
    
    legend_codes = sorted([task_suite[-1] for task_suite in task_suites])
    task_suite_full = task_suites[0][:-1] + "".join(legend_codes)

    ### 2. Infer range of train_d -> d_min, d_max
    d_min, d_max = infer_train_d_range(task_suites[0])

    ### 3. Get Plot Stats
    logfile = METRIC_SHORTHAND_TO_LOGFILE[args.metric]
    split = METRIC_SHORTHAND_TO_SPLIT[args.metric]
    plot_stats = defaultdict(lambda: defaultdict(dict)) # plot_stats[legend_name][train_d] = {"mean": mean, "std": std}
    for task_suite in task_suites:
        legend_code = task_suite[-1]
        legend_name = CODE_TO_LEGEND_NAME[legend_code.upper()]
        for task in os.listdir(os.path.join(args.output_dir, task_suite)):
            train_d = int(task[-1])
            for handle in os.listdir(os.path.join(args.output_dir, task_suite, task)):
                best_epoch_id = get_best_epoch_id(args.output_dir, task_suite, task, handle, train_d)
                if best_epoch_id is None: continue

                lines = read_logfile(args.output_dir, task_suite, task, handle, logfile)
                if lines is None: continue

                best_epoch_lineidx = locate_best_epoch_lineidx(lines, best_epoch_id, split)
                if best_epoch_lineidx is None: continue
                if lines[best_epoch_lineidx+1].strip() == "": 
                    # some <split>_metrics.txt files have an extra empty line after the keyline.
                    best_epoch_lineidx += 1

                x_y_pairs = []
                for i in range(best_epoch_lineidx+1, len(lines), 2):
                    line1 = lines[i].strip()
                    if line1 == "": break
                    if not line1.startswith(f"{split} set: d_"):
                        Print(f"Func plot(): [skip] Unexpected line format (expecting {split} set: d_x): {line1}")
                        continue
                    x = int(line1.split('_')[-1]) # extract x from line1, could be d_{x} for Training/Validation; or d_8_l_{x} for Testing

                    line2 = lines[i+1].strip()
                    try:
                        metrics = eval(line2)
                        y = get_loss_or_acc(metrics, args.loss_or_acc)
                        x_y_pairs.append((x, y))
                    except:
                        Print(f"Func plot(): [skip] Cannot parse metrics as dict from line {i+1}: {line2}")
                        continue
                
                ### 4. Get Single Metric for each handle
                # Print(f"\nBefore get_single_metric(), task={task}, \
                #       train_d={train_d}, type(train_d)={type(train_d)}\n\
                #         x_y_pairs: length={len(x_y_pairs)}\n{x_y_pairs}")
                single_metric = get_single_metric(
                    x_y_pairs, 
                    args.metric, 
                    args.mode, 
                    train_d, 
                )
                if single_metric is None: 
                    Print(f"Func plot(): [skip] Cannot compute single_metric for {task_suite}/{task}/{handle}")
                    continue
                # store single_metric for this handle; after processing all handles we will compute mean and std for this (legend_name, train_d)
                plot_stats[legend_name][train_d].setdefault("values", []).append(single_metric)
            
    # compute mean and std for each (legend_name, train_d)
    for legend_name in plot_stats:
        for train_d in plot_stats[legend_name]:
            values = plot_stats[legend_name][train_d].get("values", [])
            if len(values) == 0:
                Print(f"Func plot(): [skip] No values to compute mean and std for {legend_name} train_d={train_d}")
                continue
            mean = sum(values) / len(values)
            std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
            plot_stats[legend_name][train_d]["mean"] = mean
            plot_stats[legend_name][train_d]["std"] = std
            # we can also remove the "values" list to save memory, since we only need
            # skip for now
    # sort plot_stats.keys() according to LEGEND_NAME_TO_PLOT_ORDER
    plot_stats = dict(sorted(plot_stats.items(), key=lambda item: LEGEND_NAME_TO_PLOT_ORDER[item[0]]))
    # also sort plot_stats[legend_name]
    for legend_name in plot_stats:
        plot_stats[legend_name] = dict(sorted(plot_stats[legend_name].items()))
    ### 5. Plotting
    figure_name = f"{task_suite_full}_{args.metric}.png"
    plot(plot_stats, args, d_min, d_max, figure_name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        type=str, 
        default=MODEL, 
        help="Model type: RNN or Transformer"
    )
    parser.add_argument(
        "--supervision", 
        type=str, 
        default=SUPERVISION, 
        help="Supervision type: S1, S2, S3, S1a, S2a, S3a"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        default=MODE, 
        help="Mode: cumu or non-cumu"
    )
    parser.add_argument(
        "--loss_or_acc",
        type=str,
        default=LOSS_OR_ACC,
        help="Whether to plot loss or accuracy: one of ['loss', 'acc', 'p_acc', 'n_acc']"
    )
    parser.add_argument(
        "--metric",
        nargs="*",
        default=METRIC,
        help="Metric type"
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=NUM_HIDDEN_LAYERS,
        help="Number of hidden layers in the model"
    )
    parser.add_argument(
        "--legend_code_range",
        type=str,
        default=LEGEND_CODE_RANGE,
        help="Range of legend codes to include: e.g. 'ABCDEF'."
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for the saved plots"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output_dir = os.path.join(OUTPUT_ROOT, f"output_{args.model.lower()}")
    args.depth_mode = "d" if args.mode == "non-cumu" else "trulyleqd"

    task_suites = get_task_suites(
        args.output_dir, 
        args.supervision, 
        args.depth_mode, 
        args.num_hidden_layers, 
        args.legend_code_range.upper()
    )

    list_loss_or_acc = args.loss_or_acc
    list_metric = args.metric

    for loss_or_acc in list_loss_or_acc:
        args.loss_or_acc = loss_or_acc
        for metric in list_metric:
            # Some training_metrics.txt don’t have FP/TP/FN/TN
            # don’t try to plot “cvg” with p_acc/n_acc
            if metric == "cvg" and loss_or_acc in ["p_acc", "n_acc", "S2_p_acc", "S2_n_acc"]: continue
            if loss_or_acc == "S3_acc" and "dg" in metric: continue # depth generalization does not apply to S3.
            args.metric = metric
            run_all(args, task_suites)
