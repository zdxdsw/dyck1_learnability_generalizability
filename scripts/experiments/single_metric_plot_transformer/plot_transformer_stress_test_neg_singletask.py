from typing import Literal
import os, json, argparse, re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

MODEL = "Transformer"  # "RNN" or "Transformer"
OUTPUT_ROOT = f"/home/yingshac/workspace/tmlr/scripts/experiments/"

SUPERVISION = "S1" # ["S1", "S2", "S3", "S1a", "S2a", "S3a"]
MODE = "cumu" # ["non-cumu", "cumu"] will map to ["d", "trulyleqd"]
NUM_HIDDEN_LAYERS = 1
CAUSAL = True
PE_CODE_RANGE = "A-G" # "G-M" # "O-U" #
LOSS_OR_ACC = "acc"
MAX_TRAIN_D = 8

LOGFILE = "stress_test_negatives_metrics.txt"
SPLIT = "stress_test_negatives"
NOISE_STD = 0.1 # for jittering x values for better visualization

# ------------------------------------------------------------------
# Useful mappings for and plotting
# ------------------------------------------------------------------

def SPLITNAME_TO_XTICKLABEL(split_name: str) -> str:
    
    if split_name.startswith("one_"):
        x = split_name.split("_")
        return "Type1"
    
    elif split_name.startswith("two_"):
        if len(x) == 3:
            side = x[-1]
            return "Type2-" + side.capitalize()
        else: return "Type2"
    elif split_name.startswith("three_"):
        if len(x) == 3:
            side = x[-1]
            return "Type3-" + side.capitalize()
        else: return "Type3"
    elif split_name.startswith("four_"):
        if len(x) == 3:
            side = x[-1]
            return "Type4-" + side.capitalize()
        else: return "Type4"
    elif split_name.startswith("five_"):
        if len(x) == 3:
            side = x[-1]
            return "Type5-" + side.capitalize()
        else: return "Type5"
    
    else: raise ValueError(f"Unexpected split_name: {split_name}")

SPLITNAME_TO_X = {
    "one_rfv": 1,
    "two_rfv_in": 2,
    "two_rfv_out": 3,
    "three_rfv_in": 4,
    "three_rfv_out": 5,
    "four_rfv_in": 6,
    "four_rfv_out": 7,
    "five_rfv_in": 8,
    "five_rfv_out": 9,
    "one_pfv": 1,
    "two_pfv_in": 2,
    "two_pfv_out": 3,
    "three_pfv_in": 4,
    "three_pfv_out": 5,
    "four_pfv_in": 6,
    "four_pfv_out": 7,
    "five_pfv_in": 8,
    "five_pfv_out": 9,
    "one_ratiop": 1,
    "two_ratiop": 3,
    "three_ratiop": 5,
    "four_ratiop_in": 6,
    "four_ratiop_out": 7,
    "five_ratiop_in": 8,
    "five_ratiop_out": 9,
}

SHORT_VAR_TO_ROW_IDX = {
    "rfv": 0,
    "pfv": 1,
    "ratiop": 2,
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

PE_NAME_TO_PLOT_ORDER = {
	"rotary": 0,
	"scaler": 1,
	"sinusoidal": 2,
	"absolute": 3,
	"alibi": 4,
	"relative": 5,
	"none": 6
}

def Print(s=""): print(s, flush=True)


# ------------------------------------------------------------------
# 1. Get Task Suites
# ------------------------------------------------------------------

def get_task_suites(output_dir, supervision, depth_mode, num_hidden_layers, pe_code_range):
    
    # since this is a regex, use a raw f-string for readability:
    regex_pattern = (
        rf"{supervision}_d_{depth_mode[:-1]}"
        rf"\[\d+-\d+\]_"
        rf"{num_hidden_layers}[{pe_code_range}]"
    )

    task_suites = []
    for task_suite in os.listdir(output_dir):
        if re.match(regex_pattern, task_suite):
            task_suites.append(task_suite)
    
    Print("="*14 + " Found task_suites: " + "="*14 + f"\n{task_suites}\n")
    
    return task_suites


# ------------------------------------------------------------------
# 2. Get Plot Stats
# ------------------------------------------------------------------

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
        Print(f"Func read_logfile(): [skip] Missing logfile: {task_suite}/{task}/{handle}/{logfile}")
        return None
    
    with open(log_path, "r") as f: lines = f.readlines()
    return lines


def locate_best_epoch_lineidx(lines, split):
    for i in range(len(lines)):
        line = lines[i]
        if f"Metrics on {split} Sets:" in line:
            return i
    Print(f"Func locate_best_epoch_lineidx(): [skip] Cannot find best epoch line for split {split}")
    return None


def get_loss_or_acc(metrics, loss_or_acc):
    # metrics: a dict recovered from log
    # e.g. {'num_samples': 10000, 'FP': 0.0, 'TP': 0.88, 'FN': 0.12, 'TN': 0.0, 'loss': 0.41, 'acc': 0.88}
    return metrics[loss_or_acc]

# ------------------------------------------------------------------
# 3. Plotting
# ------------------------------------------------------------------

def plot(plot_stats, args, save_path):
    title = f"{args.supervision} {args.num_hidden_layers}-Layer, {'Causal' if args.causal else 'Non-causal'}"
    # x_label = 
    # y_label = 

    fig, axes = plt.subplots(
        3, 1,
        figsize=(5, 10)
    )
    for pe_name in plot_stats:
        X, Y, Y_err = [[], [], []], [[], [], []], [[], [], []] # three rows
        
        for split_name in plot_stats[pe_name]:
            short_var = split_name.split("_")[1] # e.g. "rfv", "pfv", "ratiop"
            row_idx = SHORT_VAR_TO_ROW_IDX[short_var]
            # randomly sample noise from a normal distribution with mean=0 and std=args.noise_std to add to the x values for better visualization
            x = SPLITNAME_TO_X[split_name] + np.random.normal(0, args.noise_std)
            X[row_idx].append(x)
            Y[row_idx].append(plot_stats[pe_name][split_name]["mean"])
            Y_err[row_idx].append(plot_stats[pe_name][split_name]["std"])
        
        Y_err = np.asarray(Y_err)
        lower = np.clip(Y - Y_err, 0, 1)
        upper = np.clip(Y + Y_err, 0, 1)
        yerr_clipped = np.vstack([
            Y - lower,   # distance from marker to lower cap
            upper - Y,   # distance from marker to upper cap
        ])
        # markers only: more opaque
        ax.scatter(
            X, Y,
            marker='s',
            facecolors='none',
            edgecolors=PE_NAME_TO_COLOR[pe_name],
            linewidths=1.2,
            alpha=0.4,
            label=pe_name,
            s=50,
        )
        ax.errorbar(
            X, Y,
            yerr=yerr_clipped,       # vertical error bar size
            fmt='none',              # circle marker
            ecolor=PE_NAME_TO_COLOR[pe_name], # error bar color
            capsize=4,               # cap width
            capthick=1.2,
            elinewidth=1.2,
            linestyle='none',        # no connecting line
            alpha=0.2,
        )
    ax.set_title(title, fontsize=12)
    ax.set_xticks(X)
    ax.set_xticklabels([SPLITNAME_TO_XTICKLABEL(split_name) for split_name in SPLITNAME_TO_X], rotation=30, ha='right', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(save_path, bbox_inches="tight", dpi=args.dpi)
    plt.close(fig)

    # save plot_stats to the same dir
    save_stats_path = save_path.replace(".png", ".json")
    with open(save_stats_path, "w") as f: json.dump(plot_stats, f, indent=4)
    
    Print("\n" + "="*19 + "Saving ..." + "="*19)
    Print(f"Plot destination: {save_path}\nStats destination: {save_stats_path}\n")



# Example of key contents in log files
"""
--------------------------------------------------
Epoch 9 Metrics on stress_test_positives Sets:

stress_test_positives set: Lleq40_RoDleq40
{'num_samples': 10000, 'FP': 0.0, 'TP': 1.0, 'FN': 0.0, 'TN': 0.0, 'loss': 0.0, 'acc': 1.0}
stress_test_positives set: L42to80_RoDleq40
{'num_samples': 10000, 'FP': 0.0, 'TP': 1.0, 'FN': 0.0, 'TN': 0.0, 'loss': 0.0, 'acc': 1.0}
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
            args.pe_code_range.upper() if args.causal else args.pe_code_range.lower()
        )
    
    pe_codes = sorted([task_suite[-1] for task_suite in task_suites])
    task_suite_full = task_suites[0][:-1] + "".join(pe_codes)
    a, b, c, d = task_suites[0].split("_")
    task_suite_singletask = "_".join([a, b]) + f"_{c.split('[')[0]}{args.max_train_d}_" + d[:-1] + "".join(pe_codes)

    ### 3. Get Plot Stats
    logfile, split = args.logfile, args.split
    plot_stats = defaultdict(lambda: defaultdict(dict)) # plot_stats[pe_name][split_name] = {"mean": mean, "std": std}
    
    for task_suite in task_suites:
        pe_code = task_suite[-1]
        pe_name = PE_CODE_TO_NAME[pe_code.upper()]
        for task in os.listdir(os.path.join(args.output_dir, task_suite)):
            train_d = int(task[-1])
            if not train_d == MAX_TRAIN_D: continue # only plot for a single value of max_train_d
            for handle in os.listdir(os.path.join(args.output_dir, task_suite, task)):

                lines = read_logfile(args.output_dir, task_suite, task, handle, logfile)
                if lines is None: continue

                best_epoch_lineidx = locate_best_epoch_lineidx(lines, split)
                if best_epoch_lineidx is None: continue
                if lines[best_epoch_lineidx+1].strip() == "": 
                    # some <split>_metrics.txt files have an extra empty line after the keyline.
                    best_epoch_lineidx += 1

                for i in range(best_epoch_lineidx+1, len(lines), 2):
                    line1 = lines[i].strip()
                    if line1 == "": break
                    if not line1.startswith(f"{split} set: "):
                        Print(f"Func plot(): [skip] Unexpected line format (expecting {split} set: d_x): {line1}")
                        continue
                    split_name = line1.split(': ')[-1] # extract split_name from line1

                    line2 = lines[i+1].strip()
                    try:
                        metrics = eval(line2)
                        y = get_loss_or_acc(metrics, args.loss_or_acc)
                        if y == -1: continue # skip when p_acc/n_acc with no positive/negative samples)
                        plot_stats[pe_name][split_name].setdefault("values", []).append(y)
                    except:
                        Print(f"Func plot(): [skip] Cannot parse metrics as dict from line {i+1}: {line2}")
                        continue
                
            
    # compute mean and std for each (pe_name, split_name)
    for pe_name in plot_stats:
        for split_name in plot_stats[pe_name]:
            values = plot_stats[pe_name][split_name].get("values", [])
            if len(values) == 0:
                Print(f"Func plot(): [skip] No values to compute mean and std for {pe_name} split_name={split_name}")
                continue
            mean = sum(values) / len(values)
            std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
            plot_stats[pe_name][split_name]["mean"] = mean
            plot_stats[pe_name][split_name]["std"] = std

    # sort plot_stats.keys() according to PE_NAME_TO_PLOT_ORDER
    plot_stats = dict(sorted(plot_stats.items(), key=lambda item: PE_NAME_TO_PLOT_ORDER[item[0]]))
    
    ### 5. Plotting
    figure_name = f"{task_suite_singletask}_{args.loss_or_acc}.png"
    save_dir = os.path.join(".", f"{args.split[:-6]}_singletask", task_suite_full)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, figure_name)
    plot(plot_stats, args, save_path)


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
        "--max_train_d",
        type=int,
        default=MAX_TRAIN_D,
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=NUM_HIDDEN_LAYERS,
        help="Number of hidden layers in the model"
    )
    parser.add_argument(
        "--causal",
        type=bool,
        default=CAUSAL,
        help="Whether the model is causal (applicable for Transformer)"
    )
    parser.add_argument(
        "--pe_code_range",
        type=str,
        default=PE_CODE_RANGE,
        help="Range of PE codes to include: e.g. 'A-G' or 'O-U'"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for the saved plots"
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default=LOGFILE,
        help="Name of the logfile to read metrics from"
    )
    parser.add_argument(
        "--split",
        type=str,
        default=SPLIT,
        help="Name of the split to read metrics for"
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=NOISE_STD,
        help="Standard deviation of noise to add to x values for jittering"
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
        args.pe_code_range.upper() if args.causal else args.pe_code_range.lower()
    )
    run_all(args, task_suites)
