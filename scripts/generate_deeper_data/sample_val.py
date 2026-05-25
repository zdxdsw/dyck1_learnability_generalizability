import sys, atexit

_original_stdout = sys.stdout
_original_stderr = sys.stderr

log_file = open("terminal_sample_val.txt", "w", buffering=1)

class Tee:
    def __init__(self, *files): self.files = files

    def write(self, obj):
        for f in self.files:
            try:
                f.write(obj)
                f.flush()
            except ValueError:
                pass

    def flush(self):
        for f in self.files:
            try: f.flush()
            except ValueError: pass

sys.stdout = Tee(_original_stdout, log_file)
sys.stderr = Tee(_original_stderr, log_file)

def cleanup(): # restoring sys.stdout and sys.stderr before closing terminal.txt
    sys.stdout = _original_stdout
    sys.stderr = _original_stderr
    log_file.close()

atexit.register(cleanup)

import random, os, pytz, json, math
from tqdm import tqdm, trange
from datetime import datetime
timezone = pytz.timezone('America/New_York') 
from collections import Counter
from utils import *

pos_data_dir = "/data/yingshac/tmlr/deeper_tmp/positives"
neg_data_dir = "/data/yingshac/tmlr/deeper_tmp/negatives"
val_data_dir = "/data/yingshac/tmlr/deeper_tmp/validation"


""" Validation: Positive samples
    total_val_pos_samples = [5000, 5000, 5000, 5000, 4000, 1500, 500, 160], for d=9-16 respectively.

    Negative samples
        Roughly 1/4 resulted from insertion/deletion; 3/4 resulted from swapping/displacement/grafting.
    
    Validation: Negative (insertion/deletion) samples
    total_val_neg_in_de_samples = [1300, 1300, 1300, 1300, 1000, 400, 130, 40], for d=9-16 respectively.

    Validation: Negative (swapping/displacement/grafting) samples
    total_val_neg_sw_dp_gr_samples = [3700, 3700, 3700, 3700, 3000, 1100, 370, 120], for d=9-16 respectively.

    Output json format:
    <val_data_dir>/fix_d_{d}.json = 
    {
        "pos": {
            "l": [s1, s2, ...],
            ...        },
        "neg": {
            "l": [s1, s2, ...],
            ...        }
    }
"""

"""
    For each d, total_val_samples[d] = V. 
    Then suppose the universe of positive samples for d has U samples.
    Sample positive samples in each <$d, \ell$> bucket proportionally. Proportion $=\frac{V}{U} \times 100\%$. 
    Put these x samples to validation set.
"""

total_val_pos_samples = [5000, 5000, 5000, 5000, 4000, 1500, 500, 160] # for d=9-16 respectively.
total_val_neg_in_de_samples = [1300, 1300, 1300, 1300, 1000, 400, 130, 40] # for d=9-16 respectively.
total_val_neg_sw_dp_gr_samples = [3700, 3700, 3700, 3700, 3000, 1100, 370, 120] # for d=9-16 respectively.

for d in range(9, 17):
    train_json = {"pos": {}, "neg": {}}
    val_json = {"pos": {}, "neg": {}}
    V = total_val_pos_samples[d-9]
    V_neg1, V_neg2 = total_val_neg_in_de_samples[d-9], total_val_neg_sw_dp_gr_samples[d-9]

    # Load the universe of pos and neg samples for this d
    with open(os.path.join(pos_data_dir, f"fix_d_{d}.json"), 'r') as f:
        pos_len_to_strings = json.load(f)
    with open(os.path.join(neg_data_dir, f"fix_d_{d}.json"), 'r') as f:
        neg_len_to_strings = json.load(f)
    
    U = sum([len(pos_len_to_strings[str(l)]) for l in pos_len_to_strings])
    U_neg1 = sum([len(neg_len_to_strings[str(l)]) for l in neg_len_to_strings if int(l)%2==1])
    U_neg2 = sum([len(neg_len_to_strings[str(l)]) for l in neg_len_to_strings if int(l)%2==0])

    for l in range(2*d, 39, 2):
        # Sample positive samples in each l bucket proportionally. Proportion V/U. 
        bucket_size = len(pos_len_to_strings[str(l)])
        proportion = V/U
        x = math.floor(proportion * bucket_size)
        # randomly choose x samples from this bucket
        val_json["pos"][str(l)] =  random.sample(pos_len_to_strings[str(l)], x)

        # Sample negative samples in each l bucket proportionally. Proportion V_neg2/U_neg2.
        bucket_size = len(neg_len_to_strings[str(l)])
        proportion = V_neg2/U_neg2
        x = math.floor(proportion * bucket_size)
        # randomly choose x samples from this bucket
        val_json["neg"][str(l)] = random.sample(neg_len_to_strings[str(l)], x)

    for l in range(2*d-1, 38, 2):
        # Sample negative samples in each l bucket proportionally. Proportion V_neg1/U_neg1.
        bucket_size = len(neg_len_to_strings[str(l)])
        proportion = V_neg1/U_neg1
        x = math.floor(proportion * bucket_size)
        # randomly choose x samples from this bucket
        val_json["neg"][str(l)] = random.sample(neg_len_to_strings[str(l)], x)

    # Because I keep using math.floor, so I will end up undersampling each l. 
    # Make it up by adding samples from l=40 for pos and neg2; from l=39 for neg1.
    last_bucket_val_pos_needed = total_val_pos_samples[d-9] - sum([len(val_json["pos"][k]) for k in val_json["pos"]])
    last_bucket_val_neg1_needed = total_val_neg_in_de_samples[d-9] - sum([len(val_json["neg"][k]) for k in val_json["neg"] if int(k)%2==1])
    last_bucket_val_neg2_needed = total_val_neg_sw_dp_gr_samples[d-9] - sum([len(val_json["neg"][k]) for k in val_json["neg"] if int(k)%2==0])
    
    l = 40; print("\nl=40 for pos samples")
    bucket_size = len(pos_len_to_strings[str(l)])
    proportion = V/U
    x = math.floor(proportion * bucket_size)
    actual_x = last_bucket_val_pos_needed
    print(f"calculated x={x}, actual_x={actual_x}")
    # randomly choose actual_x samples from this bucket
    val_json["pos"][str(l)] = random.sample(pos_len_to_strings[str(l)], actual_x)

    print(f"\nl=40 for neg2 samples")
    bucket_size = len(neg_len_to_strings[str(l)])
    proportion = V_neg2/U_neg2
    x = math.floor(proportion * bucket_size)
    actual_x = last_bucket_val_neg2_needed
    print(f"calculated x={x}, actual_x={actual_x}")
    # randomly choose actual_x samples from this bucket
    val_json["neg"][str(l)] = random.sample(neg_len_to_strings[str(l)], actual_x)    
    
    l=39; print(f"\nl=39 for neg1 samples")
    bucket_size = len(neg_len_to_strings[str(l)])
    proportion = V_neg1/U_neg1
    x = math.floor(proportion * bucket_size)
    actual_x = last_bucket_val_neg1_needed
    print(f"calculated x={x}, actual_x={actual_x}")
    # randomly choose actual_x samples from this bucket
    val_json["neg"][str(l)] = random.sample(neg_len_to_strings[str(l)], actual_x)    

    # save val_json files
    with open(os.path.join(val_data_dir, f"fix_d_{d}.json"), 'w') as f:
        json.dump(val_json, f, indent=2)
    print(f"\nfinish fix_d={d}")
    print(f"val samples={sum([len(val_json['pos'][k]) for k in val_json['pos']]) + sum([len(val_json['neg'][k]) for k in val_json['neg']])},", end="")
    print(f" (pos={sum([len(val_json['pos'][k]) for k in val_json['pos']])}, neg={sum([len(val_json['neg'][k]) for k in val_json['neg']])})")
    print("\n\n")



import matplotlib.pyplot as plt
import numpy as np

""" Training: plot summary statistics
    Save figures to "figures/train_fix_d_{d}_{pos/neg/overall}distribution.png"
    15k pos and 15k neg samples for d=2
    50k pos and 50k neg samples for d=3-8
"""
"""
for d in trange(2, 9):
    with open(os.path.join(train_data_dir, f"fix_d_{d}.json"), 'r') as f:
        train_json = json.load(f)
    pos_len_counts = {l: len(train_json["pos"][str(l)]) for l in train_json["pos"]}
    neg_len_counts = {l: len(train_json["neg"][str(l)]) for l in train_json["neg"]}
    # sort by int(l)
    pos_len_counts = dict(sorted(pos_len_counts.items(), key=lambda x: int(x[0])))
    neg_len_counts = dict(sorted(neg_len_counts.items(), key=lambda x: int(x[0])))

    ### plot distribution of l for pos samples, annotate (small font) on top of each bar
    plt.figure(figsize=(10, 6))
    x = list(pos_len_counts.keys())
    bars = plt.bar(x, pos_len_counts.values(), color='lightgreen')
    plt.title(f"Distribution of length for fix_d={d} (pos training samples)")
    plt.xlabel("length")
    plt.ylabel("Count")
    # Annotate each bar with its value
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 '%d' % int(height),
                 ha='center', va='bottom', fontsize=6)
    plt.savefig(f"figures/train_fix_d_{d}_pos_distrOfLength.png", dpi=250, bbox_inches='tight')
    plt.close()

    ### plot distribution of l for neg samples, annotate on top of each bar
    plt.figure(figsize=(10, 6))
    x = list(neg_len_counts.keys())
    bars = plt.bar(x, neg_len_counts.values(), color='lightcoral')
    plt.title(f"Distribution of length for fix_d={d} (neg training samples)")
    plt.xlabel("length")
    plt.ylabel("Count")

    # Select xticks and xticklabels: first; last; 6 evenly spaced middle positions
    tick_idx = np.linspace(0, len(x)-1, 8, dtype=int)
    plt.xticks(tick_idx, [str(x[i]) for i in tick_idx])

    # Annotate each bar with its value
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 '%d' % int(height),
                 ha='center', va='bottom', fontsize=6)
    plt.savefig(f"figures/train_fix_d_{d}_neg_distrOfLength.png", dpi=250, bbox_inches='tight')
    plt.close()

    ### plot distribution of l for combined pos and neg samples, annotate on top of each bar
    plt.figure(figsize=(10, 6))
    x = sorted(set(pos_len_counts.keys()) | set(neg_len_counts.keys()), key=lambda x: int(x))
    pos_values = [pos_len_counts.get(k, 0) for k in x]
    neg_values = [neg_len_counts.get(k, 0) for k in x]
    values = [pos_values[i] + neg_values[i] for i in range(len(x))]
    bars = plt.bar(x, values, color='lightblue')
    plt.title(f"Distribution of length for fix_d={d} (combined training samples)")
    plt.xlabel("length")
    plt.ylabel("Count")

    # Select xticks and xticklabels: first; last; 6 evenly spaced middle positions
    tick_idx = np.linspace(0, len(x)-1, 8, dtype=int)
    plt.xticks(tick_idx, [str(x[i]) for i in tick_idx])

    # Annotate each bar with its value
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 '%d' % int(height),
                 ha='center', va='bottom', fontsize=6)
    plt.savefig(f"figures/train_fix_d_{d}_overall_distrOfLength.png", dpi=250, bbox_inches='tight')
    plt.close()
"""
""" Validation: plot summary statistics
    Save figures to "figures/val_fix_d_{d}_{pos/neg/overall}distribution.png"
    5k pos and 5k neg samples for d=2-8
"""
"""
for d in trange(2, 9):
    with open(os.path.join(val_data_dir, f"fix_d_{d}.json"), 'r') as f:
        val_json = json.load(f)
    pos_len_counts = {l: len(val_json["pos"][str(l)]) for l in val_json["pos"]}
    neg_len_counts = {l: len(val_json["neg"][str(l)]) for l in val_json["neg"]}
    # sort by int(l)
    pos_len_counts = dict(sorted(pos_len_counts.items(), key=lambda x: int(x[0])))
    neg_len_counts = dict(sorted(neg_len_counts.items(), key=lambda x: int(x[0])))

    ### plot distribution of l for pos samples, annotate on top of each bar
    plt.figure(figsize=(10, 6))
    x = list(pos_len_counts.keys())
    bars = plt.bar(x, pos_len_counts.values(), color='lightgreen')
    plt.title(f"Distribution of length for fix_d={d} (pos validation samples)")
    plt.xlabel("length")
    plt.ylabel("Count")
    # Annotate each bar with its value
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 '%d' % int(height),
                 ha='center', va='bottom', fontsize=6)
    plt.savefig(f"figures/val_fix_d_{d}_pos_distrOfLength.png", dpi=250, bbox_inches='tight')
    plt.close()

    ### plot distribution of l for neg samples, annotate on top of each bar
    plt.figure(figsize=(10, 6))
    x = list(neg_len_counts.keys())
    bars = plt.bar(x, neg_len_counts.values(), color='lightcoral')
    plt.title(f"Distribution of length for fix_d={d} (neg validation samples)")
    plt.xlabel("length")
    plt.ylabel("Count")
    # Annotate each bar with its value
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 '%d' % int(height),
                 ha='center', va='bottom', fontsize=6)
    plt.savefig(f"figures/val_fix_d_{d}_neg_distrOfLength.png", dpi=250, bbox_inches='tight')
    plt.close()

    ### plot distribution of l for combined pos and neg samples, annotate on top of each bar
    plt.figure(figsize=(10, 6))
    x = sorted(set(pos_len_counts.keys()) | set(neg_len_counts.keys()), key=lambda x: int(x))
    pos_values = [pos_len_counts.get(k, 0) for k in x]
    neg_values = [neg_len_counts.get(k, 0) for k in x]
    values = [pos_values[i] + neg_values[i] for i in range(len(x))]
    bars = plt.bar(x, values, color='lightblue')
    plt.title(f"Distribution of length for fix_d={d} (combined validation samples)")
    plt.xlabel("length")
    plt.ylabel("Count")
    # Annotate each bar with its value
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 '%d' % int(height),
                 ha='center', va='bottom', fontsize=6)
    plt.savefig(f"figures/val_fix_d_{d}_overall_distrOfLength.png", dpi=250, bbox_inches='tight')
    plt.close()
"""