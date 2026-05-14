"""
import sys, atexit

_original_stdout = sys.stdout
_original_stderr = sys.stderr

log_file = open("terminal_sample_train_and_val.txt", "w", buffering=1)

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
"""

# -------------------------------------------------------------------------------------------
import random, os, pytz, json, math
from tqdm import tqdm, trange
from datetime import datetime
timezone = pytz.timezone('America/New_York') 
from collections import Counter

""" Training & Validation: Positive samples
    15000 samples for d=2
    50000 samples for d=3-8
    5000 val_samples for d=2-8

    Training & Validation: Negative (insertion/deletion) samples
    4000 samples for d=2
    13000 samples for d=3-8
    1300 val_samples for d=2-8

    Training & Validation: Negative (swapping/displacement/grafting) samples
    11000 samples for d=2
    37000 samples for d=3-8
    3700 samples for d=2-8

    Output json format:
    <train_data_dir>/fix_d_{d}.json = 
    {
        "pos": {
            "l": [s1, s2, ...],
            ...        },
        "neg": {
            "l": [s1, s2, ...],
            ...        }
    }
"""

pos_data_dir = "/data/yingshac/tmlr/positives"
neg_data_dir = "/data/yingshac/tmlr/negatives"
train_data_dir = "/data/yingshac/tmlr/train"
val_data_dir = "/data/yingshac/tmlr/validation"
"""
total_samples = [15000] + [50000]*6
total_val_samples = [5000]*7
total_neg_in_de_samples = [4000] + [13000]*6
total_val_neg_in_de_samples = [1300]*7
total_neg_sw_dp_gr_samples = [11000] + [37000]*6
total_val_neg_sw_dp_gr_samples = [3700]*7

for d in range(2, 9):
    train_json = {"pos": {}, "neg": {}}
    val_json = {"pos": {}, "neg": {}}
    T, V = total_samples[d-2], total_val_samples[d-2]
    T_neg1, V_neg1 = total_neg_in_de_samples[d-2], total_val_neg_in_de_samples[d-2]
    T_neg2, V_neg2 = total_neg_sw_dp_gr_samples[d-2], total_val_neg_sw_dp_gr_samples[d-2]

    # Load the universe of pos and neg samples for this d
    with open(os.path.join(pos_data_dir, f"fix_d_{d}.json"), 'r') as f:
        pos_len_to_strings = json.load(f)
    with open(os.path.join(neg_data_dir, f"fix_d_{d}.json"), 'r') as f:
        neg_len_to_strings = json.load(f)
    
    # Put all positives samples (y) where l = 2d, 2d+2, 2d+4 in training set.
    y = 0
    U = sum([len(pos_len_to_strings[str(l)]) for l in pos_len_to_strings])
    for l in [2*d, 2*d+2, 2*d+4]:
        y += len(pos_len_to_strings[str(l)])
        train_json["pos"][str(l)] = pos_len_to_strings[str(l)]
    T -= y; U -= y
    print(f"fix_d={d}, put all {y} samples in training set")

    # Put all negatives insertion/deletion samples (y_neg1) where l = 2d-1, 2d+1, 2d+3 in training set.
    y_neg1 = 0
    U_neg1 = sum([len(neg_len_to_strings[str(l)]) for l in neg_len_to_strings if int(l)%2==1])
    for l in [2*d-1, 2*d+1, 2*d+3]:
        y_neg1 += len(neg_len_to_strings[str(l)])
        train_json["neg"][str(l)] = neg_len_to_strings[str(l)]
    T_neg1 -= y_neg1; U_neg1 -= y_neg1
    print(f"fix_d={d}, put all {y_neg1} insertion/deletion samples in training set")

    # Put all negatives swapping/displacement/grafting samples (y_neg2) where l = 2d, 2d+2, 2d+4 in training set.
    y_neg2 = 0
    U_neg2 = sum([len(neg_len_to_strings[str(l)]) for l in neg_len_to_strings if int(l)%2==0])
    for l in [2*d, 2*d+2, 2*d+4]:
        y_neg2 += len(neg_len_to_strings[str(l)])
        train_json["neg"][str(l)] = neg_len_to_strings[str(l)]
    T_neg2 -= y_neg2; U_neg2 -= y_neg2
    print(f"fix_d={d}, put all {y_neg2} swapping/displacement/grafting samples in training set")

    for l in range(2*d+6, 39, 2):
        # Sample positive samples in each l bucket proportionally. Proportion (T+V-y)/(U-y). 
        bucket_size = len(pos_len_to_strings[str(l)])
        proportion = (T+V)/(U)
        x = math.floor(proportion * bucket_size)
        # randomly choose x samples from this bucket
        sampled_pos = random.sample(pos_len_to_strings[str(l)], x)
        # put round(T/(T+V)) of them to training set; and the rest to validation set.
        train_count = math.floor(T/(T+V) * x)
        val_count = math.floor(V/(T+V) * x)
        assert train_count + val_count <= x
        train_json["pos"][str(l)] = sampled_pos[:train_count]
        val_json["pos"][str(l)] = sampled_pos[train_count:train_count+val_count]
        # print(f"\npos samples for fix_d={d}, l={l}: \n bucket_size={bucket_size}, proportion={proportion:.2f}, x={x}, train_count={train_count}, val_count={val_count}")

        # Sample negative samples in each l bucket proportionally. Proportion (T_neg2+V_neg2-y_neg2)/(U_neg2-y_neg2).
        bucket_size = len(neg_len_to_strings[str(l)])
        proportion = (T_neg2+V_neg2)/(U_neg2)
        x = math.floor(proportion * bucket_size)
        # randomly choose x samples from this bucket
        sampled_neg2 = random.sample(neg_len_to_strings[str(l)], x)
        # put round(T_neg2/(T_neg2+V_neg2)) of them to training set; and the rest to validation set.
        train_count = math.floor(T_neg2/(T_neg2+V_neg2) * x)
        val_count = math.floor(V_neg2/(T_neg2+V_neg2) * x)
        assert train_count + val_count <= x
        train_json["neg"][str(l)] = sampled_neg2[:train_count]
        val_json["neg"][str(l)] = sampled_neg2[train_count:train_count+val_count]
        # print(f"\nneg (sw/dp/gr) samples for fix_d={d}, l={l}: \n bucket_size={bucket_size}, proportion={proportion:.2f}, x={x}, train_count={train_count}, val_count={val_count}")


    for l in range(2*d+5, 38, 2):
        # Sample negative samples in each l bucket proportionally. Proportion (T_neg1+V_neg1-y_neg1)/(U_neg1-y_neg1).
        bucket_size = len(neg_len_to_strings[str(l)])
        proportion = (T_neg1+V_neg1)/(U_neg1)
        x = math.floor(proportion * bucket_size)
        # randomly choose x samples from this bucket
        sampled_neg1 = random.sample(neg_len_to_strings[str(l)], x)
        # put round(T_neg1/(T_neg1+V_neg1)) of them to training set; and the rest to validation set.
        train_count = math.floor(T_neg1/(T_neg1+V_neg1) * x)
        val_count = math.floor(V_neg1/(T_neg1+V_neg1) * x)
        assert train_count + val_count <= x
        train_json["neg"][str(l)] = sampled_neg1[:train_count]
        val_json["neg"][str(l)] = sampled_neg1[train_count:train_count+val_count]
        # print(f"\nneg (in/de) samples for fix_d={d}, l={l}: \n bucket_size={bucket_size}, proportion={proportion:.2f}, x={x}, train_count={train_count}, val_count={val_count}")

    # Because I keep using math.floor, so I will end up undersampling each l. 
    # Make it up by adding samples from l=40 for pos and neg2; from l=39 for neg1.
    last_bucket_train_pos_needed = total_samples[d-2] - sum([len(train_json["pos"][k]) for k in train_json["pos"]])
    last_bucket_val_pos_needed = total_val_samples[d-2] - sum([len(val_json["pos"][k]) for k in val_json["pos"]])
    last_bucket_train_neg1_needed = total_neg_in_de_samples[d-2] - sum([len(train_json["neg"][k]) for k in train_json["neg"] if int(k)%2==1])
    last_bucket_val_neg1_needed = total_val_neg_in_de_samples[d-2] - sum([len(val_json["neg"][k]) for k in val_json["neg"] if int(k)%2==1])
    last_bucket_train_neg2_needed = total_neg_sw_dp_gr_samples[d-2] - sum([len(train_json["neg"][k]) for k in train_json["neg"] if int(k)%2==0])
    last_bucket_val_neg2_needed = total_val_neg_sw_dp_gr_samples[d-2] - sum([len(val_json["neg"][k]) for k in val_json["neg"] if int(k)%2==0])
    
    l = 40; print("\nl=40 for pos samples")
    bucket_size = len(pos_len_to_strings[str(l)])
    proportion = (T+V)/(U)
    x = math.floor(proportion * bucket_size)
    actual_x = last_bucket_train_pos_needed + last_bucket_val_pos_needed
    train_count = math.floor(T/(T+V) * x)
    val_count = math.floor(V/(T+V) * x)
    assert train_count + val_count <= x
    print(f"calculated x={x}, actual_x={actual_x}")
    print(f"calculated train_count={train_count}, val_count={val_count}, actual_train_count={last_bucket_train_pos_needed}, actual_val_count={last_bucket_val_pos_needed}")
    # randomly choose actual_x samples from this bucket
    sampled_pos = random.sample(pos_len_to_strings[str(l)], actual_x)
    train_json["pos"][str(l)] = sampled_pos[:last_bucket_train_pos_needed]
    val_json["pos"][str(l)] = sampled_pos[last_bucket_train_pos_needed:last_bucket_train_pos_needed+last_bucket_val_pos_needed]

    print(f"\nl=40 for neg2 samples")
    bucket_size = len(neg_len_to_strings[str(l)])
    proportion = (T_neg2+V_neg2)/(U_neg2)
    x = math.floor(proportion * bucket_size)
    actual_x = last_bucket_train_neg2_needed + last_bucket_val_neg2_needed
    train_count = math.floor(T_neg2/(T_neg2+V_neg2) * x)
    val_count = math.floor(V_neg2/(T_neg2+V_neg2) * x)
    assert train_count + val_count <= x
    print(f"calculated x={x}, actual_x={actual_x}")
    print(f"calculated train_count={train_count}, val_count={val_count}, actual_train_count={last_bucket_train_neg2_needed}, actual_val_count={last_bucket_val_neg2_needed}")
    # randomly choose actual_x samples from this bucket
    sampled_neg2 = random.sample(neg_len_to_strings[str(l)], actual_x)    
    train_json["neg"][str(l)] = sampled_neg2[:last_bucket_train_neg2_needed]
    val_json["neg"][str(l)] = sampled_neg2[last_bucket_train_neg2_needed:last_bucket_train_neg2_needed+last_bucket_val_neg2_needed]

    l=39; print(f"\nl=39 for neg1 samples")
    bucket_size = len(neg_len_to_strings[str(l)])
    proportion = (T_neg1+V_neg1)/(U_neg1)
    x = math.floor(proportion * bucket_size)
    actual_x = last_bucket_train_neg1_needed + last_bucket_val_neg1_needed
    train_count = math.floor(T_neg1/(T_neg1+V_neg1) * x)
    val_count = math.floor(V_neg1/(T_neg1+V_neg1) * x)
    assert train_count + val_count <= x
    print(f"calculated x={x}, actual_x={actual_x}")
    print(f"calculated train_count={train_count}, val_count={val_count}, actual_train_count={last_bucket_train_neg1_needed}, actual_val_count={last_bucket_val_neg1_needed}")
    # randomly choose actual_x samples from this bucket
    sampled_neg1 = random.sample(neg_len_to_strings[str(l)], actual_x)    
    train_json["neg"][str(l)] = sampled_neg1[:last_bucket_train_neg1_needed]
    val_json["neg"][str(l)] = sampled_neg1[last_bucket_train_neg1_needed:last_bucket_train_neg1_needed+last_bucket_val_neg1_needed]

    # save train_json and val_json
    with open(os.path.join(train_data_dir, f"fix_d_{d}.json"), 'w') as f:
        json.dump(train_json, f, indent=2)
    with open(os.path.join(val_data_dir, f"fix_d_{d}.json"), 'w') as f:
        json.dump(val_json, f, indent=2)
    print(f"\nfinish fix_d={d}")
    print(f"train samples={sum([len(train_json['pos'][k]) for k in train_json['pos']]) + sum([len(train_json['neg'][k]) for k in train_json['neg']])},", end="")
    print(f" (pos={sum([len(train_json['pos'][k]) for k in train_json['pos']])}, neg={sum([len(train_json['neg'][k]) for k in train_json['neg']])}),")
    print(f"val samples={sum([len(val_json['pos'][k]) for k in val_json['pos']]) + sum([len(val_json['neg'][k]) for k in val_json['neg']])},", end="")
    print(f" (pos={sum([len(val_json['pos'][k]) for k in val_json['pos']])}, neg={sum([len(val_json['neg'][k]) for k in val_json['neg']])})")
    print("\n\n")
"""
import matplotlib.pyplot as plt
import numpy as np
""" Training: plot summary statistics
    Save figures to "figures/train_fix_d_{d}_{pos/neg/overall}distribution.png"
    15k pos and 15k neg samples for d=2
    50k pos and 50k neg samples for d=3-8
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

""" Validation: plot summary statistics
    Save figures to "figures/val_fix_d_{d}_{pos/neg/overall}distribution.png"
    5k pos and 5k neg samples for d=2-8
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
