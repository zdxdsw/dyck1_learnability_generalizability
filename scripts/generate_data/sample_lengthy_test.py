import random, os, pytz, json
from tqdm import tqdm, trange
from datetime import datetime
timezone = pytz.timezone('America/New_York') 
from collections import Counter

""" Sample testing datasets

    Target quantities: for each <d, l> bucket, 
        200 positive samples; 
        50 (for odd l) and 150 (for even l) negative samples.
    
    Output json format:
    <test_data_dir>/fix_d_{d}.json = 
    {
        "pos": {
            "l": [s1, s2, ...],
            ...        },
        "neg": {
            "l": [s1, s2, ...],
            ...        }
    }
"""

pos_data_dir = "/data/yingshac/tmlr/lengthy_positives"
neg_data_dir = "/data/yingshac/tmlr/lengthy_negatives"
test_data_dir = "/data/yingshac/tmlr/test"

for d in trange(2, 9):
    test_json = {"pos": {}, "neg": {}}

    # Load the universe of pos and neg lengthy samples for this d
    with open(os.path.join(pos_data_dir, f"fix_d_{d}.json"), 'r') as f:
        pos_len_to_strings = json.load(f)
    with open(os.path.join(neg_data_dir, f"fix_d_{d}.json"), 'r') as f:
        neg_len_to_strings = json.load(f)

    for l in range(41, 121):
        neg_samples = neg_len_to_strings[str(l)]
        if l % 2 == 1: # odd length: only 50 negatives
            test_json["neg"][str(l)] = random.sample(neg_samples, 50)
        
        else: # even length: 150 negatives
            test_json["neg"][str(l)] = random.sample(neg_samples, 150)

            pos_samples = pos_len_to_strings[str(l)]
            test_json["pos"][str(l)] = random.sample(pos_samples, 200)
    
    with open(os.path.join(test_data_dir, f"fix_d_{d}.json"), 'w') as f:
        json.dump(test_json, f, indent=2)

import matplotlib.pyplot as plt
import numpy as np
""" Lengthy Test: plot summary statistics
    Save figures to "figures/test_fix_d_{d}_{pos/neg/overall}distribution.png"
    8k pos and 8k neg samples for d=2-8
"""
for d in trange(2, 9):
    with open(os.path.join(test_data_dir, f"fix_d_{d}.json"), 'r') as f:
        test_json = json.load(f)
    
    pos_len_counts = {l: len(test_json["pos"][str(l)]) for l in test_json["pos"]}
    neg_len_counts = {l: len(test_json["neg"][str(l)]) for l in test_json["neg"]}
    # sort by int(l)
    pos_len_counts = dict(sorted(pos_len_counts.items(), key=lambda x: int(x[0])))
    neg_len_counts = dict(sorted(neg_len_counts.items(), key=lambda x: int(x[0])))

    ### plot distribution of l for pos samples, annotate (small font) on top of each bar
    plt.figure(figsize=(10, 6))
    x = list(pos_len_counts.keys())
    bars = plt.bar(x, pos_len_counts.values(), color='lightgreen')
    plt.title(f"Distribution of length for fix_d={d} (pos testing samples)")
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
    plt.savefig(f"figures/test_fix_d_{d}_pos_distrOfLength.png", dpi=250, bbox_inches='tight')
    plt.close()

    ### plot distribution of l for neg samples, annotate on top of each bar
    plt.figure(figsize=(10, 6))
    x = list(neg_len_counts.keys())
    bars = plt.bar(x, neg_len_counts.values(), color='lightcoral')
    plt.title(f"Distribution of length for fix_d={d} (neg testing samples)")
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
    plt.savefig(f"figures/test_fix_d_{d}_neg_distrOfLength.png", dpi=250, bbox_inches='tight')
    plt.close()

    # plot distribution of l for combined pos and neg samples, annotate on top of each bar
    plt.figure(figsize=(10, 6))
    x = sorted(set(pos_len_counts.keys()) | set(neg_len_counts.keys()), key=lambda x: int(x))
    pos_values = [pos_len_counts.get(k, 0) for k in x]
    neg_values = [neg_len_counts.get(k, 0) for k in x]
    values = [pos_values[i] + neg_values[i] for i in range(len(x))]
    bars = plt.bar(x, values, color='lightblue')
    plt.title(f"Distribution of length for fix_d={d} (combined testing samples)")
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
    plt.savefig(f"figures/test_fix_d_{d}_overall_distrOfLength.png", dpi=250, bbox_inches='tight')
    plt.close()