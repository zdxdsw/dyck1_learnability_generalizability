import random, os, pytz, json
from tqdm import tqdm, trange
from datetime import datetime
timezone = pytz.timezone('America/New_York') 
from collections import Counter
from utils import *

### Generate Positives
data_dir = "/data/yingshac/tmlr/d_1"

"""
# open file handles for positives/negatives.txt
fh = open(os.path.join(data_dir, "positives.txt"), 'a')
# number of samples to generate per fix_d (probaly more than needed)
size = 5000
transition = [0.1, 0.0, 0.9]
# ---------------------------------------------------------------------------------------------------------------

for _ in tqdm(range(size), desc=f"Generating {size} data for fix_d=1"):
    s = generate(max_depth=1, depth=0, max_length=120, transition=transition)
    cur_d = double_check(s)
    if cur_d == 1: fh.write(s + "\n")
# close all file handles
fh.close()


# rewrite each txt file into json file with format: {"l": 20, "strings": [s1, s2, ...]}
len_to_strings = {l: [] for l in range(4, 121, 2)}
with open(os.path.join(data_dir, "positives.txt"), 'r') as f:
    for line in f:
        s = line.strip()
        l = len(s)
        if 4 <= l <= 120: len_to_strings[l].append(s)
# de-duplicate each list in len_to_strings
for l in len_to_strings: len_to_strings[l] = list(set(len_to_strings[l]))
# sort len_to_strings by l
len_to_strings = dict(sorted(len_to_strings.items(), key=lambda x: x[0]))
with open(os.path.join(data_dir, f"positives.json"), 'w') as f:
    json.dump(len_to_strings, f, indent=2)


# print, per fix_d, the number of unique strings of each length
from tabulate import tabulate
# break down l = 4-120 into row groups: l=4-22; 24-42; 44-62; 64-82; 84-102; 104-120+"total"
table = []
with open(os.path.join(data_dir, f"positives.json"), 'r') as f:
    len_to_strings = json.load(f)

intervals = [(4, 22), (24, 42), (44, 62), (64, 82), (84, 102), (104, 120)]
for e, interval in enumerate(intervals):
    # row1 looks like: "l", 4, 6, 8, ..., 22
    row1 = ["l"] + [str(l) for l in range(interval[0], interval[1]+1, 2)]
    # row2 looks like: "#unique pos", .., ..
    row2 = ["#unique pos"] + [len(len_to_strings[str(l)]) for l in range(interval[0], interval[1]+1, 2)]
    if e == len(intervals)-1:
        row1.append("total") 
        row2.append(sum([len(len_to_strings[str(l)]) for l in range(4, 121, 2)]))
    table.append(row1)
    table.append(row2)
print(tabulate(table, tablefmt="rounded_grid"))
"""

### Generate Negatives
"""
    Only include negatives that have an "off-by-one" error: 
    strings that differ from a positive by one insertion/swap/deletion
"""

"""
# do insert/delete/swap/move_last_close for TRIAL times on all of its positives; 
# I think grafting for d=1 is the same as displace_last_close, so skip it.
insert_trials, delete_trials, displace_trials = 1000, 1000, 1000
neg_len_to_strings = {l: set() for l in range(3, 121)}
with open(os.path.join(data_dir, f"positives.json"), 'r') as f:
    pos_len_to_strings = json.load(f)
for l in trange(4, 121, 2):
    for s in pos_len_to_strings[str(l)]:
        if l<120:
            for _ in range(insert_trials):
                neg_len_to_strings[l+1].add(insert_open(s))
                neg_len_to_strings[l+1].add(insert_close(s))
        for _ in range(delete_trials):
            neg_len_to_strings[l-1].add(delete_open(s))
            neg_len_to_strings[l-1].add(delete_close(s))
        for _ in range(displace_trials):
            candidate = displace_last_close(s)
            if double_check(candidate) == -1: neg_len_to_strings[l].add(candidate)
        while len(neg_len_to_strings[l]) < l:
            candidate = swap(s)
            if double_check(candidate) == -1: 
                neg_len_to_strings[l].add(candidate)
# double-check whether they are true negatives, then de-duplicate
for l in neg_len_to_strings:
    neg_len_to_strings[l] = list(set([s for s in neg_len_to_strings[l] if double_check(s) == -1]))
# sort neg_len_to_strings by l
neg_len_to_strings = dict(sorted(neg_len_to_strings.items(), key=lambda x: x[0]))
with open(os.path.join(data_dir, f"negatives.json"), 'w') as f:
    json.dump(neg_len_to_strings, f, indent=2)


# break down l = 3-120 into row groups: l=3-22; 23-42; 43-62; 63-82; 83-102; 103-120+"total"
table = []
with open(os.path.join(data_dir, f"negatives.json"), 'r') as f:
    len_to_strings = json.load(f)

intervals = [(3, 17), (18, 32), (33, 47), (48, 62), (63, 77), (78, 92), (93, 107), (108, 120)]
for e, interval in enumerate(intervals):
    # row1 looks like: "l", 3, 4, 5, ..., 17
    row1 = ["l"] + [str(l) for l in range(interval[0], interval[1]+1)]
    # row2 looks like: "#unique neg", .., ..
    row2 = ["#unique neg"] + [len(len_to_strings[str(l)]) for l in range(interval[0], interval[1]+1)]
    if e == len(intervals)-1:
        row1.extend([" ", " ", "total"]) 
        row2.extend([" ", " ",str(sum([len(len_to_strings[str(l)]) for l in range(3, 121)]))])
    table.append(row1)
    table.append(row2)
print(tabulate(table, tablefmt="rounded_grid"))


### Generate train/fix_d_1.json, val/fix_d_1.json for SHORTER
# there are 216 unique negatives for l=3-20
#   write all, duplicating 10 times to /data/yingshac/tmlr/shorter/train/fix_d_1.json; sample 9 and write to /data/yingshac/tmlr/shorter/validation/fix_d_1.json
# then 9 unique positives for l=3-20, write all, duplicating 240 times to /data/yingshac/tmlr/shorter/train/fix_d_1.json
#   write the same 9 unique positives to /data/yingshac/tmlr/shorter/validation/fix_d_1.json
train_json = {"pos": {}, "neg": {}}
val_json = {"pos": {}, "neg": {}}
with open(os.path.join(data_dir, f"negatives.json"), 'r') as f:
    len_to_strings = json.load(f)
    # aggregate all negative strings
    all_negatives = []
    for l in len_to_strings:
        if int(l)<=20: all_negatives.extend(list(len_to_strings[l]))
sample_val_negatives = random.sample(all_negatives, 9)
for l in len_to_strings:
    if int(l)<=20: 
        train_json["neg"][l] = len_to_strings[l] * 10
        val_json["neg"][l] = [s for s in len_to_strings[l] if s in sample_val_negatives]

with open(os.path.join(data_dir, f"positives.json"), 'r') as f:
    len_to_strings = json.load(f)
for l in len_to_strings:
    if int(l)<=20: 
        train_json["pos"][l] = len_to_strings[l] * 240
        val_json["pos"][l] = len_to_strings[l]

# double check total train_pos, train_neg, val_pos, val_neg
train_pos_count = sum([len(train_json["pos"][k]) for k in train_json["pos"]])
train_neg_count = sum([len(train_json["neg"][k]) for k in train_json["neg"]])
val_pos_count = sum([len(val_json["pos"][k]) for k in val_json["pos"]])
val_neg_count = sum([len(val_json["neg"][k]) for k in val_json["neg"]])
print(f"train_pos_count={train_pos_count}, train_neg_count={train_neg_count}, val_pos_count={val_pos_count}, val_neg_count={val_neg_count}")

with open("/data/yingshac/tmlr/shorter/validation/fix_d_1.json", 'w') as f:
    json.dump(val_json, f, indent=2)
with open("/data/yingshac/tmlr/shorter/train/fix_d_1.json", 'w') as f:
    json.dump(train_json, f, indent=2)


# sanity check
with open("/data/yingshac/tmlr/shorter/train/fix_d_1.json", 'r') as f:
    len_to_strings = json.load(f)
    pos_len_to_strings = len_to_strings["pos"]
    neg_len_to_strings = len_to_strings["neg"]
    l = random.choice([k for k in pos_len_to_strings.keys() if pos_len_to_strings[k]])
    print(len(set(pos_len_to_strings[l])), len(pos_len_to_strings[l]))
    print(f"Pos: l={l}, d={double_check(random.choice(pos_len_to_strings[l]))}")
    l = random.choice([k for k in neg_len_to_strings.keys() if neg_len_to_strings[k]])
    print(len(set(neg_len_to_strings[l])), len(neg_len_to_strings[l]))
    print(f"Neg: l={l}, d={double_check(random.choice(neg_len_to_strings[l]))}")
"""

### Generate train/fix_d_1.json, val/fix_d_1.json for l=3-40
# there are 836 unique negatives for l=3-40
#   write all, duplicating 10 times to /data/yingshac/tmlr/train/fix_d_1.json; sample 19 and write to /data/yingshac/tmlr/validation/fix_d_1.json
# then 19 unique positives for l=3-40, write all, duplicating 440 times to /data/yingshac/tmlr/train/fix_d_1.json
#   write the same 19 unique positives to /data/yingshac/tmlr/validation/fix_d_1.json
train_json = {"pos": {}, "neg": {}}
val_json = {"pos": {}, "neg": {}}
with open(os.path.join(data_dir, f"negatives.json"), 'r') as f:
    len_to_strings = json.load(f)
    # aggregate all negative strings
    all_negatives = []
    for l in len_to_strings:
        if int(l)<=40: all_negatives.extend(list(len_to_strings[l]))
sample_val_negatives = random.sample(all_negatives, 19)
for l in len_to_strings:
    if int(l)<=40: 
        train_json["neg"][l] = len_to_strings[l] * 10
        val_json["neg"][l] = [s for s in len_to_strings[l] if s in sample_val_negatives]

with open(os.path.join(data_dir, f"positives.json"), 'r') as f:
    len_to_strings = json.load(f)
for l in len_to_strings:
    if int(l)<=40: 
        train_json["pos"][l] = len_to_strings[l] * 440
        val_json["pos"][l] = len_to_strings[l]

# double check total train_pos, train_neg, val_pos, val_neg
train_pos_count = sum([len(train_json["pos"][k]) for k in train_json["pos"]])
train_neg_count = sum([len(train_json["neg"][k]) for k in train_json["neg"]])
val_pos_count = sum([len(val_json["pos"][k]) for k in val_json["pos"]])
val_neg_count = sum([len(val_json["neg"][k]) for k in val_json["neg"]])
print(f"train_pos_count={train_pos_count}, train_neg_count={train_neg_count}, val_pos_count={val_pos_count}, val_neg_count={val_neg_count}")

with open("/data/yingshac/tmlr/validation/fix_d_1.json", 'w') as f:
    json.dump(val_json, f, indent=2)
with open("/data/yingshac/tmlr/train/fix_d_1.json", 'w') as f:
    json.dump(train_json, f, indent=2)