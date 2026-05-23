import random, os, pytz, json
from tqdm import tqdm, trange
from datetime import datetime
timezone = pytz.timezone('America/New_York') 
from collections import Counter
from utils import *
from tabulate import tabulate

data_dir = "/data/yingshac/tmlr/shorter/positives"
neg_dir = "/data/yingshac/tmlr/shorter/negatives"
"""
    Only include negatives that have an "off-by-one" error: 
    strings that differ from a positive by one insertion/swap/deletion
"""

"""
# for each <fix_d, l>, suppose we have generated N positives
# do insert/delete/swap/move_last_close on all of its positives; then do grafting N times.
swap_trials, displace_trials = 5, 5
for d in range(2, 7):
    neg_len_to_strings = {l: set() for l in range(3, 21)}
    with open(os.path.join(data_dir, f"fix_d_{d}.json"), 'r') as f:
        pos_len_to_strings = json.load(f)
    for l in trange(4, 21, 2):
        for s in pos_len_to_strings[str(l)]:
            if l<20:
                neg_len_to_strings[l+1].add(insert_open(s))
                neg_len_to_strings[l+1].add(insert_close(s))
            neg_len_to_strings[l-1].add(delete_open(s))
            neg_len_to_strings[l-1].add(delete_close(s))

            for _ in range(swap_trials):
                candidate = swap(s)
                if double_check(candidate) == -1: neg_len_to_strings[l].add(candidate)
            for _ in range(displace_trials):
                candidate = displace_last_close(s)
                if double_check(candidate) == -1: neg_len_to_strings[l].add(candidate)
        if l >= 4*d+2:
            for _ in range(len(pos_len_to_strings[str(l)])):
                neg_len_to_strings[l].add(grafting(d, l, pos_len_to_strings))
    # double-check whether they are true negatives, then de-duplicate
    for l in neg_len_to_strings:
        neg_len_to_strings[l] = list(set([s for s in neg_len_to_strings[l] if double_check(s) == -1]))
    # sort neg_len_to_strings by l
    neg_len_to_strings = dict(sorted(neg_len_to_strings.items(), key=lambda x: x[0]))
    with open(os.path.join(neg_dir, f"fix_d_{d}.json"), 'w') as f:
        json.dump(neg_len_to_strings, f, indent=2)


# print, per fix_d, the number of unique strings of each length
# print two tables; one for even l, one for odd l 
# (since only negatives resulted from insertion/deletion can have odd length)
headers = ["fix_d", "l=3"] + [f"{l}" for l in range(5, 18, 2)] + ["l=19", "total"]
table = []
for d in range(2, 7):
    with open(os.path.join(neg_dir, f"fix_d_{d}.json"), 'r') as f:
        len_to_strings = json.load(f)
    row = [d] + [len(len_to_strings[str(l)]) for l in range(3, 20, 2)]
    row.append(sum(row[1:]))
    table.append(row)
print(tabulate(table, headers=headers, tablefmt="rounded_grid"))
headers = ["fix_d", "l=4"] + [f"{l}" for l in range(6, 19, 2)] + ["l=20", "total"]
table = []
for d in range(2, 7):
    with open(os.path.join(neg_dir, f"fix_d_{d}.json"), 'r') as f:
        len_to_strings = json.load(f)
    row = [d] + [len(len_to_strings[str(l)]) for l in range(4, 21, 2)]
    row.append(sum(row[1:]))
    table.append(row)
print(tabulate(table, headers=headers, tablefmt="rounded_grid"))
"""
headers = ["fix_d", "total (l=3-20)"]
table = []
for d in range(2, 7):
    with open(os.path.join(neg_dir, f"fix_d_{d}.json"), 'r') as f:
        len_to_strings = json.load(f)
    row = [d] + [sum([len(len_to_strings[str(l)]) for l in range(3, 21)])]
    table.append(row)
print(tabulate(table, headers=headers, tablefmt="rounded_grid"))


"""
# sanity check
with open(os.path.join(neg_dir, "fix_d_4.json"), 'r') as f:
    len_to_strings = json.load(f)
    l = random.choice([k for k in len_to_strings.keys() if len_to_strings[k]])
    print(len(set(len_to_strings[l])), len(len_to_strings[l]))
    print(f"l={l}, d={double_check(random.choice(len_to_strings[l]))}")
    # print(len_to_strings[l])
"""