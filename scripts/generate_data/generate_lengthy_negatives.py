"""
import sys, atexit

_original_stdout = sys.stdout
_original_stderr = sys.stderr

log_file = open("terminal_generate_lengthy_negatives.txt", "w", buffering=1)

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
import random, os, pytz, json
from tqdm import tqdm, trange
from datetime import datetime
timezone = pytz.timezone('America/New_York') 
from collections import Counter
from utils import generate, double_check
from utils import insert_open, insert_close, delete_open, delete_close, swap, displace_last_close, grafting

data_dir = "/data/yingshac/tmlr/lengthy_positives"
neg_dir = "/data/yingshac/tmlr/lengthy_negatives"

"""
    Only include negatives that have an "off-by-one" error: 
    strings that differ from a positive by one insertion/swap/deletion
"""

"""
# for each <fix_d, l>, suppose we have generated N positives
# do insert/delete/swap/move_last_close on all of its positives; then do grafting N times.
swap_trials, displace_trials = 5, 5
for d in range(2, 9):
    neg_len_to_strings = {l: set() for l in range(41, 121)}
    with open(os.path.join(data_dir, f"fix_d_{d}.json"), 'r') as f:
        pos_len_to_strings = json.load(f)
    # for grafting, can also graft with shorter positives from /data/yingshac/tmlr/positives
    with open(os.path.join("/data/yingshac/tmlr/positives", f"fix_d_{d}.json"), 'r') as f:
        shorter_pos_len_to_strings = json.load(f)
        # merge into grafting_pos_len_to_strings (they have non-overlapping lengths, so no conflict)
        grafting_pos_len_to_strings = {**shorter_pos_len_to_strings, **pos_len_to_strings}
    for l in trange(42, 121, 2, file=_original_stdout):
        for s in pos_len_to_strings[str(l)]:
            if l<120:
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
            neg_len_to_strings[l].add(grafting(d, l, grafting_pos_len_to_strings))
    # double-check whether they are true negatives, then de-duplicate
    for l in neg_len_to_strings:
        neg_len_to_strings[l] = list(set([s for s in neg_len_to_strings[l] if double_check(s) == -1]))
    # sort neg_len_to_strings by l
    neg_len_to_strings = dict(sorted(neg_len_to_strings.items(), key=lambda x: x[0]))
    with open(os.path.join(neg_dir, f"fix_d_{d}.json"), 'w') as f:
        json.dump(neg_len_to_strings, f, indent=2)


# print, per fix_d, the number of unique strings of each length
from tabulate import tabulate
# print two tables; one for even l, one for odd l 
# (since only negatives resulted from insertion/deletion can have odd length)
headers = ["fix_d", "l=41"] + [f"{l}" for l in range(43, 118, 2)] + ["l=119", "total"]
table = []
for d in range(2, 9):
    with open(os.path.join(neg_dir, f"fix_d_{d}.json"), 'r') as f:
        len_to_strings = json.load(f)
    row = [d] + [len(len_to_strings[str(l)]) for l in range(41, 120, 2)]
    row.append(sum(row[1:]))
    # divide each cell by 1000, then round to the flooring integer
    row = [row[0]] + [int(round(cell/1000)) for cell in row[1:]]
    table.append(row)
print(tabulate(table, headers=headers, tablefmt="rounded_grid"))
headers = ["fix_d", "l=42"] + [f"{l}" for l in range(44, 120, 2)] + ["l=120", "total"]
table = []
for d in range(2, 9):
    with open(os.path.join(neg_dir, f"fix_d_{d}.json"), 'r') as f:
        len_to_strings = json.load(f)
    row = [d] + [len(len_to_strings[str(l)]) for l in range(42, 121, 2)]
    row.append(sum(row[1:]))
    # divide each cell by 1000, then round to the flooring integer
    row = [row[0]] + [int(round(cell/1000)) for cell in row[1:]]
    table.append(row)
print(tabulate(table, headers=headers, tablefmt="rounded_grid"))
"""

# sanity check
with open(os.path.join(neg_dir, "fix_d_3.json"), 'r') as f:
    len_to_strings = json.load(f)
    l = random.choice([k for k in len_to_strings.keys() if len_to_strings[k]])
    print(len(set(len_to_strings[l])), len(len_to_strings[l]))
    print(f"l={l}, d={double_check(random.choice(len_to_strings[l]))}")
    # print(len_to_strings[l])