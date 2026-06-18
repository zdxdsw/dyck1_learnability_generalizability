"""
import sys, atexit

_original_stdout = sys.stdout
_original_stderr = sys.stderr

log_file = open("terminal_generate_lengthy_positives.txt", "w", buffering=1)

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

data_dir = "/data/yingshac/tmlr/deeper_tmp/lengthy_positives"

"""
# open 7 file handles for fix_d 9-16
file_handles = {d: open(os.path.join(data_dir, f"fix_d_{d}.txt"), 'a') for d in range(9, 17)}

# ----- these numbers are hand-tuned to get a good distribution of unique samples per length for each fix_d -----
# number of samples to generate per fix_d (probaly more than needed)

sizes =  [0, 0, 0, 0, 0, 100000, 100000, 100000]
transitions = [[0.1, 0.75, 0.15]] * 8

# sizes =  [300000, 700000, 80000, 30000, 10000, 90000, 500000]
# transitions = [[0.08, 0.05, 0.87]] * 2 + [[0.05, 0.3, 0.65]] *2 + [[0.07, 0.43, 0.5]] * 3

# ---------------------------------------------------------------------------------------------------------------

print()
for d in range(16, 8, -1):
    for _ in tqdm(range(sizes[d-9]), file=_original_stdout, desc=f"Generating {sizes[d-9]} data for fix_d={d}"):
        s = generate(max_depth=d, depth=0, max_length=120, transition=transitions[d-9])
        cur_d = double_check(s)
        if (cur_d != 15 and cur_d != 14) or 44 < len(s): continue
        if 9 <= cur_d <= 16 and 42 <= len(s) <= 120: file_handles[cur_d].write(s + "\n")
# close all file handles
for fh in file_handles.values(): fh.close()


# rewrite each txt file into json file with format: {"l": 42, "strings": [s1, s2, ...]}
for d in trange(9, 17):
    go_through_files = [f"fix_d_{x}.txt" for x in range(d, 17)]
    len_to_strings = {l: [] for l in range(42, 121, 2)}
    for file in go_through_files:
        with open(os.path.join(data_dir, file), 'r') as f:
            for line in f:
                s = line.strip()
                cur_d = double_check(s)
                l = len(s)
                if 42 <= l <= 120 and cur_d == d:
                    len_to_strings[l].append(s)
    # de-duplicate each list in len_to_strings
    for l in len_to_strings:
        len_to_strings[l] = list(set(len_to_strings[l]))
    # sort len_to_strings by l
    len_to_strings = dict(sorted(len_to_strings.items(), key=lambda x: x[0]))
    with open(os.path.join(data_dir, f"fix_d_{d}.json"), 'w') as f:
        json.dump(len_to_strings, f, indent=2)


# print, per fix_d, the number of unique strings of each length
from tabulate import tabulate
# First half
headers1 = ["fix_d", "l=42"] + [str(l) for l in range(44, 84, 2)] + ["l=80"]
table1 = []

# Second half
headers2 = ["fix_d", "l=84"] + [str(l) for l in range(86, 120, 2)] + ["l=120", "total"]
table2 = []

for d in range(9, 17):
    with open(os.path.join(data_dir, f"fix_d_{d}.json"), "r") as f:
        len_to_strings = json.load(f)

    row1 = [d] + [
        len(len_to_strings[str(l)])
        for l in range(42, 84, 2)
    ]

    row2 = [d] + [
        len(len_to_strings[str(l)])
        for l in range(84, 121, 2)
    ]

    total = sum(row1[1:] + row2[1:])
    row2.append(total)

    table1.append(row1)
    table2.append(row2)

print("\nl = 42–82\n")
print(tabulate(table1, headers=headers1, tablefmt="rounded_grid"))

print("\nl = 84–120\n")
print(tabulate(table2, headers=headers2, tablefmt="rounded_grid"))

# check if every table cell is >= 1000
pass_check = True
for row in table1 + table2:
    for cell in row[1:]:
        if cell < 1000:
            pass_check = False
print(f"\nEvery cell >= 1000: {pass_check}")
"""


# sanity check
with open(os.path.join(data_dir, "fix_d_15.json"), 'r') as f:
    len_to_strings = json.load(f)
    l = random.choice([k for k in len_to_strings.keys() if len_to_strings[k]])
    print(len(set(len_to_strings[l])), len(len_to_strings[l]))
    print(f"l={l}, d={double_check(random.choice(len_to_strings[l]))}")
    # print(len_to_strings[l])

