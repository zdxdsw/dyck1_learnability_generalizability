import random, os, pytz, json
from tqdm import tqdm, trange
from datetime import datetime
timezone = pytz.timezone('America/New_York') 
from collections import Counter
from utils import generate, double_check

data_dir = "/data/yingshac/tmlr/shorter/positives"
"""
# open 7 file handles for fix_d 2-6
file_handles = {d: open(os.path.join(data_dir, f"fix_d_{d}.txt"), 'a') for d in range(2, 7)}

# ----- these numbers are hand-tuned to get a good distribution of unique samples per length for each fix_d -----
# number of samples to generate per fix_d (probaly more than needed)
sizes =  [50000, 5000, 100000, 500000, 500000]
transitions = [[0.05, 0.2, 0.75]] + [[0.01, 0.3, 0.69]] + [[0.05, 0.55, 0.4]] * 3
# ---------------------------------------------------------------------------------------------------------------

for d in range(6, 1, -1):
    for _ in tqdm(range(sizes[d-2]), desc=f"Generating {sizes[d-2]} data for fix_d={d}"):
        s = generate(max_depth=d, depth=0, max_length=20, transition=transitions[d-2])
        cur_d = double_check(s)
        if 2 <= cur_d <=6: file_handles[cur_d].write(s + "\n")
# close all file handles
for fh in file_handles.values(): fh.close()


# rewrite each txt file into json file with format: {"l": 20, "strings": [s1, s2, ...]}
for d in trange(2, 7):
    go_through_files = [f"fix_d_{x}.txt" for x in range(d, 7)]
    len_to_strings = {l: [] for l in range(4, 21, 2)}
    for file in go_through_files:
        with open(os.path.join(data_dir, file), 'r') as f:
            for line in f:
                s = line.strip()
                cur_d = double_check(s)
                l = len(s)
                if 4 <= l <= 20 and cur_d == d:
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
headers = ["fix_d", "l=4"] + [f"{l}" for l in range(6, 21, 2)] + ["l=20", "total"]
table = []
for d in range(2, 7):
    with open(os.path.join(data_dir, f"fix_d_{d}.json"), 'r') as f:
        len_to_strings = json.load(f)
    row = [d] + [len(len_to_strings[str(l)]) for l in range(4, 21, 2)]
    row.append(sum(row[1:]))
    table.append(row)
print(tabulate(table, headers=headers, tablefmt="rounded_grid"))
"""

# sanity check
with open(os.path.join(data_dir, "fix_d_3.json"), 'r') as f:
    len_to_strings = json.load(f)
    l = random.choice([k for k in len_to_strings.keys() if len_to_strings[k]])
    print(len(set(len_to_strings[l])), len(len_to_strings[l]))
    print(f"l={l}, d={double_check(random.choice(len_to_strings[l]))}")
    # print(len_to_strings[l])
