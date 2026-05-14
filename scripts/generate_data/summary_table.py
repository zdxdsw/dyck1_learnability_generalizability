import sys, atexit

_original_stdout = sys.stdout
_original_stderr = sys.stderr

log_file = open("summary_table.txt", "w", buffering=1)

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

import os, json
from tabulate import tabulate

train_data_dir = "/data/yingshac/tmlr/train"
val_data_dir = "/data/yingshac/tmlr/validation"
test_data_dir = "/data/yingshac/tmlr/test"

""" json formal
    <*_data_dir>/fix_d_{d}.json = 
    {
        "pos": {
            "l": [s1, s2, ...],
            ...        },
        "neg": {
            "l": [s1, s2, ...],
            ...        }
    }

"""

headers = [
    "\n\nDepth",
    "Train\n-----\nlength", # span of lengths f"{_min}-{_max}"
    "\n\npos",
    "\n\nneg",
    "\n\ntotal",
    "Val\n-----\nlength",
    "\n\npos",
    "\n\nneg",
    "\n\ntotal",
    "Testing\n-----\nlength",
    "\n\npos",
    "\n\nneg",
    "\n\ntotal"
]

table = []
# write table rows according to header
for d in range(2, 9):
    with open(os.path.join(train_data_dir, f"fix_d_{d}.json"), 'r') as f:
        train_json = json.load(f)
    with open(os.path.join(val_data_dir, f"fix_d_{d}.json"), 'r') as f:
        val_json = json.load(f)
    with open(os.path.join(test_data_dir, f"fix_d_{d}.json"), 'r') as f:
        test_json = json.load(f)
    train_pos_counts = {l: len(train_json["pos"][str(l)]) for l in train_json["pos"]}
    train_neg_counts = {l: len(train_json["neg"][str(l)]) for l in train_json["neg"]}
    val_pos_counts = {l: len(val_json["pos"][str(l)]) for l in val_json["pos"]}
    val_neg_counts = {l: len(val_json["neg"][str(l)]) for l in val_json["neg"]}
    test_pos_counts = {l: len(test_json["pos"][str(l)]) for l in test_json["pos"]}
    test_neg_counts = {l: len(test_json["neg"][str(l)]) for l in test_json["neg"]}
    train_total_counts = {l: train_pos_counts.get(l, 0) + train_neg_counts.get(l, 0) for l in set(train_pos_counts.keys()) | set(train_neg_counts.keys())}
    val_total_counts = {l: val_pos_counts.get(l, 0) + val_neg_counts.get(l, 0) for l in set(val_pos_counts.keys()) | set(val_neg_counts.keys())}
    test_total_counts = {l: test_pos_counts.get(l, 0) + test_neg_counts.get(l, 0) for l in set(test_pos_counts.keys()) | set(test_neg_counts.keys())}
    train_length_span = f"{min([int(l) for l in train_total_counts.keys()])}-{max([int(l) for l in train_total_counts.keys()])}"
    val_length_span = f"{min([int(l) for l in val_total_counts.keys()])}-{max([int(l) for l in val_total_counts.keys()])}"
    test_length_span = f"{min([int(l) for l in test_total_counts.keys()])}-{max([int(l) for l in test_total_counts.keys()])}"
    row = [
        d,
        train_length_span,
        sum(train_pos_counts.values()),
        sum(train_neg_counts.values()),
        sum(train_total_counts.values()),
        val_length_span,
        sum(val_pos_counts.values()),
        sum(val_neg_counts.values()),
        sum(val_total_counts.values()),
        test_length_span,
        sum(test_pos_counts.values()),
        sum(test_neg_counts.values()),
        sum(test_total_counts.values())
    ]
    table.append(row)
print(tabulate(table, headers=headers, tablefmt="rounded_grid", stralign="center"))