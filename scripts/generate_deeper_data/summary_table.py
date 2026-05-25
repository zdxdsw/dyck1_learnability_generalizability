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

val_data_dir = "/data/yingshac/tmlr/deeper_tmp/validation"

""" json format
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
    "Val\n-----\nlength",
    "\n\npos",
    "\n\nneg",
    "\n\ntotal",
]

table = []
# write table rows according to header
for d in range(9, 17):
    with open(os.path.join(val_data_dir, f"fix_d_{d}.json"), 'r') as f:
        val_json = json.load(f)
    
    val_pos_counts = {l: len(val_json["pos"][str(l)]) for l in val_json["pos"]}
    val_neg_counts = {l: len(val_json["neg"][str(l)]) for l in val_json["neg"]}
    val_total_counts = {l: val_pos_counts.get(l, 0) + val_neg_counts.get(l, 0) for l in set(val_pos_counts.keys()) | set(val_neg_counts.keys())}
    val_length_span = f"{min([int(l) for l in val_total_counts.keys()])}-{max([int(l) for l in val_total_counts.keys()])}"
    
    row = [
        d,
        val_length_span,
        sum(val_pos_counts.values()),
        sum(val_neg_counts.values()),
        sum(val_total_counts.values()),
    ]
    table.append(row)
print(tabulate(table, headers=headers, tablefmt="rounded_grid", stralign="center"))