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

import random, os, pytz, json
from tqdm import tqdm, trange
from datetime import datetime
timezone = pytz.timezone('America/New_York') 
from collections import Counter
from utils import *

pos_data_dir = "/data/yingshac/tmlr/shorter/positives"
neg_data_dir = "/data/yingshac/tmlr/shorter/negatives"
train_data_dir = "/data/yingshac/tmlr/shorter/train"
val_data_dir = "/data/yingshac/tmlr/shorter/validation"

"""

| d | #Neg  | #Pos |   Multiply #Pos  | #Pos+#Neg | Multiply #Total   | Cum. #Total |
|---| ----- | ---- | ---------------- | --------- | ----------------- | ----------- |
| 2 | 3924  | 486  | 8 * 486 = 3888   | 7812      | 1                 | 7812        |
| 3 | 11433 | 1789 | 7 * 1789 = 12523 | 23956     | 1                 | 31768       |
| 4 | 5726  | 1102 | 5 * 1102 = 5510  | 11236     | 2 * 11236 = 22472 | 54240       |
| 5 | 2283  | 530  | 4 * 530 = 2120   | 4403      | 6 * 4403 = 26418  | 80658       |
| 6 | 697   | 183  | 4 * 183 = 732    | 1429      | 20 * 1429 = 28580 | 109238      |

"""
Multiply_Pos = {2: 8, 3: 7, 4: 5, 5: 4, 6: 4}
Multiply_Total = {2: 1, 3: 1, 4: 2, 5: 6, 6: 20}

for d in range(2, 7):
    train_json = {"pos": {}, "neg": {}}
    val_json = {"pos": {}, "neg": {}}

    # Load all unique pos and neg samples for this d
    with open(os.path.join(pos_data_dir, f"fix_d_{d}.json"), 'r') as f:
        pos_len_to_strings = json.load(f)
    with open(os.path.join(neg_data_dir, f"fix_d_{d}.json"), 'r') as f:
        neg_len_to_strings = json.load(f)

    ### Train JSON construction:
    # put all neg samples into train_json
    for l in neg_len_to_strings:
        train_json["neg"][l] = neg_len_to_strings[l]
    # put all pos samples, duplicated by "Multiply #Pos" (see table) times, into train_json
    for l in pos_len_to_strings:
        train_json["pos"][l] = pos_len_to_strings[l] * Multiply_Pos[d]
    # multiple both pos and neg training samples by "Multiply #Total" (see table) times
    for l in train_json["pos"]:
        train_json["pos"][l] = train_json["pos"][l] * Multiply_Total[d]
    for l in train_json["neg"]:
        train_json["neg"][l] = train_json["neg"][l] * Multiply_Total[d]

    ### Validation JSON construction:
    # sample 100 pos and 100 neg from unique pos and neg samples for validation set
    # aggregate length
    all_unique_pos, all_unique_neg = [], []
    for l in pos_len_to_strings: all_unique_pos += pos_len_to_strings[l]
    for l in neg_len_to_strings: all_unique_neg += neg_len_to_strings[l]
    val_unique_pos = random.sample(all_unique_pos, min(100, len(all_unique_pos)))
    val_unique_neg = random.sample(all_unique_neg, min(100, len(all_unique_neg)))
    # put them into val_json
    for s in val_unique_pos:
        l = str(len(s))
        if l not in val_json["pos"]: val_json["pos"][l] = []
        val_json["pos"][l].append(s)
    for s in val_unique_neg:
        l = str(len(s))
        if l not in val_json["neg"]: val_json["neg"][l] = []
        val_json["neg"][l].append(s)

    ### save train_json and val_json
    with open(os.path.join(train_data_dir, f"fix_d_{d}.json"), 'w') as f:
        json.dump(train_json, f, indent=2)
    with open(os.path.join(val_data_dir, f"fix_d_{d}.json"), 'w') as f:
        json.dump(val_json, f, indent=2)
    print(f"\nfinish fix_d={d}")
    print(f"train samples={sum([len(train_json['pos'][k]) for k in train_json['pos']]) + sum([len(train_json['neg'][k]) for k in train_json['neg']])},", end="")
    print(f" (pos={sum([len(train_json['pos'][k]) for k in train_json['pos']])}, neg={sum([len(train_json['neg'][k]) for k in train_json['neg']])}),")
    print(f"val samples={sum([len(val_json['pos'][k]) for k in val_json['pos']]) + sum([len(val_json['neg'][k]) for k in val_json['neg']])},", end="")
    print(f" (pos={sum([len(val_json['pos'][k]) for k in val_json['pos']])}, neg={sum([len(val_json['neg'][k]) for k in val_json['neg']])})")
