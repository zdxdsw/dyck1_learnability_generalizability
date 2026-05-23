"""import sys, atexit

_original_stdout = sys.stdout
_original_stderr = sys.stderr

log_file = open("terminal_sample_lengthy_test.txt", "w", buffering=1)

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
from utils import double_check

""" Sample lengthy_test:

lengthy_pos_len_to_strings: merge from
    "/data/yingshac/tmlr/lengthy_positives" + "/data/yingshac/tmlr/positives"
    for l=21-60

lengthy_neg_len_to_strings: merge from
    "/data/yingshac/tmlr/lengthy_negatives" + "/data/yingshac/tmlr/negatives"
    for l=21-60

Sample 200 positives per l from lengthy_pos_len_to_strings.
Sample 50 negatives per odd l and 150 negatives per even l from lengthy_neg_len_to_strings

End up: 4k pos & 4k neg for lengthy_test.
"""

test_data_dir = "/data/yingshac/tmlr/shorter/test"
pos_data_dir1 = "/data/yingshac/tmlr/lengthy_positives"
pos_data_dir2 = "/data/yingshac/tmlr/positives"
neg_data_dir1 = "/data/yingshac/tmlr/lengthy_negatives"
neg_data_dir2 = "/data/yingshac/tmlr/negatives"

"""
for d in trange(2, 7):
    lengthy_pos_len_to_strings, lengthy_neg_len_to_strings = {}, {}
    test_json = {"pos": {}, "neg": {}}

    # Load the universe of pos and neg lengthy samples for this d (l=21-60)
    with open(os.path.join(pos_data_dir1, f"fix_d_{d}.json"), 'r') as f:
        pos_len_to_strings = json.load(f)
        for l in pos_len_to_strings:
            if 21 <= int(l) <= 60:
                lengthy_pos_len_to_strings[l] = pos_len_to_strings[l]
    with open(os.path.join(pos_data_dir2, f"fix_d_{d}.json"), 'r') as f:
        pos_len_to_strings = json.load(f)
        for l in pos_len_to_strings:
            if 21 <= int(l) <= 60:
                if l not in lengthy_pos_len_to_strings: lengthy_pos_len_to_strings[l] = []
                lengthy_pos_len_to_strings[l] += pos_len_to_strings[l]
    with open(os.path.join(neg_data_dir1, f"fix_d_{d}.json"), 'r') as f:
        neg_len_to_strings = json.load(f)
        for l in neg_len_to_strings:
            if 21 <= int(l) <= 60:
                lengthy_neg_len_to_strings[l] = neg_len_to_strings[l]
    with open(os.path.join(neg_data_dir2, f"fix_d_{d}.json"), 'r') as f:
        neg_len_to_strings = json.load(f)
        for l in neg_len_to_strings:
            if 21 <= int(l) <= 60:
                if l not in lengthy_neg_len_to_strings: lengthy_neg_len_to_strings[l] = []
                lengthy_neg_len_to_strings[l] += neg_len_to_strings[l]

    for l in range(21, 61):
        neg_samples = lengthy_neg_len_to_strings[str(l)]
        if l % 2 == 1: # odd length: only 50 negatives
            test_json["neg"][str(l)] = random.sample(neg_samples, 50)
        
        else: # even length: 150 negatives
            test_json["neg"][str(l)] = random.sample(neg_samples, 150)

            pos_samples = lengthy_pos_len_to_strings[str(l)]
            test_json["pos"][str(l)] = random.sample(pos_samples, 200)
    
    # check #unique pos and neg samples for this d
    num_unique_pos = sum(len(set(test_json["pos"][l])) for l in test_json["pos"])
    num_unique_neg = sum(len(set(test_json["neg"][l])) for l in test_json["neg"])
    print(f"d={d}: Unique pos samples: {num_unique_pos}, Unique neg samples: {num_unique_neg}")

    with open(os.path.join(test_data_dir, f"fix_d_{d}.json"), 'w') as f:
        json.dump(test_json, f, indent=2)
"""

### sanity check
with open(os.path.join(test_data_dir, "fix_d_3.json"), 'r') as f:
    len_to_strings = json.load(f)
    pos_len_to_strings = len_to_strings["pos"]
    neg_len_to_strings = len_to_strings["neg"]
    l = random.choice([k for k in pos_len_to_strings.keys() if pos_len_to_strings[k]])
    print(len(set(pos_len_to_strings[l])), len(pos_len_to_strings[l]))
    print(f"Pos: l={l}, d={double_check(random.choice(pos_len_to_strings[l]))}")
    l = random.choice([k for k in neg_len_to_strings.keys() if neg_len_to_strings[k]])
    print(len(set(neg_len_to_strings[l])), len(neg_len_to_strings[l]))
    print(f"Neg: l={l}, d={double_check(random.choice(neg_len_to_strings[l]))}")