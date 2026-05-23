import os, json
from utils import double_check
from tqdm import trange, tqdm
from collections import Counter

def check_unique_examples(lines):
    Spos, Sneg = set(), set()
    #breakdown non-unique pos samples by depth
    C = Counter()
    #breakdown labels
    L = Counter()
    for line in lines:
        x, label = json.loads(line.strip())
        d = double_check("".join(x))
        C[d] += 1
        L[label[-1]] += 1
        if label[-1] == '1':
            assert d > 0
            Spos.add("".join(x))
        else:
            assert d == -1
            Sneg.add("".join(x))
    print("Breakdown of labels (non-unique input): ", dict(L))
    return Spos, Sneg, C

def pos_samples_per_depth(Spos):
    # breakdown unique positive samples by depth
    C = Counter()
    for s in Spos:
        d = double_check(s)
        C[d] += 1
    return C

with open("/data/yingshac/model_succ/data/dyck1_2/0327_213426/train.txt", "r") as f:
    lines = f.readlines()
with open("/home/yingshac/workspace/model_succ/data/dyck1_2/0327_213426/val.txt", "r") as f:
    lines += f.readlines()
print(f"\nd=2\nTotal samples in train.txt: {len(lines)}")
Spos, Sneg, C = check_unique_examples(lines)
print(f"#unique total: {len(Spos) + len(Sneg)}, #unique positives in train.txt: {len(Spos)}, #unique negatives in train.txt: {len(Sneg)}")
print(f"#non-unique positives per depth: {dict(C)}")
C = pos_samples_per_depth(Spos)
print(f"#unique positives per depth: {dict(C)}")

with open("/data/yingshac/model_succ/data/dyck1_3/0327_221731/train.txt", "r") as f:
    lines = f.readlines()
with open("/home/yingshac/workspace/model_succ/data/dyck1_3/0327_221731/val.txt", "r") as f:
    lines += f.readlines()
print(f"\nd=3\nTotal samples in train.txt: {len(lines)}")
Spos, Sneg, C = check_unique_examples(lines)
print(f"#unique total: {len(Spos) + len(Sneg)}, #unique positives in train.txt: {len(Spos)}, #unique negatives in train.txt: {len(Sneg)}")
print(f"#non-unique positives per depth: {dict(C)}")
C = pos_samples_per_depth(Spos)
print(f"#unique positives per depth: {dict(C)}")

with open("/data/yingshac/model_succ/data/dyck1_4/0327_214440/train.txt", "r") as f:
    lines = f.readlines()
with open("/home/yingshac/workspace/model_succ/data/dyck1_4/0327_214440/val.txt", "r") as f:
    lines += f.readlines()
print(f"\nd=4\nTotal samples in train.txt: {len(lines)}")
Spos, Sneg, C = check_unique_examples(lines)
print(f"#unique total: {len(Spos) + len(Sneg)}, #unique positives in train.txt: {len(Spos)}, #unique negatives in train.txt: {len(Sneg)}")
print(f"#non-unique positives per depth: {dict(C)}")
C = pos_samples_per_depth(Spos)
print(f"#unique positives per depth: {dict(C)}")