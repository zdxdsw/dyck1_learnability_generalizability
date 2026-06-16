"""
This script 
(1) checks that all handles in "../output_<MODEL>/<task_suite>/<task>/<handle>"
    are present in "/data/yingshac/tmlr/ckpts_<MODEL>/<task_suite>/<task>/<handle>".

(2) if any handles are present in the ckpt directory but not in the output directory, 
    print them out and delete "rm -rf /data/yingshac/tmlr/ckpts_<MODEL>/<task_suite>/<task>/<handle>"

First, collect handles in the output directory and arrange them as a dict:
    <task> as first key, <task_suite_suffix> as second key.
    <task_suite> is something like "S1_d_[2-8]_4B", we extract "4B" as the suffix.
"""

MODEL = "transformer"
OUTPUT_DIR = f"/home/yingshac/workspace/tmlr/scripts/experiments/output_{MODEL}/"
CKPT_DIR = f"/data/yingshac/tmlr/ckpts_{MODEL}/"
TASK_AND_SUFFIX_TO_TASK_SUITE = {}

import os, shutil, json
from collections import defaultdict, Counter
from tqdm import tqdm, trange

### collect_handles
"""
Arrange handles in the output directory as a dict:
<task> as first key, <task_suite_suffix> as second key.
"""
print(f"Collecting handles in {OUTPUT_DIR}...")
output_handles = defaultdict(lambda: defaultdict(list))
for task_suite in os.listdir(OUTPUT_DIR):
    task_suite_path = os.path.join(OUTPUT_DIR, task_suite)
    task_suite_suffix = task_suite.split("_")[-1]
    for task in os.listdir(task_suite_path):
        TASK_AND_SUFFIX_TO_TASK_SUITE[(task, task_suite_suffix)] = task_suite
        task_path = os.path.join(task_suite_path, task)
        for handle in os.listdir(task_path):
            output_handles[task][task_suite_suffix].append(handle)

### Sanity check: All handles in the output directory must be present in the ckpt directory!
print("Sanity check: All handles in the output directory must be present in the ckpt directory!")
for task, suffix_to_handles in output_handles.items():
    for suffix, handles in suffix_to_handles.items():
        for handle in handles:
            task_suite = TASK_AND_SUFFIX_TO_TASK_SUITE.get((task, suffix), None)
            ckpt_handle_path = os.path.join(CKPT_DIR, task_suite, task, handle)
            if not os.path.exists(ckpt_handle_path):
                print(f"Warning: handle {handle} under task {task} and suffix {suffix} is not in the ckpt directory!")
print("Sanity check Done!\n")

### Delete handles in the ckpt directory that are not in the output directory
print("Deleting handles in the ckpt directory that are not in the output directory...")

for task_suite in os.listdir(CKPT_DIR):
    print(f"\nTask Suite {task_suite}:", end=" ")
    to_delete = []
    ckpt_task_suite_path = os.path.join(CKPT_DIR, task_suite)
    for task in os.listdir(ckpt_task_suite_path):
        ckpt_task_path = os.path.join(ckpt_task_suite_path, task)
        for handle in os.listdir(ckpt_task_path):
            if handle not in output_handles[task][task_suite.split("_")[-1]]:
                handle_path = os.path.join(ckpt_task_path, handle)
                to_delete.append(handle_path)
                print(f"Preview: rm -rf {handle_path}")
    if len(to_delete) > 0:
        print(f"Found {len(to_delete)} surplus handles.")
    else: print("pass", end="")
    ### BE CAREFUL!
    # for handle_path in to_delete: shutil.rmtree(handle_path)
print("\nDeleting surplus ckpt handles Done!")