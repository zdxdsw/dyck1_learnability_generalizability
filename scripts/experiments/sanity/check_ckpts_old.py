"""
This script 
(1) checks that all handles in "../output_<MODEL>/<task_suite>/<task>/<handle>"
    are present in "/data/yingshac/tmlr/ckpts_<MODEL>/<task>/<handle>".
    Note that the ckpt directory is flattened over task_suite. 
    This wasn't intentional but it’s not worth fixing at this point.

(2) if any handles are present in the ckpt directory but not in the output directory, 
    print them out and delete "rm -rf /data/yingshac/tmlr/ckpts_<MODEL>/<task>/<handle>"

First, collect handles in the output directory and arrange them as a dict:
    <task> as first key, <task_suite_suffix> as second key.
    Sanity check: All handles under the same <task> must be distinct!

    <task_suite> is something like "S1_d_[2-8]_4B", we extract "4B" as the suffix.
"""

MODEL = "rnn"
OUTPUT_DIR = f"/home/yingshac/workspace/tmlr/scripts/experiments/output_{MODEL}/"
CKPT_DIR = f"/data/yingshac/tmlr/ckpts_{MODEL}/"
# CKPT_DIR = f"/data/yingshac/tmlr/ckpts/"
# NEW_CKPT_DIR = f"/data/yingshac/tmlr/ckpts_{MODEL}/"
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

avail_task_suites = [
    "S1_d_[2-8]_A", "S1_d_trulyleq[2-8]_A", "S1_d_trulyleq[2-8]_C", 
    "S1_d_trulyleq[2-8]_D", "S1_d_trulyleq[2-8]_E", "S1_d_trulyleq[2-8]_F",
    "S3_d_[2-8]_A", "S3_d_trulyleq[2-8]_A" 
] # for RNN, only process task suites that were completed last month.

for task_suite in os.listdir(OUTPUT_DIR):
    if task_suite not in avail_task_suites: continue
    task_suite_path = os.path.join(OUTPUT_DIR, task_suite)
    task_suite_suffix = task_suite.split("_")[-1]
    for task in os.listdir(task_suite_path):
        TASK_AND_SUFFIX_TO_TASK_SUITE[(task, task_suite_suffix)] = task_suite
        task_path = os.path.join(task_suite_path, task)
        for handle in os.listdir(task_path):
            output_handles[task][task_suite_suffix].append(handle)

"""
### Sanity check1: All handles under the same <task> must be distinct!
print("Sanity check 1: All handles under the same <task> must be distinct!")
to_delete = []
makeup_jobs = [] # [(<task_suite>, <task>, <seed>)]
for task, suffix_to_handles in output_handles.items():
    handles = []
    for suffix, h in suffix_to_handles.items():
        handles.extend(h)
    handle_counter = Counter(handles)
    for handle, count in handle_counter.items():
        if count > 1: # print warning
            print(f"Warning: handle {handle} appears {count} times under task {task} in the output directory!")
            # delete duplicated handles from both output and ckpt directories. Will launch new job to compensate.
            for suffix, h in suffix_to_handles.items():
                if handle in h:
                    task_suite = TASK_AND_SUFFIX_TO_TASK_SUITE.get((task, suffix), None)
                    delete_output_path = os.path.join(OUTPUT_DIR, task_suite, task, handle)
                    # get seed from "<delete_output_path>/config.json"
                    with open(os.path.join(delete_output_path, "config.json"), "r") as f:
                        config = json.load(f)
                    seed = config.get("seed", None)
                    makeup_jobs.append((task_suite, task, seed))
                    to_delete.append(delete_output_path)
                    print(f"Preview: rm -rf {delete_output_path}")
            delete_ckpt_path = os.path.join(CKPT_DIR, task, handle)
            to_delete.append(delete_ckpt_path)
            print(f"Preview: rm -rf {delete_ckpt_path}")

if makeup_jobs:
    print("\nThe following new jobs will be launched to compensate for the deleted duplicated jobs:")
    for task_suite, task, seed in makeup_jobs:
        print(f"Task suite: {task_suite}, Task: {task}, Seed: {seed}")

### BE CAREFUL!
# for handle_path in to_delete: shutil.rmtree(handle_path)
print("Sanity check 1 Done!\n")

"""
### Sanity check2: All handles in the output directory must be present in the ckpt directory!
print("Sanity check 2: All handles in the output directory must be present in the ckpt directory!")
for task, suffix_to_handles in output_handles.items():
    for suffix, handles in suffix_to_handles.items():
        for handle in handles:
            ckpt_handle_path = os.path.join(CKPT_DIR, task, handle)
            if not os.path.exists(ckpt_handle_path):
                print(f"Warning: handle {handle} under task {task} and suffix {suffix} is not in the ckpt directory!")
print("Sanity check 2 Done!\n")

### Delete handles in the ckpt directory that are not in the output directory
print("Deleting handles in the ckpt directory that are not in the output directory...")

all_task_unique_output_handles = 0
all_task_unique_ckpt_handles = 0
for task in os.listdir(CKPT_DIR):
    if len(task.split("_")) > 3: continue # skip already refactored directories with structure "<CKPT_DIR>/<task_suite>/<task>/<handle>"
    print(f"\nTask {task}:", end=" ")
    to_delete = []
    list_output_handles, list_ckpt_handles = [], []

    task_path = os.path.join(CKPT_DIR, task)
    list_output_handles_for_cur_task = []
    for suffix, handles in output_handles[task].items():
        list_output_handles_for_cur_task.extend(handles)
    list_output_handles.extend(list_output_handles_for_cur_task)
    for handle in os.listdir(task_path):
        list_ckpt_handles.append(handle)
        handle_path = os.path.join(task_path, handle)        
        if handle not in list_output_handles_for_cur_task:
            to_delete.append(handle_path)
            print(f"Preview: rm -rf {handle_path}")
    
    all_task_unique_output_handles += len(set(list_output_handles))
    all_task_unique_ckpt_handles += len(set(list_ckpt_handles))
    if len(to_delete) > 0 or len(list_ckpt_handles) != len(set(list_ckpt_handles)) or len(list_output_handles) != len(set(list_output_handles)):
        print(f"\n|list_output_handles| = {len(list_output_handles)}")
        print(f"|list_ckpt_handles| = {len(list_ckpt_handles)}")
        print(f"|unique_output_handles| = {len(set(list_output_handles))}")
        print(f"|unique_ckpt_handles| = {len(set(list_ckpt_handles))}")
        print(f"|unique_ckpt_handles exclude unique_output_handles| = {len(to_delete)} = #handles to delete.")
    else: print("pass")
    
    ### BE CAREFUL!
    # for handle_path in to_delete: shutil.rmtree(handle_path)

print(f"\nTotal unique handles in output directory across all tasks: {all_task_unique_output_handles}")
print(f"Total unique handles in ckpt directory across all tasks: {all_task_unique_ckpt_handles}")

"""

### Refactor ckpt directory
print("\nRefactoring ckpt directory to have the same structure as the output directory...")
# if sanity check 1 & 2 are passed, 
# then each "<CKPT_DIR>/<task>/<handle>" must correspond to a unique "<OUTPUT_DIR>/<task_suite>/<task>/<handle>"
# makedir "<CKPT_DIR>/<task_suite>/<task>/<handle>", move "<CKPT_DIR>/<task>/<handle>" to "<CKPT_DIR>/<task_suite>/<task>/<handle>", and delete empty "<CKPT_DIR>/<task>"
move = 0
for task in os.listdir(CKPT_DIR):
    if len(task.split("_")) > 3: continue # skip already refactored directories with structure "<CKPT_DIR>/<task_suite>/<task>/<handle>"
    ckpt_task_path = os.path.join(CKPT_DIR, task)
    for handle in os.listdir(ckpt_task_path):
        ckpt_handle_path = os.path.join(ckpt_task_path, handle)
        # find corresponding task_suite from output directory
        corresponding_task_suite = None
        for suffix, handles in output_handles[task].items():
            if handle in handles:
                corresponding_task_suite = TASK_AND_SUFFIX_TO_TASK_SUITE.get((task, suffix), None)
                break
        if not corresponding_task_suite in avail_task_suites: continue
        if corresponding_task_suite is None:
            print(f"Error: cannot find corresponding task suite for task {task} and handle {handle}!")
            continue
        new_ckpt_handle_path = os.path.join(NEW_CKPT_DIR, corresponding_task_suite, task, handle)
        # os.makedirs(os.path.dirname(new_ckpt_handle_path), exist_ok=True)
        # print(f"Preview: mv {ckpt_handle_path} {new_ckpt_handle_path}")
        move += 1
        ### BE CAREFUL!
        # shutil.move(ckpt_handle_path, new_ckpt_handle_path)
    
    if len(os.listdir(ckpt_task_path)) == 0: # delete empty task directory
        print(f"Preview: rm -rf {ckpt_task_path}")
        
        ### BE CAREFUL!
        shutil.rmtree(ckpt_task_path)
print(f"Will move {move} handles to the new structure.")
print("Refactoring done!")
"""