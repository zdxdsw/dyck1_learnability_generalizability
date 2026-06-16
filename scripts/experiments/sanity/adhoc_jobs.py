"""
Need to launch compensatory jobs to make up for the deleted duplicated jobs in the sanity check. 
This script is used to create commands for these jobs.

Here's output of sanity check:

**The following new jobs will be launched to compensate for the deleted duplicated jobs:**

Task suite: S1_d_trulyleq[2-8]_4F, Task: S1_d_trulyleq5, Seed: 42
Task suite: S1_d_trulyleq[2-8]_4G, Task: S1_d_trulyleq5, Seed: 42
Task suite: S1_d_trulyleq[2-8]_2d, Task: S1_d_trulyleq5, Seed: 61
Task suite: S1_d_trulyleq[2-8]_4f, Task: S1_d_trulyleq5, Seed: 49
Task suite: S1_d_trulyleq[2-8]_4c, Task: S1_d_trulyleq2, Seed: 89
Task suite: S1_d_trulyleq[2-8]_2E, Task: S1_d_trulyleq2, Seed: 77
Task suite: S1_d_trulyleq[2-8]_2b, Task: S1_d_trulyleq2, Seed: 89
Task suite: S1_d[2-8]_4C, Task: S1_d_3, Seed: 42
Task suite: S1_d[2-8]_4A, Task: S1_d_3, Seed: 49

The idea is to go to <OUTPUT_DIR>/<task_suite>/<task>/", fetch the first <handle>
Then read the config from "<OUTPUT_DIR>/<task_suite>/<task>/<handle>/config.json"

change the seed to the new seed
create date (i.e. new_handle)
modify the output_dir and ckpt_dir if needed; makedirs as needed.
dump the modified config to a new directory "<OUTPUT_DIR>/<task_suite>/<task>/<new_handle>/config.json"

Then collect commands to launch trainer.py in the same way as in run.py:

  os.system("{} accelerate launch --main_process_port {} --num_processes 1 trainer.py --date {} --task {} --output_dir '{}' | tee {}".format(
    args.cuda,
    args.port,
    config.date,
    args.task,
    config.output_dir,
    os.path.join(config.output_dir, date, "terminal.txt"),
  ))
"""

import os, json, pytz
from datetime import datetime
timezone = pytz.timezone('America/New_York')


JOBS_NEEDED = """
Task suite: S1_d_trulyleq[2-8]_4G, Task: S1_d_trulyleq5, Seed: 42
Task suite: S1_d_trulyleq[2-8]_2d, Task: S1_d_trulyleq5, Seed: 61
Task suite: S1_d_trulyleq[2-8]_4f, Task: S1_d_trulyleq5, Seed: 49
Task suite: S1_d_trulyleq[2-8]_4c, Task: S1_d_trulyleq2, Seed: 89
Task suite: S1_d_trulyleq[2-8]_2E, Task: S1_d_trulyleq2, Seed: 77
Task suite: S1_d_trulyleq[2-8]_2b, Task: S1_d_trulyleq2, Seed: 89
Task suite: S1_d_[2-8]_4C, Task: S1_d_3, Seed: 42
Task suite: S1_d_[2-8]_4A, Task: S1_d_3, Seed: 49
"""

SUITE_TASK_SEED_LIST = []
for line in JOBS_NEEDED.strip().split("\n"):
    line = line.strip()
    if not line: continue
    parts = line.split(", ")
    task_suite = parts[0].split(": ")[1]
    task = parts[1].split(": ")[1]
    seed = int(parts[2].split(": ")[1])
    SUITE_TASK_SEED_LIST.append((task_suite, task, seed))
print(SUITE_TASK_SEED_LIST)


MODEL = "transformer"
OUTPUT_DIR = f"/home/yingshac/workspace/tmlr/scripts/experiments/output_{MODEL}/"
CKPT_DIR = f"/data/yingshac/tmlr/ckpts_{MODEL}/"

to_launch = []
port = 29500

for task_suite, task, seed in SUITE_TASK_SEED_LIST:
    ### fetch config from the first existing handle
    output_task_suite_path = os.path.join(OUTPUT_DIR, task_suite)
    output_task_path = os.path.join(output_task_suite_path, task)
    existing_handles = os.listdir(output_task_path)
    if not existing_handles:
        print(f"Error: no existing handle found under {output_task_path}!")
        continue
    existing_handle = existing_handles[0]
    existing_config_path = os.path.join(output_task_path, existing_handle, "config.json")
    with open(existing_config_path, "r") as f:
        config = json.load(f)
    ### modify config
    date = datetime.now(timezone).strftime("%m%d_%H%M%S")
    config["date"] = date
    config["seed"] = seed
    config["output_dir"] = os.path.join(OUTPUT_DIR, task_suite, task)
    config["ckpt_dir"] = os.path.join(CKPT_DIR, task_suite, task)
    ### makedirs
    os.makedirs(os.path.join(config["output_dir"], date), exist_ok=True)
    os.makedirs(os.path.join(config["ckpt_dir"], date), exist_ok=True)
    ### dump modified config
    with open(os.path.join(config["output_dir"], date, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    ### prepare command 
    cuda = f"CUDA_VISIBLE_DEVICES=\"{port%4}\""
    command = "{} accelerate launch --main_process_port {} --num_processes 1 trainer.py --date {} --task {} --output_dir '{}' | tee '{}'".format(
        cuda,
        port,
        config["date"],
        task,
        config["output_dir"],
        os.path.join(config["output_dir"], date, "terminal.txt"),
    )
    to_launch.append(command)
    port += 1

print(f"\nTotal: {len(to_launch)} Commands to launch:")
for cmd in to_launch:
    print(cmd)
    
print("\nDone!")
