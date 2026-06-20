import os, json, re

JOBS_NEEDED = """
Job Unfinished: S1_d_trulyleq[2-8]_2P/S1_d_trulyleq8/0619_203905                                           
Job Unfinished: S1_d_trulyleq[2-8]_2P/S1_d_trulyleq3/0619_172018
"""

MODEL = "transformer"
OUTPUT_DIR = f"/home/yingshac/workspace/tmlr/scripts/experiments/output_{MODEL}/"
CKPT_DIR = f"/data/yingshac/tmlr/ckpts_{MODEL}/"

"""
From each line in JOBS_NEEDED:
    extract <task_suite>/<task>/<handle> using regex,
    pattern = ".... S{\d}_d_.../S{\d}_d_.../\d{4}_\d{6} ..."
    task_suite, task, handle = result.split("/")
"""

SUITE_TASK_HANDLE_LIST = []
pattern = r"S\d_d_[^/]+/S\d_d_[^/]+/\d{4}_\d{6}"
for line in JOBS_NEEDED.strip().split("\n"):
    line = line.strip()
    if not line: continue
    match = re.search(pattern, line)
    if match:
        result = match.group()
        task_suite, task, handle = result.split("/")
        SUITE_TASK_HANDLE_LIST.append((task_suite, task, handle))
    else:
        print(f"Error: no match found in line: {line}")
print(SUITE_TASK_HANDLE_LIST)

to_launch = []
port = 29500

for task_suite, task, handle in SUITE_TASK_HANDLE_LIST:
    ### prepare command 
    output_dir = os.path.join(OUTPUT_DIR, task_suite, task)
    cuda = f"CUDA_VISIBLE_DEVICES=\"{port%4}\""
    command = "{} accelerate launch --main_process_port {} --num_processes 1 trainer.py --date {} --task {} --output_dir '{}' 2> '{}' | tee '{}'  >/dev/null &".format(
        cuda,
        port,
        handle,
        task,
        output_dir,
        os.path.join(output_dir, handle, "stderr.txt"),
        os.path.join(output_dir, handle, "terminal.txt"),
    )
    to_launch.append(command)
    port += 1

print("\n" + "="*15 + f" Total: {len(to_launch)} Commands to launch: " + "="*15 + "\n")
for cmd in to_launch:
    print(cmd)
    
print("\nDone!")