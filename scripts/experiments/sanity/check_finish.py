import os
from tqdm import tqdm

MODEL = "rnn"
OUTPUT_DIR = f"/home/yingshac/workspace/tmlr/scripts/experiments/output_{MODEL}/"

TASK_SUITES = [
    # "S1_d_[2-8]_A",
    "S1_d_[2-8]_B",
    # "S1_d_[2-8]_C",
    # "S1_d_[2-8]_D",
    # "S1_d_[2-8]_E",
    # "S1_d_[2-8]_F",
    # "S1_d_trulyleq[2-8]_A",
    "S1_d_trulyleq[2-8]_B",
    # "S1_d_trulyleq[2-8]_C",
    # "S1_d_trulyleq[2-8]_D",
    # "S1_d_trulyleq[2-8]_E",
    # "S1_d_trulyleq[2-8]_F",
    # "S2_d_trulyleq[2-8]_A",
    "S2_d_trulyleq[2-8]_B",
    # "S2_d_trulyleq[2-8]_C",
    # "S2_d_trulyleq[2-8]_D",
    # "S2_d_trulyleq[2-8]_E",
    # "S2_d_trulyleq[2-8]_F",
    # "S2_d_[2-8]_A",
    "S2_d_[2-8]_B",
    # "S2_d_[2-8]_C",
    # "S2_d_[2-8]_D",
    # "S2_d_[2-8]_E",
    # "S2_d_[2-8]_F",
    # "S3_d_trulyleq[2-8]_A",
    "S3_d_trulyleq[2-8]_B",
    # "S3_d_trulyleq[2-8]_C",
    # "S3_d_trulyleq[2-8]_D",
    # "S3_d_trulyleq[2-8]_E",
    # "S3_d_trulyleq[2-8]_F",
    # "S3_d_[2-8]_A",
    "S3_d_[2-8]_B",
    # "S3_d_[2-8]_C",
    # "S3_d_[2-8]_D",
    # "S3_d_[2-8]_E",
    # "S3_d_[2-8]_F", 
]

CHECK_TRAINING_METRICS = True
if not CHECK_TRAINING_METRICS: print("Haven't checked training metrics.")
"""
Iterate through OUTPUT_DIR: task_suite/task/handle

read terminal.txt: if finished, the last non-empty line should contain "Finish!!!"
    If terminal.txt doesn't exist, print Job Missing: task_suite/task/handle
    If not finished, print Job Unfinished: task_suite/task/handle

read training_metrics.txt: if finished, the last non-empty line should contain "Finish!!!"
    If training_metrics.txt doesn't exist, print Training Metrics Missing: task_suite/task/handle
    If not finished, print Training Metrics Unfinished: task_suite/task/handle
"""

for task_suite in TASK_SUITES:
    task_suite_dir = os.path.join(OUTPUT_DIR, task_suite)
    for task in tqdm(os.listdir(task_suite_dir), desc=f"Checking {task_suite}", unit="task"):
        task_dir = os.path.join(task_suite_dir, task)
        for handle in os.listdir(task_dir):
            handle_dir = os.path.join(task_dir, handle)
            terminal_path = os.path.join(handle_dir, "terminal.txt")
            training_metrics_path = os.path.join(handle_dir, "training_metrics.txt")
            
            # Check terminal.txt
            if not os.path.exists(terminal_path):
                tqdm.write(f"Job Missing: {task_suite}/{task}/{handle}")
            else:
                with open(terminal_path, "r") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    if not lines or "Finish!!!" not in lines[-1]:
                        tqdm.write(f"Job Unfinished: {task_suite}/{task}/{handle}")
            
            if not CHECK_TRAINING_METRICS: continue
            # Check training_metrics.txt
            if not os.path.exists(training_metrics_path):
                tqdm.write(f"Training Metrics Missing: {task_suite}/{task}/{handle}")
            else:
                with open(training_metrics_path, "r") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    if not lines or "Finish!!!" not in lines[-1]:
                        tqdm.write(f"Training Metrics Unfinished: {task_suite}/{task}/{handle}")

print("\nDone!!!\n")