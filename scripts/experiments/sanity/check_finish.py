import os
from tqdm import tqdm

MODEL = "transformer"
OUTPUT_DIR = f"/home/yingshac/workspace/tmlr/scripts/experiments/output_{MODEL}/"

TASK_SUITES = [
    "S3_d_trulyleq[2-8]_2A",
    "S3_d_trulyleq[2-8]_2B",
    "S3_d_trulyleq[2-8]_2C",
    "S3_d_trulyleq[2-8]_2D",
    "S3_d_trulyleq[2-8]_2E",
    "S3_d_trulyleq[2-8]_2F",
    "S3_d_trulyleq[2-8]_2G",
]

TASKS = [
    "S1_d_trulyleq8",
    "S2_d_trulyleq8",
    "S3_d_trulyleq8",
    "S1_d_8",
    "S2_d_8",
    "S3_d_8",
]

CHECK_TERMINAL = False
if not CHECK_TERMINAL: print("Don't check terminal.txt.")
CHECK_TRAINING_METRICS = False
if not CHECK_TRAINING_METRICS: print("Haven't checked training metrics.")

CHECK_STRESS_TEST_POS_METRICS = True
if not CHECK_STRESS_TEST_POS_METRICS: print("Haven't checked stress test pos metrics.")
CHECK_STRESS_TEST_NEG_METRICS = True
if not CHECK_STRESS_TEST_NEG_METRICS: print("Haven't checked stress test neg metrics.")
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
        if TASKS and task not in TASKS: continue
        task_dir = os.path.join(task_suite_dir, task)
        for handle in os.listdir(task_dir):
            handle_dir = os.path.join(task_dir, handle)
            terminal_path = os.path.join(handle_dir, "terminal.txt")
            training_metrics_path = os.path.join(handle_dir, "training_metrics.txt")
            
            # Check terminal.txt
            if CHECK_TERMINAL:
                if not os.path.exists(terminal_path):
                    tqdm.write(f"Job Missing: {task_suite}/{task}/{handle}")
                else:
                    with open(terminal_path, "r") as f:
                        lines = [line.strip() for line in f if line.strip()]
                        if not lines or "Finish!!!" not in lines[-1]:
                            tqdm.write(f"Job Unfinished: {task_suite}/{task}/{handle}")
            
            if CHECK_TRAINING_METRICS:
                # Check training_metrics.txt
                if not os.path.exists(training_metrics_path):
                    tqdm.write(f"Training Metrics Missing: {task_suite}/{task}/{handle}")
                else:
                    with open(training_metrics_path, "r") as f:
                        lines = [line.strip() for line in f if line.strip()]
                        if not lines or "Finish!!!" not in lines[-1]:
                            tqdm.write(f"Training Metrics Unfinished: {task_suite}/{task}/{handle}")

            if CHECK_STRESS_TEST_POS_METRICS: 
                # Check stress_test_positives_metrics.txt
                stress_test_pos_metrics_path = os.path.join(handle_dir, "stress_test_positives_metrics.txt")
                if not os.path.exists(stress_test_pos_metrics_path):
                    tqdm.write(f"Stress Test Pos Metrics Missing: {task_suite}/{task}/{handle}")
                else:
                    with open(stress_test_pos_metrics_path, "r") as f:
                        lines = [line.strip() for line in f if line.strip()]
                        if not lines or "Finish!!!" not in lines[-1]:
                            tqdm.write(f"Stress Test Pos Metrics Unfinished: {task_suite}/{task}/{handle}")
                
            if CHECK_STRESS_TEST_NEG_METRICS:
                # Check stress_test_negatives_metrics.txt
                stress_test_neg_metrics_path = os.path.join(handle_dir, "stress_test_negatives_metrics.txt")
                if not os.path.exists(stress_test_neg_metrics_path):
                    tqdm.write(f"Stress Test Neg Metrics Missing: {task_suite}/{task}/{handle}")
                else:
                    with open(stress_test_neg_metrics_path, "r") as f:
                        lines = [line.strip() for line in f if line.strip()]
                        if not lines or "Finish!!!" not in lines[-1]:
                            tqdm.write(f"Stress Test Neg Metrics Unfinished: {task_suite}/{task}/{handle}")

print("\nDone!!!\n")