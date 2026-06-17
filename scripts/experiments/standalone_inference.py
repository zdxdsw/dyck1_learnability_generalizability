"""
Assumed directory layout:

OUTPUT_DIR/
  task_suite/
    task/
      handle/
        config.json
        terminal.txt
        training_metrics.txt   # written by this script

CKPT_DIR/
  task_suite/
    task/
      handle/
        4_15625_Transformer.pt
        ...

Example:
    python standalone_inference.py
"""

import os, re, ast, json, random, argparse, traceback
from types import SimpleNamespace
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rnn_model import RNNModel
from transformer_model import TransformerModel
from dataset import *
from utils import sequence_collator_builder, inference


# ---------------------------------------------------------------------
# User-facing defaults: edit these
# ---------------------------------------------------------------------

MODEL = "Transformer"  # "RNN" or "Transformer"
OUTPUT_DIR = f"/home/yingshac/workspace/tmlr/scripts/experiments/output_{MODEL.lower()}"
CKPT_DIR = f"/data/yingshac/tmlr/ckpts_{MODEL.lower()}"
EVAL_BATCH_SIZE = 1024

TASK_SUITES = [
    # "S1_d_trulyleq[2-8]_4a",
    # "S1_d_trulyleq[2-8]_4A",
    # "S1_d_trulyleq[2-8]_4b",
    # "S1_d_trulyleq[2-8]_4B",
    # "S1_d_trulyleq[2-8]_4c",
    # "S1_d_trulyleq[2-8]_4C",
    # "S1_d_trulyleq[2-8]_4d",
    # "S1_d_trulyleq[2-8]_4D",
    # "S1_d_trulyleq[2-8]_4e",
    # "S1_d_trulyleq[2-8]_4E",
    # "S1_d_trulyleq[2-8]_4f",
    "S1_d_trulyleq[2-8]_2E",
    "S1_d_trulyleq[2-8]_2d",
    "S1_d_[2-8]_4C",
    "S1_d_[2-8]_4A",
]


# ---------------------------------------------------------------------
# Printing / logging utilities
# ---------------------------------------------------------------------

class Tee:
    """Write stdout/stderr to both terminal and a file."""

    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def Print(s=""): print(s, flush=True)


# ---------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------

def load_config_from_json(config_path):
    with open(config_path, "r") as f:
        return SimpleNamespace(**json.load(f))


def parse_task_d(task):
    """
    Extract d from task names like: S1_d_trulyleq5, S1_d_5, S2a_d_7
    Returns the last int d.
    """
    patterns = [
        r"d_trulyleq(\d+)",
        r"d_leq(\d+)",
        r"d_(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, task)
        if m:
            return int(m.group(1))

    raise ValueError(f"Func parse_task_d(): Could not infer d from task name: {task}")


def infer_dataset_name_from_train_file(train_file):
    """
    Infer dataset name from filenames like:
        /data/yingshac/tmlr/train/fix_d_2.json -> d_2
    """
    
    base = os.path.basename(train_file)
    return base.split(".")[0].split("fix_")[-1] 


def get_last_nonempty_line(lines):
    for line in reversed(lines):
        if line.strip():
            return line.strip()
    return ""


def safe_literal_eval_metrics(line):
    """
    Parse a metrics dict line such as:
        {'num_samples': 10000, 'loss': 0.0, 'acc': 1.0}

    Uses ast.literal_eval rather than eval.
    """
    return ast.literal_eval(line.strip())


def get_best_epoch_id(output_dir, task_suite, task, handle, same_d):
    """
    Read terminal.txt and select the epoch with smallest same-d validation loss.
    Tie-breaking: If multiple epochs have the same loss, choose the largest epoch_id.
    Returns: int epoch_id, or None if the run should be skipped.
    """
    terminal_path = os.path.join(output_dir, task_suite, task, handle, "terminal.txt")

    if not os.path.exists(terminal_path):
        Print(f"Func get_best_epoch_id(): [skip] Missing terminal.txt: {terminal_path}")
        return None

    with open(terminal_path, "r") as f: lines = f.readlines()
    # the last non-empty line should print f"Finish!!! {hangle}", so we can check if the job is finished
    if not lines[-1].startswith(f"Finish!!! {handle}"):
        Print(f"Func get_best_epoch_id(): [skip] Job {task_suite}/{task}/{handle} is not finished.")
        return None
    
    best_epoch_id = -1
    smallest_same_d_val_loss = float("inf")
    for i in range(len(lines)):
        line = lines[i]
        if line.startswith("Epoch") and "Metrics on Validation Sets" in line:
            epoch_line_idx = i
            epoch_id = int(line.strip().split("Epoch ")[1].split(" Step")[0])
            # extract the metrics for each validation set
            for j in range(epoch_line_idx+1, len(lines)):
                line = lines[j]
                if line.startswith("Epoch") and "Metrics on Validation Sets" in line:
                    break
                if line.startswith("Validation set: d_"):
                    val_d = int(line.strip().split("Validation set: d_")[1])
                    if val_d != same_d: continue
                    metrics = eval(lines[j+1].strip())
                    if metrics["loss"] <= smallest_same_d_val_loss:
                        smallest_same_d_val_loss = metrics["loss"]
                        best_epoch_id = epoch_id

    if best_epoch_id == -1:
        Print(f"Func get_best_epoch_id(): [skip] Cannot find best_val_loss Metrics in {task_suite}/{task}/{handle}")
        return None

    Print(f"Best epoch: {best_epoch_id}, same-d val loss: {smallest_same_d_val_loss}")
    return best_epoch_id


def find_ckpt_path(ckpt_dir, epoch_id):
    """
    Find checkpoint file whose name starts with: {epoch_id}_
    ckpt_dir already integrates task_suite/task/handle
    
    Example: epoch_id=4 -> 4_15625_Transformer.pt
    """
    if not os.path.isdir(ckpt_dir):
        Print(f"Func find_ckpt_path(): Checkpoint FileNotFoundError: {ckpt_dir}")
        return None

    candidates = [
        fn for fn in os.listdir(ckpt_dir)
        if fn.endswith(".pt") and fn.startswith(f"{epoch_id}_")
    ]

    if not candidates:
        Print(f"Func find_ckpt_path(): No .pt checkpoint starting with '{epoch_id}_' found in {ckpt_dir}")
        return None

    if len(candidates) > 1:
        # Usually there should be only one checkpoint per epoch.
        # If not, choose the one with largest step id.
        def step_id(fn):
            parts = fn.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
            return -1

        candidates = sorted(candidates, key=step_id)
        Print(f"Func find_ckpt_path(): [warn] Multiple ckpts found for epoch {epoch_id}; using latest step: {candidates[-1]}")

    return os.path.join(ckpt_dir, candidates[-1])


# ---------------------------------------------------------------------
# Dataset / model construction
# ---------------------------------------------------------------------

def get_model_class(config):
    model_class_name = f"{config.model}Model"
    return globals()[model_class_name]

def get_dataset_class(config):
    dataset_class_name = f"{config.dataset}Dataset"
    return globals()[dataset_class_name]


def build_eval_dataloader(dataset, config):
    collator = sequence_collator_builder(config)
    return DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.eval_batch_size, # override training configs
        collate_fn=collator,
        pin_memory=True,
        num_workers=1,
        persistent_workers=False,
    )


def build_train_dataloaders(config):
    """
    Unlike trainer.py, this creates one train dataset per train file, with inferred dataset names.
    """
    DATASET_CLASS = get_dataset_class(config)
    dataloaders = []

    for train_file in config.train_files:
        name = infer_dataset_name_from_train_file(train_file)
        dataset = DATASET_CLASS([train_file], config, name=name)
        dataloaders.append(build_eval_dataloader(dataset, config))

    return dataloaders


def build_val_dataloaders(config):
    DATASET_CLASS = get_dataset_class(config)
    dataloaders = []

    for f in config.val_files:
        # following trainer.py:
        # DATASET_CLASS([f[0]], config, name=f[1])
        dataset = DATASET_CLASS([f[0]], config, name=f[1])
        dataloaders.append(build_eval_dataloader(dataset, config))

    return dataloaders


def build_test_dataloaders(config):
    DATASET_CLASS = get_dataset_class(config)
    dataloaders = []

    for f in config.test_files:
        # following trainer.py:
        # DATASET_CLASS([f[0]], config, name=f[1], specific_length=f[2])
        dataset = DATASET_CLASS([f[0]], config, name=f[1], specific_length=f[2])
        dataloaders.append(build_eval_dataloader(dataset, config))

    return dataloaders


def load_model_from_ckpt(config, ckpt_path, device):
    MODEL_CLASS = get_model_class(config)

    model = MODEL_CLASS(config)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


# ---------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------

@torch.no_grad()
def run_metrics_for_dataloaders(section_name, model, dataloaders, device, config):

    for dataloader in dataloaders:
        metrics = inference(
            model,
            dataloader,
            device,
            config,
        )
        Print(f"{section_name}{dataloader.dataset.name}")
        Print(metrics)


def run_one_handle(output_dir, ckpt_root, task_suite, task, handle, device):
    handle_output_dir = os.path.join(output_dir, task_suite, task, handle)
    config_path = os.path.join(handle_output_dir, "config.json")

    if not os.path.exists(config_path):
        Print(f"Func run_one_handle(): [skip] Missing config.json: {config_path}")
        return

    config = load_config_from_json(config_path)

    if hasattr(config, "hf_cache_dir") and config.hf_cache_dir is not None:
        os.environ["HF_HOME"] = config.hf_cache_dir

    # Reproducibility is not crucial for deterministic eval, but keep consistent with trainer.py.
    if hasattr(config, "seed"):
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

    Print("=" * 80)
    Print(f"Task suite: {task_suite}")
    Print(f"Task:       {task}")
    Print(f"Handle:     {handle}")
    Print(f"Config:     {config_path}")

    same_d = parse_task_d(task)
    best_epoch_id = get_best_epoch_id(
        output_dir=output_dir,
        task_suite=task_suite,
        task=task,
        handle=handle,
        same_d=same_d,
    )
    if best_epoch_id is None: return

    ckpt_dir = os.path.join(ckpt_root, task_suite, task, handle)
    ckpt_path = find_ckpt_path(ckpt_dir, best_epoch_id)
    if ckpt_path is None: return
    
    Print(f"Checkpoint: {ckpt_path}")
    Print("=" * 80)

    model = load_model_from_ckpt(config, ckpt_path, device)

    train_dataloaders = build_train_dataloaders(config)
    # val_dataloaders = build_val_dataloaders(config)
    # test_dataloaders = build_test_dataloaders(config)

    Print(f"\nEpoch {best_epoch_id} Metrics on Training Sets:\n")
    run_metrics_for_dataloaders("Training set: ", model, train_dataloaders, device, config)

    # Print(f"\nEpoch {best_epoch_id} Metrics on Validation Sets:")
    # run_metrics_for_dataloaders("Validation set: ", model, val_dataloaders, device, config)

    # Print(f"\nEpoch {best_epoch_id} Metrics on Testing Sets:")
    # run_metrics_for_dataloaders("Testing set: ", model, test_dataloaders, device, config)


def collect_handles(output_dir, task_suites):
    handles = []
    for task_suite in task_suites:
        task_suite_dir = os.path.join(output_dir, task_suite)

        if not os.path.isdir(task_suite_dir):
            Print(f"Func iter_handles(): [skip] Missing task_suite directory: {task_suite_dir}")
            continue

        for task in sorted(os.listdir(task_suite_dir)):
            task_dir = os.path.join(task_suite_dir, task)
            if not os.path.isdir(task_dir):
                Print(f"Func iter_handles(): [skip] Missing task directory: {task_dir}")
                continue

            for handle in sorted(os.listdir(task_dir)):
                handle_dir = os.path.join(task_dir, handle)
                if not os.path.isdir(handle_dir):
                    continue

                handles.append((task_suite, task, handle))

    return handles

def run_all(args):
    task_suites = args.task_suites
    device = f"cuda:{args.cuda}"
    device = torch.device(device)
    Print(f"Using device: {device}")
    if device.type == "cuda": torch.cuda.set_device(device)

    handles = collect_handles(args.output_dir, task_suites)
    Print(f"Found {len(handles)} handles.")

    for task_suite, task, handle in tqdm(handles, file=os.sys.__stderr__): # force tqdm to use the original terminal stream
        handle_output_dir = os.path.join(args.output_dir, task_suite, task, handle)
        metrics_path = os.path.join(handle_output_dir, "training_metrics.txt")
        os.makedirs(handle_output_dir, exist_ok=True)

        with open(metrics_path, "w") as f:
            tee = Tee(f)  # <- inner loop output will be written to file only; no console

            with redirect_stdout(tee), redirect_stderr(tee):
                try:
                    run_one_handle(
                        output_dir=args.output_dir,
                        ckpt_root=args.ckpt_dir,
                        task_suite=task_suite,
                        task=task,
                        handle=handle,
                        device=device,
                    )

                    Print(f"\nFinish!!! {task_suite}/{task}/{handle}")

                except Exception:
                    Print(f"\n[error] Failed on {task_suite}/{task}/{handle}")
                    traceback.print_exc()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--eval_batch_size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Root output directory containing task_suite/task/handle/config.json.",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=CKPT_DIR,
        help="Root checkpoint directory containing task_suite/task/handle/*.pt.",
    )
    # nargs="*" means that this argument can take zero or more values
    #           argparse will collect them into a Python list.
    # Use: --task_suites 'S1_d_[2-8]_4B' 'S1_d_[2-8]_4C' 'S1_d_[2-8]_4F'
    parser.add_argument("--task_suites", nargs="*", default=TASK_SUITES)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all(args)