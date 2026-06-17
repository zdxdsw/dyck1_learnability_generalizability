import random, time, os, pytz, argparse, sys, json, warnings
# from config import *
# from config_taskspecific import *
from copy import deepcopy
from configs import flat_everything_config_builder
from utils import add_suffix_to_outermost_dir, add_suffix_to_innermost_dir
from tqdm import trange
from datetime import datetime
timezone = pytz.timezone('America/New_York') 

if os.path.exists('/data/yingshac/'): 
    os.environ['HF_HOME'] = '/data/yingshac/hf_cache'

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default="")
parser.add_argument('--sleep', type=int)
parser.add_argument('--port', type=str, default="")
parser.add_argument('--cuda', type=str, default="")
args = parser.parse_args()
if args.cuda: args.cuda = f"CUDA_VISIBLE_DEVICES=\"{args.cuda}\""

"""------------- Preparing configs -------------"""
base_config = flat_everything_config_builder(args.task)
base_config.output_dir = add_suffix_to_outermost_dir(base_config.output_dir, f"_{base_config.model.lower()}")
base_config.ckpt_dir = base_config.ckpt_dir.replace("ckpts", f"ckpts_{base_config.model.lower()}")
# add_suffix_to_innermost_dir(base_config.ckpt_dir, f"_{base_config.model.lower()}")
base_config.output_dir = os.path.join(base_config.output_dir, base_config.task)
base_config.ckpt_dir = os.path.join(base_config.ckpt_dir, base_config.task)

"""------------ Inline Resume Block ------------"""
if base_config.load_from_dir is not None:
    resume_config_path = os.path.join(
        base_config.output_dir,
        base_config.load_from_dir,
        "config.json",
    )

    with open(resume_config_path, "r") as f: resume_from_config = json.load(f)

    skip_keys = {
        "learning_rate",
        "num_epochs",
        "logging_steps",
        "load_from_dir",
        "date",
    }

    for k in vars(base_config): # use vars(config) instead of dir(config) because it only iterates over actual config attributes.
        if k in skip_keys: continue
        if k in resume_from_config:
            setattr(base_config, k, resume_from_config[k])
        else:
            warnings.warn(
                f"Config key {k!r} not found in resumed config "
                f"{resume_config_path!r}; keeping current value: "
                f"{getattr(base_config, k)!r}"
            )

"""-------------- Sleep if needed --------------"""
if args.sleep is not None:
  for i in trange(args.sleep, desc="Sleeping"):
    time.sleep(60)

"""---------- Launch Individual Jobs -----------"""
SEEDS = base_config.seed
if isinstance(SEEDS, int): SEEDS = [SEEDS]
for seed in SEEDS:
  config = deepcopy(base_config)
  date = datetime.now(timezone).strftime("%m%d_%H%M%S")
  config.date = date
  config.seed = seed

  os.makedirs(os.path.join(config.ckpt_dir, config.date), exist_ok=True)
  os.makedirs(os.path.join(config.output_dir, config.date), exist_ok=True)

  with open(os.path.join(config.output_dir, config.date, "config.json"), "w") as f:
    json.dump(vars(config), f, indent=2) # dump config

  os.system("{} accelerate launch --main_process_port {} --num_processes 1 trainer.py --date {} --task {} --output_dir '{}' | tee {}".format(
    args.cuda,
    args.port,
    config.date,
    args.task,
    config.output_dir,
    os.path.join(config.output_dir, date, "terminal.txt"),
  ))