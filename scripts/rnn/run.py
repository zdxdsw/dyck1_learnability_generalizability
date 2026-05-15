import random, time, os, pytz, argparse, sys, json, warnings
from config import *
from config_taskspecific import *
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
if len(args.task): config = eval(f"{args.task}_Config")()
else: config = Basic_Config()
default_config = Default_Config()

config.task = args.task
config.ckpt_dir = os.path.join(config.ckpt_dir, config.task)
config.output_dir = os.path.join(config.output_dir, config.task)

SEEDS = config.seed
if isinstance(SEEDS, int): SEEDS = [SEEDS]
for seed in SEEDS:
  date = datetime.now(timezone).strftime("%m%d_%H%M%S")
  config.date = date
  config.seed = seed

  if config.output_dir is not None:
    os.makedirs(os.path.join(config.ckpt_dir, config.date), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, config.date), exist_ok=True)

    if config.load_from_dir is not None:
      resume_from_config = json.load(open(os.path.join(config.output_dir, config.load_from_dir, "config.json"), "r"))
      config_keys = dir(config)
      for k in config_keys:
        if k.startswith("__"): continue
        if k not in ["learning_rate", "num_epochs", "logging_steps", "load_from_dir", "date"]:
          if k in resume_from_config:
            setattr(config, k, resume_from_config[k])
          else:
            setattr(config, k, default_config.__getattribute__(k))
            warnings.warn(f"Cannot find {k} in the resume_from_config. Set to {default_config.__getattribute__(k)} by default.")

    # dump config
    C = {k:config.__getattribute__(k) for k in dir(config) if not k.startswith("__")}
    with open(os.path.join(config.output_dir, config.date, "config.json"), "w") as f:
      json.dump(C, f, indent=2)

  """------------- Sleep if needed -------------"""
  if args.sleep is not None:
    for i in trange(args.sleep, desc="Sleeping"):
      time.sleep(60)

  os.system("{} accelerate launch --main_process_port {} --num_processes 1 trainer.py --date {} --task {} | tee {}".format(
    args.cuda,
    args.port,
    config.date,
    args.task,
    os.path.join(config.output_dir, date, "terminal.txt"),
  ))