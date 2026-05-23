import os, sys, json, random, io, pytz, argparse, pytz
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from functools import partial
from model import RNNModel
from dataset import *
from config import *
from utils import sequences_collator, inference
from config_taskspecific import *
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.nn.attention import sdpa_kernel, SDPBackend

def Print(s):
   if not Accelerator().process_index: print(s)


parser = argparse.ArgumentParser()
parser.add_argument('--date', type=str, default="debug")
parser.add_argument('--task', type=str)
parser.add_argument('--output_dir', type=str)
args = parser.parse_args()


"""----------- Preparing Accelerator -----------"""

config = eval(f"{args.task}_Config")()
tmp = json.load(open(os.path.join(args.output_dir, args.date, "config.json"), "r"))
for k, v in tmp.items(): setattr(config, k, v)
if "hf_cache_dir" in dir(config) and config.hf_cache_dir is not None: os.environ['HF_HOME'] = config.hf_cache_dir

# Fix all seeds to ensure reproducibility
SEED = config.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.enabled = True

ddp_kwargs = DistributedDataParallelKwargs()
accelerator = Accelerator(
    mixed_precision="fp16",
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    kwargs_handlers=[ddp_kwargs],
    project_dir=os.path.join(config.output_dir, config.date),
)
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    os.makedirs(os.path.join(config.ckpt_dir, config.date), exist_ok=True)


Print(f"\n------------- Preparing {config.model} model -------------")

model = eval(f"{config.model}Model")(config)
Print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# use AdamW optimizer, constant lr, weight_decay applies to all parameters
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=config.learning_rate, 
    weight_decay=config.weight_decay,
)


Print("\n------------- Preparing data -------------")

DATASET_CLASS = eval(f"{config.dataset}Dataset")
train_dataset = DATASET_CLASS(config.train_files, config.in_vocab, config.out_vocab)
val_datasets = [DATASET_CLASS([f[0]], config.in_vocab, config.out_vocab, name=f[1]) for f in config.val_files]
test_datasets = [DATASET_CLASS([f[0]], config.in_vocab, config.out_vocab, name=f[1], specific_length=f[2]) for f in config.test_files]

collator = partial(sequences_collator, 
                    in_vocab=config.in_vocab,
                    max_seq_len=config.max_seq_len,
                )

print("config.shuffle_training_data = ", config.shuffle_training_data)
train_dataloader = DataLoader(train_dataset, shuffle=config.shuffle_training_data, batch_size=config.per_device_train_batch_size, collate_fn=collator, pin_memory=True, num_workers=1, persistent_workers=False)
val_dataloaders = [DataLoader(val_dataset, shuffle=False, batch_size=config.per_device_eval_batch_size, collate_fn=collator, pin_memory=True, num_workers=1, persistent_workers=False) for val_dataset in val_datasets]
test_dataloaders = [DataLoader(test_dataset, shuffle=False, batch_size=config.per_device_eval_batch_size, collate_fn=collator, pin_memory=True, num_workers=1, persistent_workers=False) for test_dataset in test_datasets]

Print(f"num train = {len(train_dataset)}")
Print(f"num val (per file) = {[len(val_dataset) for val_dataset in val_datasets]}")
Print(f"num val total = {sum(len(val_dataset) for val_dataset in val_datasets)}")
Print(f"num test (per file) = {[len(test_dataset) for test_dataset in test_datasets]}")
Print(f"num test total = {sum(len(test_dataset) for test_dataset in test_datasets)}")

"""------------ Prepare Initialization ------------"""

model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

global_step = 0
global_epoch = 0

# Resume from ckpt
if config.load_from_dir is not None:
    ckpt_dir = os.path.join(config.ckpt_dir, config.load_from_dir)
    load_from_pt = sorted(os.listdir(ckpt_dir), key=lambda x: int(x.split("_")[1]))[-1]

    global_epoch = int(load_from_pt.split("_")[0]) + 1
    global_step = int(load_from_pt.split("_")[1]) + 1

    Print(f"resume from ckpt: {load_from_pt}\n\tepoch {global_epoch} step {global_step}")
    torch.cuda.set_device(accelerator.device)
    state_dict = torch.load(os.path.join(ckpt_dir, load_from_pt), map_location=accelerator.device)
    _model = accelerator.unwrap_model(model)
    _model.load_state_dict(state_dict, strict=True)

if config.init_from_ckpt is not None:
    init_from_ckpt = os.path.join(config.ckpt_dir, config.init_from_ckpt)
    accelerator.print(f"init from ckpt: {init_from_ckpt}")
    torch.cuda.set_device(accelerator.device)
    state_dict = torch.load(init_from_ckpt, map_location=accelerator.device)
    _model = accelerator.unwrap_model(model)
    _model.load_state_dict(state_dict, strict=True)


Print(f"------------ Start job {config.task}, {config.date} ------------")
progress_bar = tqdm(total=(config.num_epochs-global_epoch)*len(train_dataloader), disable=not accelerator.is_local_main_process, mininterval=10)
for epoch in range(global_epoch, config.num_epochs):

    for step, batch in enumerate(train_dataloader):
        
        ### One training step
        accelerator.wait_for_everyone()
        # with accelerator.autocast() as autocast, torch.backends.cuda.sdp_kernel(enable_flash=False) as disable:
        # "torch.backends.cuda.sdp_kernel(...)" has been replace by "torch.nn.attention.sdpa_kernel(...)"
        # the equivalent intent is "disable FlashAttention, allow the other SDPA backends."
        with accelerator.autocast(), sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
            optimizer.zero_grad()
                        
            logits = model(
                batch['input_id'].to(accelerator.device),
            )

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), # bs*seq_len, vocab_size
                batch['label'].to(accelerator.device).view(-1),
                ignore_index=-1,
                reduction="mean",
            )

            accelerator.backward(loss / config.gradient_accumulation_steps)

            if (global_step+1) % config.gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                #lr_scheduler.step() # Currently disable lr_scheduler
            
        logs = {"loss": loss.detach().item(), "step": global_step}
        accelerator.log(logs, step=global_step)
        global_step += 1

        if global_step % config.logging_steps == 0: progress_bar.set_postfix(**logs)
        progress_bar.update(1)
        
    
    ### Evaluate on validation and testing sets at the end of every epoch
    with accelerator.autocast():
        if accelerator.is_main_process and ((1+epoch) % config.more_eval_th_by_epoch == 0 or epoch==config.num_epochs-1):
            model_to_eval = accelerator.unwrap_model(model)
            model_to_eval.eval()
            Print(f"\nEpoch {epoch} Step {global_step} Metrics on Validation Sets:\n")
            
            val_dataloader = val_dataloaders[0]
            metrics = inference(
                model_to_eval,
                val_dataloader,
                accelerator.device,
                config
            )
            Print(f"Validation set: {val_dataloader.dataset.name}")
            Print(metrics)
            more_eval = True if (metrics["acc"] >= config.more_eval_th_by_val_acc or epoch==config.num_epochs-1) else False

            for val_dataloader in val_dataloaders[1:]:
                metrics = inference(
                    model_to_eval,
                    val_dataloader,
                    accelerator.device,
                    config
                )
                Print(f"Validation set: {val_dataloader.dataset.name}")
                Print(metrics)

            if more_eval:
                Print(f"\nEpoch {epoch} Step {global_step} Metrics on Testing Sets:\n")
                num_test_samples, total_test_correct = 0, 0
                for test_dataloader in test_dataloaders:
                    metrics = inference(
                        model_to_eval,
                        test_dataloader,
                        accelerator.device,
                        config
                    )
                    Print(f"Testing set: {test_dataloader.dataset.name}")
                    Print(metrics)
                    num_test_samples += metrics["num_samples"]
                    total_test_correct += metrics["num_samples"] * metrics["acc"]
                Print(f"\nOverall Testing Accuracy: {round(total_test_correct/num_test_samples, 2)}\n")

            model_to_eval.train()
    accelerator.wait_for_everyone()
    
    save_path = os.path.join(config.ckpt_dir, config.date, f"{epoch}_{global_step}_{config.model}.pt")
    torch.save(accelerator.unwrap_model(model).state_dict(), save_path)

accelerator.wait_for_everyone()
accelerator.end_training()
accelerator.free_memory()

# if accelerator.is_main_process:
#     Print("\n\n------------- Start Inference -------------")
#     infr_command = "python tester.py --handle {} --test_files \"{}\"".format(config.date, " ".join(["val"]+config.test_files))
#     os.system(infr_command)
Print(f"Finish!!! {config.date}")
os._exit(0)