from aalpy.learning_algs import run_Lstar
from aalpy.oracles import StatePrefixEqOracle, TransitionFocusOracle
from aalpy.utils import save_automaton_to_file, visualize_automaton

import os, sys, pytz, json, glob, argparse, random
from datetime import datetime
timezone = pytz.timezone('America/New_York') 
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from types import SimpleNamespace
from copy import deepcopy

from pathlib import Path
THIS_FILE = Path(__file__).resolve()
RNN_DIR = THIS_FILE.parents[1]
SCRIPTS_DIR = THIS_FILE.parents[2]
EXTRACTION_DIR = SCRIPTS_DIR / "extraction"
DYNET_PYTHON_DIR = EXTRACTION_DIR / "dynet" / "python"

for p in [str(DYNET_PYTHON_DIR), str(EXTRACTION_DIR), str(RNN_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# sanity check
import dynet_config
print("dynet_config from:", dynet_config.__file__) 
# >>dynet_config from: /home/yingshac/workspace/tmlr/scripts/extraction/dynet/python/dynet_config.py

from rnn_model import RNNModel
from config import *
from config_taskspecific import *
from dataset import *
from utils import *

from DataProcessing import parse_data, preprocess_binary_classification_data
from RNN_SULs import RnnBinarySUL
from TrainAndExtract import train_RNN_on_tomita_grammar, train_and_extract_bp, train_RNN_and_extract_FSM

def load_config_from_json(config_path):
    with open(config_path, "r") as f:
        return SimpleNamespace(**json.load(f))
    
class TorchRNNAdapter:
    """
    Adapter so a PyTorch RNNModel can be used by RnnBinarySUL.

    Expected by RnnBinarySUL:
        rnn.rnn.initial_state()
        rnn.state
        rnn.step(letter)
        rnn.renew()
    """

    def __init__(self, torch_model, device=None, output_position=-1):
        self.torch_model = torch_model
        self.config = torch_model.config

        if device is None: device = next(torch_model.parameters()).device
        self.device = device

        self.torch_model.to(self.device)
        self.torch_model.eval()

        # RnnBinarySUL expects rnn.rnn.initial_state()
        self.rnn = self # shim so RnnBinarySUL.pre() works
        self.state = None

        self.alphabet = [i for i in self.config.in_vocab if i not in ['<pad>', '<eos>']]

        self.token_dict = {tok: i for i, tok in enumerate(self.config.in_vocab) if i not in ['<pad>', '<eos>']}
        
        if '<eos>' in self.config.in_vocab:
            self.consume_eos_and_classify = True
            self.eos_id = self.config.in_vocab.index('<eos>')
        else:
            self.consume_eos_and_classify = False

        self.output_position = output_position

    def initial_state(self):
        num_layers = self.config.num_hidden_layers
        hidden_size = self.config.hidden_size

        return torch.zeros(
            num_layers,
            1,              # batch size = 1 for membership query
            hidden_size,
            device=self.device,
        )

    def renew(self):
        self.state = None

    @torch.no_grad()
    def step(self, letter):
        if self.state is None and letter is None:
            # hard-code: accept empty string. return the idx of self.config.out_vocab that is 'a'
            self.state = self.initial_state()
            if 'a' in self.config.out_vocab:
                return self.config.out_vocab.index('a')
            else:
                raise ValueError("Output vocab does not contain 'a', cannot return correct output for empty string")
        if self.state is None:
            self.state = self.initial_state()
        
        out = self.state
        if letter is not None: # do nothing on empty input. 
            token_id = self.token_dict[letter]
            x = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
            x = self.torch_model.encoder(x)              # [1, 1, hidden_size]
            out, self.state = self.torch_model.model(x, self.state)
        
        # run additional transition on <eos> if needed
        if self.consume_eos_and_classify:
            token_id = self.eos_id
            x = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
            x = self.torch_model.encoder(x)              # [1, 1, hidden_size]
            out, _  = self.torch_model.model(x, self.state)

        # decode 
        logits = self.torch_model.decoder(out)       # [1, 1, out_vocab_size]

        logits = logits[:, self.output_position, :]  # [1, out_vocab_size]
        pred = 1-torch.argmax(logits, dim=-1)          # [1]

        return int(pred.item())

# ---------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------

ckpt_dir = "/data/yingshac/tmlr/ckpts"
task_suite = "S1a_d_trulyleq[2-8]_A"
output_dir = f"/home/yingshac/workspace/tmlr/scripts/experiments/output_rnn"
output_dir = os.path.join(output_dir, task_suite)
task = "S1a_d_trulyleq4"
handle = "0526_221713"
epoch = None
""" ckpt paths look like this:
    /data/yingshac/tmlr/ckpts/<task>/<handle>/<epoch>_<step>_RNN.py
    load the most recent epoch (i.e. largest epoch) if epoch is None, otherwise load the specified epoch
    
    config is initialized from `config = eval(f"{args.task}_Config")()`
    
    then load config from training logs: 
    /home/yingshac/workspace/tmlr/scripts/experiments/output_rnn/<task_suite>/<task>/<handle>/config.json
"""
saved_config_path = os.path.join(output_dir, task, handle, "config.json")
assert os.path.exists(saved_config_path), f"Config file not found at {saved_config_path}"
config = load_config_from_json(saved_config_path)

# ---------------------------------------------------------------------
# Initialize and Load Model
# ---------------------------------------------------------------------

print("\nInitialize and load model:")
model = RNNModel(config)
# model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)

if epoch is not None:
    ckpt_path = os.path.join(ckpt_dir, task, handle, f"{epoch}_*_RNN.pt")
else:
    ckpt_path = os.path.join(ckpt_dir, task, handle, f"*_*_RNN.pt")
ckpt_files = glob.glob(ckpt_path)
# get the most recent checkpoint by sorting by epoch number (extracted from filename)
ckpt_files.sort(key=lambda x: int(os.path.basename(x).split("_")[0]), reverse=True)
ckpt_file = ckpt_files[0]
# get actual epoch
epoch = int(os.path.basename(ckpt_file).split("_")[0])

print(f"\n\nLoading epoch={epoch} checkpoint from {ckpt_file}\n")
state_dict = torch.load(ckpt_file, map_location="cpu")
model.load_state_dict(state_dict)

# ---------------------------------------------------------------------
# Extract DFA
# ---------------------------------------------------------------------

print("\nExtract DFA:")

rnn = TorchRNNAdapter(model, device="cuda")
sul = RnnBinarySUL(rnn)

state_eq_oracle = StatePrefixEqOracle(rnn.alphabet, sul, walks_per_state=500, walk_len=40)
# state_eq_oracle = TransitionFocusOracle(rnn.alphabet, sul, num_random_walks=500, walk_len=20, same_state_prob=0.1)

dfa = run_Lstar(alphabet=rnn.alphabet, sul=sul, eq_oracle=state_eq_oracle, automaton_type='dfa',
                cache_and_non_det_check=True, max_learning_rounds=3)

save_dir = f'{task_suite}/{task}/{handle}'
os.makedirs(save_dir, exist_ok=True)
save_automaton_to_file(dfa, os.path.join(save_dir, f'{epoch}_dfa.dot'))

# ---------------------------------------------------------------------
# Check DFA Faithfulness
# ---------------------------------------------------------------------

"""
# ---------------------------------------------------------------------
# Prepare data for conformance check
# ---------------------------------------------------------------------

Randomly sample 8k training sequences, 
                8k validation sequences (val_d > max_train_d),
                8k test sequences.

Useful entries in config:

train_files = ["/data/yingshac/tmlr/train/fix_d_2.json"]
val_files = [
    ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
    ("/data/yingshac/tmlr/validation/fix_d_1.json", "d_1", None),
    ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
    ...
    ("/data/yingshac/tmlr/validation/fix_d_15.json", "d_15", None),
    ("/data/yingshac/tmlr/validation/fix_d_16.json", "d_16", None),
]
test_files = [("/data/yingshac/tmlr/test/fix_d_2.json", f"d_2_l_{l}", l) for l in range(41, 121)]

If <split>_files is a list of strings, then they are already a list of paths.

If <split>_files is a list of tuples (path, name, length), 
    then merge them into a single list of paths, discard name and length info.

DATASET_CLASS = eval(f"{config.dataset}Dataset")
Build three datasets:
    <split>_dataset = DATASET_CLASS(config.<split>_files, config)
    Sample 8k sequences from <split>_dataset.data <- maybe modify this private list.

Build a dataloader for each split, shuffle=False.
"""
parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=str, default="0")
args = parser.parse_args()

device = f"cuda:{args.cuda}"
device = torch.device(device)
if device.type == "cuda": torch.cuda.set_device(device)

EVAL_BATCH_SIZE = 2048
SEED = 42
random.seed(SEED)
DATASET_CLASS = eval(f"{config.dataset}Dataset")

SAMPLE_SIZES = {"train": 8000, "val": 8000, "test": 8000}

if isinstance(config.train_files[0], (tuple, list)):
    train_files = [path for path, _, _ in config.train_files]
else:
    train_files = config.train_files
train_dataset = DATASET_CLASS(train_files, config)
train_dataset.data = random.sample(train_dataset.data, min(SAMPLE_SIZES["train"], len(train_dataset.data)))

if isinstance(config.val_files[0], (tuple, list)):
    val_files = [path for path, _, _ in config.val_files]
else:
    val_files = config.val_files
val_dataset = DATASET_CLASS(val_files, config)
val_dataset.data = random.sample(val_dataset.data, min(SAMPLE_SIZES["val"], len(val_dataset.data)))

if isinstance(config.test_files[0], (tuple, list)):
    test_files = [path for path, _, _ in config.test_files]
else:
    test_files = config.test_files
test_dataset = DATASET_CLASS(test_files, config)
test_dataset.data = random.sample(test_dataset.data, min(SAMPLE_SIZES["test"], len(test_dataset.data)))

collator = sequence_collator_builder(config)

# test_collator needs a larger max_seq_len (e.g. 121): deepcopy config and build a new collator
test_config = deepcopy(config)
test_config.max_seq_len = 121
test_collator = sequence_collator_builder(test_config)

train_dataloader = DataLoader(train_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collator)
val_dataloader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collator)
test_dataloader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=test_collator)

print("\nPrepare data for conformance check:")
print(f"num train = {len(train_dataset)}")
print(f"num val = {len(val_dataset)}")
print(f"num test = {len(test_dataset)}")

"""
# ---------------------------------------------------------------------
# Get RNN predictions
# ---------------------------------------------------------------------

Only get predictions, not metrics for now.

Batched processing:
    forward_result = model_to_eval(
        **prepare_forward_inputs(
            batch=batch, 
            config=config, 
            device=device
        )
    )
    logits = forward_result['logits']

    # Get the last position's prediction that's not padding.
    pred = logits.argmax(dim=-1) # [batch, seq_len]
    mask = labels != ignore_index
    pred = pred[mask]
    label = labels[mask]

Concat pred across batches -> [num_samples=8k]
Convert pred to binary label: config.out_vocab[pred] == 'a' -> 1, else 0.

RNN_predictions = {
    <split>: [pred_1, pred_2, ..., pred_8000],
    ...
}
"""

print(f"\nGet RNN predictions: (Using device: {device})")
RNN_predictions = {}

for split, dataloader in zip(["train", "val", "test"], [train_dataloader, val_dataloader, test_dataloader]):
    all_preds = []
    for batch in tqdm(dataloader, desc=f"Run RNN on {split} samples"):
        
        forward_result = model(
            **prepare_forward_inputs(
                batch=batch, 
                config=config, 
                device=device
            )
        )
        logits = forward_result['logits'] # [batch, seq_len, out_vocab_size]
        pred = logits.argmax(dim=-1) # [batch, seq_len]
        labels = batch['labels']
        mask = labels != getattr(config, "ignore_index", -1)
        has_target = mask.any(dim=1) # [batch]

        # get prediction at the last non-padding position for each sample in the batch
        positions = torch.arange(labels.size(1), device=labels.device).unsqueeze(0)
        last_idx = torch.where(mask, positions, -1).max(dim=1).values

        pred_last = pred[has_target, last_idx[has_target]] # shape = torch.Size([B_valid])
        label_last = labels[has_target, last_idx[has_target]] # shape = torch.Size([B_valid])

        all_preds.append(pred_last.cpu())

    all_preds = torch.cat(all_preds, dim=0) # [num_samples]
    binary_preds = (all_preds == config.out_vocab.index('a')).long().tolist()

    RNN_predictions[split] = binary_preds


"""
# ---------------------------------------------------------------------
# Get DFA predictions.
# ---------------------------------------------------------------------

I don't need dataloaders because dataloaders return tensors. 

DFA_predictions = {
    <split>: [dfa.predict(seq_1), dfa.predict(seq_2), ..., dfa.predict(seq_8000)],
    ...
}

"""

print("\nGet DFA predictions:")

from dfa import DFA
dfa = DFA(dot_path=os.path.join(save_dir, f'{epoch}_dfa.dot'))

DFA_predictions = {}

for split, dataset in zip(["train", "val", "test"], [train_dataset, val_dataset, test_dataset]):
    all_preds = []
    
    for item in tqdm(dataset.data, desc=f"Run DFA on {split} samples"):
        # Two accommodations: remove eos_token_id from seq, and map them back to tokens rather than token_ids.
        seq = item['input']
        if getattr(dataset, "eos_token_id", None) is not None:
            seq = [token for token in seq if token != dataset.eos_token_id]
        seq = [config.in_vocab[i] for i in seq]

        pred = int(dfa.predict(seq))
        all_preds.append(pred)
    
    DFA_predictions[split] = all_preds


"""
Compare faithfulness (exact match) between RNN_predictions and DFA_predictions for each split.

faithfulness = {
    <split>: num_matches / 8000,
    ...
}

Append results to the same file (os.path.join(save_dir, f'{epoch}_dfa.dot'))
Division line: "=" * 80
"""

print("\nCompare RNN and DFA predictions:")
faithfulness = {}
for split in ["train", "val", "test"]:
    rnn_preds = RNN_predictions[split]
    dfa_preds = DFA_predictions[split]

    assert len(rnn_preds) == len(dfa_preds) == 8000, f"Expected 8000 samples for {split}, got {len(rnn_preds)} RNN preds and {len(dfa_preds)} DFA preds"

    matches = sum(r == d for r, d in zip(rnn_preds, dfa_preds))
    faithfulness[split] = matches / len(rnn_preds)

    print(f"{split}: {matches} / {len(rnn_preds)} = {faithfulness[split]:.4f}")

with open(os.path.join(save_dir, f'{epoch}_dfa.dot'), "a") as f:
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("Faithfulness (exact match between RNN and DFA predictions):\n")
    for split in ["train", "val", "test"]:
        f.write(f"{split}: {faithfulness[split]:.4f}\n")

