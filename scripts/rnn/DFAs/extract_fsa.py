from aalpy.learning_algs import run_Lstar
from aalpy.oracles import StatePrefixEqOracle
from aalpy.utils import save_automaton_to_file, visualize_automaton

import os, sys, pytz, json, glob
from datetime import datetime
timezone = pytz.timezone('America/New_York') 
import torch

# os.chdir("../../extraction") # don't use this. This makes relative paths fragile.
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

from model import RNNModel
from config import *
from config_taskspecific import *

from DataProcessing import parse_data, preprocess_binary_classification_data
from RNN_SULs import RnnBinarySUL
from TrainAndExtract import train_RNN_on_tomita_grammar, train_and_extract_bp, train_RNN_and_extract_FSM


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
        # in step(), when letter = None, treat is as '<eos>' if 'eos' in in_vocab, otherwise treat it as '<pad>'
        # None_id is the idx in enumerate(self.config.in_vocab) corresponding to '<eos>' if it exists, otherwise the idx corresponding to '<pad>'
        self.None_id = -1
        for i, tok in enumerate(self.config.in_vocab):
            if tok == '<eos>':
                self.None_id = i
                break
            elif tok == '<pad>':
                self.None_id = i
        assert self.None_id != -1, "Neither <eos> nor <pad> found in in_vocab"

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
        if self.state is None:
            self.state = self.initial_state()
        if letter is None: token_id = self.None_id
        else: token_id = self.token_dict[letter]

        x = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        x = self.torch_model.encoder(x)              # [1, 1, hidden_size]
        out, self.state = self.torch_model.model(x, self.state)
        logits = self.torch_model.decoder(out)       # [1, 1, out_vocab_size]

        logits = logits[:, self.output_position, :]  # [1, out_vocab_size]
        pred = torch.argmax(logits, dim=-1)          # [1]

        return int(pred.item())

ckpt_dir = "/data/yingshac/tmlr/ckpts"
task_suite = "S1_d_trulyleq[2-8]_A"
output_dir = f"/home/yingshac/workspace/tmlr/scripts/rnn/output"
output_dir = os.path.join(output_dir, task_suite)
task = "S1_d_trulyleq4"
handle = "0526_222518"
epoch = None
""" ckpt paths look like this:
    /data/yingshac/tmlr/ckpts/<task>/<handle>/<epoch>_<step>_RNN.py
    load the most recent epoch (i.e. largest epoch) if epoch is None, otherwise load the specified epoch
    
    config is initialized from `config = eval(f"{args.task}_Config")()`
    
    then load config from training logs: 
    /home/yingshac/workspace/tmlr/scripts/rnn/output/<task_suite>/<task>/<handle>/config.json
"""
config = eval(f"{task}_Config")()
saved_config_path = os.path.join(output_dir, task, handle, "config.json")
assert os.path.exists(saved_config_path), f"Config file not found at {saved_config_path}"
tmp = json.load(open(saved_config_path, "r"))
for k, v in tmp.items(): setattr(config, k, v)

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
rnn = TorchRNNAdapter(model, device="cuda")
sul = RnnBinarySUL(rnn)

state_eq_oracle = StatePrefixEqOracle(rnn.alphabet, sul, walks_per_state=500, walk_len=6)
dfa = run_Lstar(alphabet=rnn.alphabet, sul=sul, eq_oracle=state_eq_oracle, automaton_type='dfa',
                cache_and_non_det_check=True, max_learning_rounds=3)

save_dir = f'{task_suite}/{task}/{handle}'
os.makedirs(save_dir, exist_ok=True)
save_automaton_to_file(dfa, os.path.join(save_dir, f'{epoch}_dfa.dot'))