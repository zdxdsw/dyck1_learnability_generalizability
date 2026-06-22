import torch, json
import torch.nn.functional as F

class PositionIdBuilder:
    def __init__(self, config):
        self.pe_aug_type = config.pe_aug_type
        self.max_position_embeddings = config.max_position_embeddings

    def __call__(self, seq_len: int) -> torch.LongTensor:
        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"seq_len={seq_len} exceeds "
                f"max_position_embeddings={self.max_position_embeddings}"
            )
        
        if self.pe_aug_type == "none":
            return torch.arange(seq_len, dtype=torch.long)
        elif self.pe_aug_type == "shifted":
            # torch.randint is exclusive of the upper bound.
            offset = torch.randint(0, self.max_position_embeddings - seq_len + 1, (1,)).item()
            return torch.arange(offset, offset + seq_len, dtype=torch.long)
        elif self.pe_aug_type == "randomized":
            # torch.randperm(n) randomly permutes integers from 0 to n - 1
            return torch.randperm(self.max_position_embeddings)[:seq_len].sort().values
        else:
            raise ValueError(f"Unknown pe_aug_type: {self.pe_aug_type}")

class BaseSequenceCollator:
    def __init__(self, config):
        self.in_vocab = config.in_vocab
        self.max_seq_len = config.max_seq_len
        self.ignore_idx = getattr(config, "ignore_index", -1)
        self.in_v2i = {v: i for i, v in enumerate(self.in_vocab)}
        if '<pad>' not in self.in_v2i:
            raise ValueError("'<pad>' must be in config.in_vocab")
        self.pad_token_id = self.in_v2i.get('<pad>')

    def __call__(self, batched_items) -> dict[str, torch.LongTensor]:
        input_ids = []
        input_lengths = []
        labels = {}

        for b in batched_items:
            x = list(b["input"])
            input_lengths.append(len(x))
            x = x + [self.pad_token_id] * (self.max_seq_len - len(x))
            input_ids.append(x)

            for k, v in b.items():
                if "labels" not in k: continue
                y = list(v)
                y = y + [self.ignore_idx] * (self.max_seq_len - len(y))
                if k not in labels: labels[k] = []
                labels[k].append(y)
        
        result = {'input_ids': torch.LongTensor(input_ids), 'input_lengths': torch.LongTensor(input_lengths)}
        result.update({k: torch.LongTensor(v) for k, v in labels.items()})
        return result

class TransformerSequenceCollator(BaseSequenceCollator):
    def __init__(self, config):
        super().__init__(config)
        self.position_id_builder = PositionIdBuilder(config)

    def __call__(self, batched_items) -> dict[str, torch.LongTensor]:
        result = super().__call__(batched_items)

        binary_padding_mask = torch.zeros(
            result['input_ids'].shape, # batch_size, seq_len
            dtype=torch.long,
        )

        position_ids = torch.zeros(
            result['input_ids'].shape, # batch_size, seq_len
            dtype=torch.long,
        )

        for i, pre_padding_len_tensor in enumerate(result['input_lengths']):
            pre_padding_len = int(pre_padding_len_tensor.item())
            binary_padding_mask[i, :pre_padding_len] = 1

            # Padding positions stay at 0. They should be ignored through binary_padding_mask.
            position_ids[i, :pre_padding_len] = self.position_id_builder(pre_padding_len)

        result['binary_mask'] = binary_padding_mask
        result['position_ids'] = position_ids
        return result    

def sequence_collator_builder(config):
    """
    For RNN, batch contains:
    {
        "input_ids": LongTensor[B, T],
        "lengths": LongTensor[B],
        "labels": LongTensor[B, T],
        "labels_S2": LongTensor[B, T],  # if present
        "labels_S3": LongTensor[B, T],  # if present
    }

    For Transformer, batch contains the above plus:
    {
        "binary_mask": LongTensor[B, T], # 1 for non-padding positions, 0 for padding positions, 
        "position_ids": LongTensor[B, T],
    }
    """
    
    if config.model == "RNN":
        return BaseSequenceCollator(config)
    elif config.model == "Transformer":
        return TransformerSequenceCollator(config)
    else:
        raise ValueError(f"Haven't implemented collator for model type: {config.model}")

def get_acc(logits, labels, ignore_index, binary_label=True):
    """
        logits: [batch, seq_len, vocab_size]
        labels: [batch, seq_len]
        labels use (if binary_label=True):
            0 = accept / 'a'
            1 = reject / 'r'

        otherwise, don't compute TP, FP, TN, FN
    """
    pred = logits.argmax(dim=-1) # [batch, seq_len]

    mask = labels != ignore_index
    pred = pred[mask]
    label = labels[mask]

    correct = (pred == label).sum()
    demo = label.numel()

    if not binary_label: 
        return (
            correct.item(),
            demo,
            None,
            None,
            None,
            None,
        )
    """ Convention:
        TP: pred 0, label 0
        FP: pred 0, label 1
        FN: pred 1, label 0
        TN: pred 1, label 1
    """
    TP = ((pred == 0) & (label == 0)).sum()
    FP = ((pred == 0) & (label == 1)).sum()
    FN = ((pred == 1) & (label == 0)).sum()
    TN = ((pred == 1) & (label == 1)).sum()

    return (
        correct.item(),
        demo,
        FP.item(),
        TP.item(),
        FN.item(),
        TN.item(),
    )


""" Prepare model-specific batched input since different model.forward() require different input keys """

MODEL_INPUT_KEYS = {
    "RNN": {
        "input_ids",
    },

    "Transformer": {
        "input_ids",
        "position_ids",
        "binary_padding_mask",
        "expose_attn_weights",
    },
}
def prepare_forward_inputs(batch, config, device):
    accepted_input_keys = MODEL_INPUT_KEYS[config.model]

    result = {}
    for k in batch:
        if k in accepted_input_keys:
            v = batch[k]
            if torch.is_tensor(v):
                v = v.to(device)
            result[k] = v
    return result

def inference(model_to_eval, dataloader, device, config):
    
    correct, demo = 0, 0
    FP, TP, FN, TN = 0, 0, 0, 0
    loss, num_samples = 0, 0
    more_stats = {}
    for supervision_tag in ["S2", "S3"]:
        # for S3, FP, TP, FN, TN don't apply
        if supervision_tag in config.task:
            more_stats[supervision_tag] = {
                "loss": 0,
                "correct": 0,
                "demo": 0,
            }
            if supervision_tag == "S2":
                more_stats[supervision_tag].update({
                    "FP": 0,
                    "TP": 0,
                    "FN": 0,
                    "TN": 0,
                })

    # k = 0
    for i, batch in enumerate(dataloader):
        forward_result = model_to_eval(
            **prepare_forward_inputs(
                batch=batch, 
                config=config, 
                device=device
            )
        )
        
        logits = forward_result['logits']
        ce = F.cross_entropy(
            logits.view(-1, logits.size(-1)), # bs*seq_len, vocab_size
            batch['labels'].to(device).view(-1), # 1, bs*seq_len
            ignore_index=getattr(config, "ignore_index", -1),
            reduction="sum",
        )
        loss += ce.detach().item()
        num_samples += batch['input_ids'].size(0) # batch_size
        
        _correct, _demo, _FP, _TP, _FN, _TN = get_acc(
            logits.detach().cpu(), 
            batch['labels'].detach().cpu(), 
            ignore_index=getattr(config, "ignore_index", -1),
        )
        correct += _correct
        demo += _demo
        FP += _FP
        TP += _TP
        FN += _FN
        TN += _TN

        for supervision_tag in ["S2", "S3"]:
            if supervision_tag in config.task:
                binary_label = supervision_tag == "S2"
                supervision_logits = forward_result.get(f"{supervision_tag}_logits")
                supervision_ce = F.cross_entropy(
                    supervision_logits.view(-1, supervision_logits.size(-1)),
                    batch[f"{supervision_tag}_labels"].to(device).view(-1),
                    ignore_index=getattr(config, "ignore_index", -1),
                    reduction="sum",
                )
                more_stats[supervision_tag]["loss"] += supervision_ce.detach().item()

                _correct, _demo, _FP, _TP, _FN, _TN = get_acc(
                    supervision_logits.detach().cpu(),
                    batch[f"{supervision_tag}_labels"].to(device).detach().cpu(),
                    ignore_index=getattr(config, "ignore_index", -1),
                    binary_label=binary_label,
                )

                more_stats[supervision_tag]["correct"] += _correct
                more_stats[supervision_tag]["demo"] += _demo

                if binary_label:
                    more_stats[supervision_tag]["FP"] += _FP
                    more_stats[supervision_tag]["TP"] += _TP
                    more_stats[supervision_tag]["FN"] += _FN
                    more_stats[supervision_tag]["TN"] += _TN

    # float numbers to 2 decimal places
    metrics = {"num_samples": num_samples,}
    for supervision_tag in ["S2", "S3"]:
        if supervision_tag in config.task:
            binary_label = supervision_tag == "S2"
            metrics.update({
                f"{supervision_tag}_loss": round(more_stats[supervision_tag]["loss"] / more_stats[supervision_tag]["demo"], 2) if more_stats[supervision_tag]["demo"] > 0 else -1,
                f"{supervision_tag}_acc": round(more_stats[supervision_tag]["correct"] / more_stats[supervision_tag]["demo"], 2) if more_stats[supervision_tag]["demo"] > 0 else -1,
                f"{supervision_tag}_numel": more_stats[supervision_tag]["demo"],
            })
            if binary_label:
                assert more_stats[supervision_tag]["correct"] + more_stats[supervision_tag]["FP"] + more_stats[supervision_tag]["FN"] == more_stats[supervision_tag]["demo"], f"Failed sanity check for {supervision_tag}: correct + FP + FN == demo; correct={more_stats[supervision_tag]['correct']}, FP={more_stats[supervision_tag]['FP']}, FN={more_stats[supervision_tag]['FN']}, demo={more_stats[supervision_tag]['demo']}"
                assert more_stats[supervision_tag]["TP"] + more_stats[supervision_tag]["TN"] == more_stats[supervision_tag]["correct"], f"Failed sanity check for {supervision_tag}: TP + TN == correct; TP={more_stats[supervision_tag]['TP']}, TN={more_stats[supervision_tag]['TN']}, correct={more_stats[supervision_tag]['correct']}"
                metrics.update({
                    f"{supervision_tag}_FP": more_stats[supervision_tag]["FP"],
                    f"{supervision_tag}_TP": more_stats[supervision_tag]["TP"],
                    f"{supervision_tag}_FN": more_stats[supervision_tag]["FN"],
                    f"{supervision_tag}_TN": more_stats[supervision_tag]["TN"],
                })
    
    assert correct + FP + FN == demo, f"Failed sanity check: correct + FP + FN == demo; correct={correct}, FP={FP}, FN={FN}, demo={demo}"
    assert TP + TN == correct, f"Failed sanity check: TP + TN == correct; TP={TP}, TN={TN}, correct={correct}"
    metrics.update({
        # "demo": demo, # sanity check
        "FP": round(FP / demo, 2) if demo > 0 else -1,
        "TP": round(TP / demo, 2) if demo > 0 else -1,
        "FN": round(FN / demo, 2) if demo > 0 else -1,
        "TN": round(TN / demo, 2) if demo > 0 else -1,
        "loss": round(loss / num_samples, 2),
        "acc": round(correct / demo, 2) if demo > 0 else -1,
    })
    
    return metrics

import os

""" Prepare output_dir and ckpt_dir with suffixes that denote model name """
def add_suffix_to_outermost_dir(path: str, suffix: str) -> str:
    """
    Add suffix to the first directory component.

    Example:
        output/S1a_d_[2-8]_H/ + _rnn
        -> output_rnn/S1a_d_[2-8]_H/
    """
    parts = path.split(os.sep)

    # Preserve absolute path leading slash if present.
    if path.startswith(os.sep):
        # Example: /a/b/c -> ["", "a", "b", "c"]
        if len(parts) < 2 or parts[1] == "":
            raise ValueError(f"Cannot add suffix to path: {path}")
        parts[1] = parts[1] + suffix
    else:
        # Example: output/foo -> ["output", "foo"]
        if len(parts) < 1 or parts[0] == "":
            raise ValueError(f"Cannot add suffix to path: {path}")
        parts[0] = parts[0] + suffix

    return os.sep.join(parts)


def add_suffix_to_innermost_dir(path: str, suffix: str) -> str:
    """
    Add suffix to the final directory component.

    Example:
        /data/yingshac/tmlr/ckpts + _rnn
        -> /data/yingshac/tmlr/ckpts_rnn
    """
    path = path.rstrip(os.sep)
    parent, name = os.path.split(path)

    if not name:
        raise ValueError(f"Cannot add suffix to path: {path}")

    return os.path.join(parent, name + suffix)
