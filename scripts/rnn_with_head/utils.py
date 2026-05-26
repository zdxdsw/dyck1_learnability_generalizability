import torch, json
import numpy as np
import torch.nn.functional as F

def sequences_collator(batched_items, in_vocab, max_seq_len):
    in_v2i = {v: i for i, v in enumerate(in_vocab)}
    pad_token_id = in_v2i.get('<pad>')
    input_ids = []
    labels = {}
    for x in batched_items:
        input_id = x["input"]
        input_id += [pad_token_id] * (max_seq_len - len(input_id))
        input_ids.append(input_id)

        for k in [i for i in x if "labels" in i]:
            label = x[k]
            label += [-1] * (max_seq_len - len(label))
            if k not in labels: labels[k] = []
            labels[k].append(label)
            if None in label:
                print("Warning: found None in label")
                print(x["input"])
                print(label)
    result = {'input_ids': torch.LongTensor(input_ids)}
    result.update({k: torch.LongTensor(v) for k, v in labels.items()})
    return result

def get_acc(logits, labels, ignore_index):
    pred = logits.argmax(dim=-1)

    correct, demo = 0, 0

    for _pred, _labels in zip(pred, labels):
        pred = _pred[_labels != ignore_index]
        label = _labels[_labels != ignore_index]
        correct += (pred == label).float().sum().item()
        demo += label.numel()

    return correct, demo 


def inference(model_to_eval, dataloader, device, config):
    
    correct, demo = 0, 0
    loss, num_samples = 0, 0
    more_stats = {}
    for supervision_tag in ["S2", "S3"]:
        if supervision_tag in config.task:
            more_stats[supervision_tag] = {
                "loss": 0,
                "correct": 0,
                "demo": 0,
            }

    # testing_output = {}
    # k = 0
    for i, batch in enumerate(dataloader):
        forward_result = model_to_eval(batch['input_ids'].to(device),)
        
        logits = forward_result['logits']
        ce = F.cross_entropy(
            logits.view(-1, logits.size(-1)), # bs*seq_len, vocab_size
            batch['labels'].to(device).view(-1), # 1, bs*seq_len
            ignore_index=-1,
            reduction="sum",
        )
        loss += ce.detach().item()
        num_samples += batch['input_ids'].size(0) # batch_size
        
        _correct, _demo = get_acc(
            logits.detach().cpu(), 
            batch['labels'].detach().cpu(), 
            ignore_index=-1,
        )
        correct += _correct
        demo += _demo

        for supervision_tag in ["S2", "S3"]:
            if supervision_tag in config.task:
                supervision_logits = forward_result.get(f"{supervision_tag}_logits")
                supervision_ce = F.cross_entropy(
                    supervision_logits.view(-1, supervision_logits.size(-1)),
                    batch[f"{supervision_tag}_labels"].to(device).view(-1),
                    ignore_index=-1,
                    reduction="sum",
                )
                more_stats[supervision_tag]["loss"] += supervision_ce.detach().item()

                _correct, _demo = get_acc(
                    supervision_logits.detach().cpu(),
                    batch[f"{supervision_tag}_labels"].to(device).detach().cpu(),
                    ignore_index=-1,
                )
                more_stats[supervision_tag]["correct"] += _correct
                more_stats[supervision_tag]["demo"] += _demo
        
        # for input_id, gth_id, pred_id in zip(batch['input_ids'], batch['labels'], logits.argmax(dim=-1)):
        #     input_seq = [vocab[i] for i in input_id if vocab[i]!='<pad>']
        #     gth_seq = [vocab[gth_id[i]] for i in range(len(gth_id)) if gth_id[i]!=-1]
        #     pred_seq = [vocab[pred_id[i]] for i in range(len(gth_id)) if gth_id[i]!=-1][:len(gth_seq)]
        #     testing_output[k] = {
        #         "input": " ".join(input_seq),
        #         "gth": " ".join(gth_seq),
        #         "pred": " ".join(pred_seq),
        #     }
        #     k+=1

    # float numbers to 2 decimal places
    metrics = {"num_samples": num_samples,}
    for supervision_tag in ["S2", "S3"]:
        if supervision_tag in config.task:
            metrics.update({
                f"{supervision_tag}_loss": round(more_stats[supervision_tag]["loss"] / more_stats[supervision_tag]["demo"], 2) if more_stats[supervision_tag]["demo"] > 0 else -1,
                f"{supervision_tag}_acc": round(more_stats[supervision_tag]["correct"] / more_stats[supervision_tag]["demo"], 2) if more_stats[supervision_tag]["demo"] > 0 else -1,
                f"{supervision_tag}_numel": more_stats[supervision_tag]["demo"],
            })
    metrics.update({
        # "demo": demo, # sanity check
        "loss": round(loss / num_samples, 2),
        "acc": round(correct / demo, 2) if demo > 0 else -1,
    })
    
    return metrics