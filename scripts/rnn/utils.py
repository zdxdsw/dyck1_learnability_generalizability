import torch, json
import numpy as np
import torch.nn.functional as F

def sequences_collator(batched_items, in_vocab, max_seq_len):
    in_v2i = {v: i for i, v in enumerate(in_vocab)}
    pad_token_id = in_v2i.get('<pad>')
    input_ids = []
    labels = []
    for x in batched_items:
        input_id = x[0]
        input_id += [pad_token_id] * (max_seq_len - len(input_id))

        label = x[1]
        label += [-1] * (max_seq_len - len(label))

        input_ids.append(input_id)
        labels.append(label)
        
    return {
        'input_id': torch.LongTensor(input_ids),
        'label': torch.LongTensor(labels),
    }

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
    # testing_output = {}
    # k = 0
    for i, batch in enumerate(dataloader):
        logits = model_to_eval(batch['input_id'].to(device),)
        batch['label'] = batch['label'].to(device)
        
        ce = F.cross_entropy(
            logits.view(-1, logits.size(-1)), # bs*seq_len, vocab_size
            batch['label'].view(-1), # 1, bs*seq_len
            ignore_index=-1,
            reduction="sum",
        )
        loss += ce.detach().item()
        num_samples += batch['input_id'].size(0) # batch_size
        
        _correct, _demo = get_acc(
            logits.detach().cpu(), 
            batch['label'].detach().cpu(), 
            ignore_index=-1,
        )
        correct += _correct
        demo += _demo

        # for input_id, gth_id, pred_id in zip(batch['input_id'], batch['label'], logits.argmax(dim=-1)):
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
    metrics = {
        "num_samples": num_samples,
        "loss": round(loss / num_samples, 2),
        "acc": round(correct / demo, 2) if demo > 0 else -1,
    }

    return metrics