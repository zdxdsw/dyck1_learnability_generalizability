import random, torch, json
from torch.utils.data import Dataset

class L1Dataset(Dataset):

    def __init__(self, files, vocab, specific_length=None):
        """ files is a list of paths, each (<*_data_dir>/fix_d_{d}.json) is a json file with the format:
        {
            "pos": {
                "l": [s1, s2, ...],
                ...        },
            "neg": {
                "l": [s1, s2, ...],
                ...        }
        }
        """
        self.vocab = vocab
        self.i2v = {i: v for i, v in enumerate(vocab)}
        self.v2i = {v: i for i, v in enumerate(vocab)}
        self.eos_token_id = vocab["<eos>"]

        # process all json files, create a list of [([token_ids], [labels]), ...]
        self.data = []
        for file in files:
            with open(file, "r") as f:
                J = json.load(f)
            if specific_length is not None:
                pos_samples = J["pos"].get(str(specific_length), [])
                neg_samples = J["neg"].get(str(specific_length), [])
            else:
                pos_samples = [s for l in J["pos"] for s in J["pos"][l]]
                neg_samples = [s for l in J["neg"] for s in J["neg"][l]]
            for s in pos_samples:
                self.data.append((
                    list(map(self.v2i.get, s)) + [self.eos_token_id], 
                     [-1]*len(s) + [1]
                ))
            for s in neg_samples:
                self.data.append((
                    list(map(self.v2i.get, s)) + [self.eos_token_id], 
                     [-1]*len(s) + [0]
                ))
            # shuffle
            random.shuffle(self.data)
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]