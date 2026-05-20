import random, torch, json
from torch.utils.data import Dataset

class S1Dataset(Dataset):

    def __init__(self, files, in_vocab, out_vocab, name="", specific_length=None):
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
        self.name = name
        self.in_vocab = in_vocab
        self.in_v2i = {v: i for i, v in enumerate(in_vocab)}
        self.eos_token_id = self.in_v2i.get("<eos>")

        self.out_vocab = out_vocab
        self.out_v2i = {v: i for i, v in enumerate(out_vocab)}

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
                    list(map(self.in_v2i.get, s)) + [self.eos_token_id], 
                     [-1]*len(s) + [self.out_v2i.get("a")]
                ))
            for s in neg_samples:
                self.data.append((
                    list(map(self.in_v2i.get, s)) + [self.eos_token_id], 
                     [-1]*len(s) + [self.out_v2i.get("r")]
                ))
        # shuffle
        random.shuffle(self.data)
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class S1a_Dataset(Dataset): # no eos; supervised labels for the last token

    def __init__(self, files, in_vocab, out_vocab, name="", specific_length=None):
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
        self.name = name
        self.in_vocab = in_vocab
        self.in_v2i = {v: i for i, v in enumerate(in_vocab)}

        self.out_vocab = out_vocab
        self.out_v2i = {v: i for i, v in enumerate(out_vocab)}

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
                    list(map(self.in_v2i.get, s)), 
                     [-1]*(len(s) - 1) + [self.out_v2i.get("a")]
                ))
            for s in neg_samples:
                self.data.append((
                    list(map(self.in_v2i.get, s)), 
                     [-1]*(len(s) - 1) + [self.out_v2i.get("r")]
                ))
        # shuffle
        random.shuffle(self.data)
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]