import random, torch, json
from torch.utils.data import Dataset

class S1Dataset(Dataset):

    def __init__(self, files, config, name="", specific_length=None):
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
        self.in_vocab = config.in_vocab
        self.in_v2i = {v: i for i, v in enumerate(self.in_vocab)}
        self.eos_token_id = self.in_v2i.get("<eos>")

        self.out_vocab = config.out_vocab
        self.out_v2i = {v: i for i, v in enumerate(self.out_vocab)}

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
                self.data.append(({
                    "input": list(map(self.in_v2i.get, s)) + [self.eos_token_id],
                    "labels": [-1]*len(s) + [self.out_v2i.get("a")]
                }))
            for s in neg_samples:
                self.data.append(({
                    "input": list(map(self.in_v2i.get, s)) + [self.eos_token_id],
                    "labels": [-1]*len(s) + [self.out_v2i.get("r")]
                }))
        # shuffle
        random.shuffle(self.data)
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class S1a_Dataset(Dataset): # no eos; supervised a/r labels for the last token

    def __init__(self, files, config, name="", specific_length=None):
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
        self.in_vocab = config.in_vocab
        self.in_v2i = {v: i for i, v in enumerate(self.in_vocab)}

        self.out_vocab = config.out_vocab
        self.out_v2i = {v: i for i, v in enumerate(self.out_vocab)}

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
                self.data.append(({
                    "input": list(map(self.in_v2i.get, s)),
                    "labels": [-1]*(len(s) - 1) + [self.out_v2i.get("a")]
                }))
            for s in neg_samples:
                self.data.append(({
                    "input": list(map(self.in_v2i.get, s)),
                    "labels": [-1]*(len(s) - 1) + [self.out_v2i.get("r")]
                }))
        # shuffle
        random.shuffle(self.data)
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_S2_idx(s):
    """
    Return the earliest index of s where current_depth = -1
    Return len(s) if s is positive or if s is negative but current_depth never reaches -1
    """
    current_depth = 0
    result = len(s)
    for i, ch in enumerate(s):
        if ch == '(':
            current_depth += 1
        elif ch == ')':
            current_depth -= 1
            if current_depth < 0:
                result = i
                return result
    return result

class S2Dataset(Dataset): 
    # 2nd head labels: 1 until the first symbol that implies rejection, 0 until the end.

    def __init__(self, files, config, name="", specific_length=None):
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
        self.in_vocab = config.in_vocab
        self.in_v2i = {v: i for i, v in enumerate(self.in_vocab)}
        self.eos_token_id = self.in_v2i.get("<eos>")

        self.out_vocab = config.out_vocab
        self.out_v2i = {v: i for i, v in enumerate(self.out_vocab)}

        self.out_S2_vocab = config.out_S2_vocab
        self.out_S2_v2i = {v: i for i, v in enumerate(self.out_S2_vocab)}

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
                self.data.append(({
                    "input": list(map(self.in_v2i.get, s)) + [self.eos_token_id], 
                    "labels": [-1]*len(s) + [self.out_v2i.get("a")],
                    "S2_labels": [self.out_S2_v2i.get("1")]*len(s),
                }))
            for s in neg_samples:
                S2_idx = get_S2_idx(s)
                S2_labels = [self.out_S2_v2i.get("1")]*S2_idx + [self.out_S2_v2i.get("0")]*(len(s) - S2_idx)
                self.data.append(({
                    "input": list(map(self.in_v2i.get, s)) + [self.eos_token_id],  
                    "labels": [-1]*len(s) + [self.out_v2i.get("r")], 
                    "S2_labels": S2_labels,
                }))
        # shuffle
        random.shuffle(self.data)
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class S2a_Dataset(Dataset): # no eos; supervised a/r labels for the last token
    # 2nd head labels: 1 until the first symbol that implies rejection, 0 until the end.

    def __init__(self, files, config, name="", specific_length=None):
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
        self.in_vocab = config.in_vocab
        self.in_v2i = {v: i for i, v in enumerate(self.in_vocab)}

        self.out_vocab = config.out_vocab
        self.out_v2i = {v: i for i, v in enumerate(self.out_vocab)}

        self.out_S2_vocab = config.out_S2_vocab
        self.out_S2_v2i = {v: i for i, v in enumerate(self.out_S2_vocab)}

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
                self.data.append(({
                    "input": list(map(self.in_v2i.get, s)), 
                    "labels": [-1]*(len(s) - 1) + [self.out_v2i.get("a")],
                    "S2_labels": [self.out_S2_v2i.get("1")]*len(s),
                }))
            for s in neg_samples:
                S2_idx = get_S2_idx(s)
                S2_labels = [self.out_S2_v2i.get("1")]*S2_idx + [self.out_S2_v2i.get("0")]*(len(s) - S2_idx)
                self.data.append(({
                    "input": list(map(self.in_v2i.get, s)), 
                    "labels": [-1]*(len(s) - 1) + [self.out_v2i.get("r")],
                    "S2_labels": S2_labels,
                }))
        # shuffle
        random.shuffle(self.data)
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def get_S3_labels(s):
    """
    Supervise the current_depth on every symbol of s.
    Use symbols: "d0", "d1", "d2", ...
    Negative depth: "d-1".
    """
    current_depth = 0
    result = []
    for i, ch in enumerate(s):
        if ch == '(':
            current_depth += 1
        elif ch == ')':
            current_depth -= 1
        S3_label = f"d{current_depth}" if current_depth >= 0 else "d-1"
        result.append(S3_label)
    return result

class S3Dataset(Dataset): 
    # Supervise current_depth on every symbol of s

    def __init__(self, files, config, name="", specific_length=None):
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
        self.in_vocab = config.in_vocab
        self.in_v2i = {v: i for i, v in enumerate(self.in_vocab)}
        self.eos_token_id = self.in_v2i.get("<eos>")

        self.out_vocab = config.out_vocab
        self.out_v2i = {v: i for i, v in enumerate(self.out_vocab)}

        self.out_S3_vocab = config.out_S3_vocab
        self.out_S3_v2i = {v: i for i, v in enumerate(self.out_S3_vocab)}

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
                S3_labels = list(map(self.out_S3_v2i.get, get_S3_labels(s))) + [-1] # don't supervise depth on eos
                self.data.append(({
                    "input": list(map(self.in_v2i.get, s)) + [self.eos_token_id], 
                    "labels": [-1]*len(s) + [self.out_v2i.get("a")],
                    "S3_labels": S3_labels,
                }))
            for s in neg_samples:
                S3_labels = list(map(self.out_S3_v2i.get, get_S3_labels(s))) + [-1] # don't supervise depth on eos
                self.data.append(({
                    "input": list(map(self.in_v2i.get, s)) + [self.eos_token_id],  
                    "labels": [-1]*len(s) + [self.out_v2i.get("r")], 
                    "S3_labels": S3_labels,
                }))
        # shuffle
        random.shuffle(self.data)
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class S3a_Dataset(Dataset): # no eos; supervised a/r labels for the last token
    # Supervise current_depth on every symbol of s

    def __init__(self, files, config, name="", specific_length=None):
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
        self.in_vocab = config.in_vocab
        self.in_v2i = {v: i for i, v in enumerate(self.in_vocab)}

        self.out_vocab = config.out_vocab
        self.out_v2i = {v: i for i, v in enumerate(self.out_vocab)}

        self.out_S3_vocab = config.out_S3_vocab
        self.out_S3_v2i = {v: i for i, v in enumerate(self.out_S3_vocab)}

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
                S3_labels = list(map(self.out_S3_v2i.get, get_S3_labels(s)))
                self.data.append(({
                    "input": list(map(self.in_v2i.get, s)), 
                    "labels": [-1]*(len(s) - 1) + [self.out_v2i.get("a")],
                    "S3_labels": S3_labels,
                }))
            for s in neg_samples:
                S3_labels = list(map(self.out_S3_v2i.get, get_S3_labels(s)))
                self.data.append(({
                    "input": list(map(self.in_v2i.get, s)), 
                    "labels": [-1]*(len(s) - 1) + [self.out_v2i.get("r")],
                    "S3_labels": S3_labels,
                }))
        # shuffle
        random.shuffle(self.data)
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
  