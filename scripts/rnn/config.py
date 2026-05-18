from dataclasses import dataclass

@dataclass
class Basic_Config:
    seed = [31, 42, 49, 57, 61, 77, 79, 89, 91]
    model = "RNN"
    task = ""
    date = ""
    hf_cache_dir = '/data/yingshac/hf_cache'
    num_hidden_layers = 1
    hidden_size = 64
    max_seq_len = 42
    num_epochs = 12
    dropout = 0.1
    per_device_train_batch_size = 32
    gradient_accumulation_steps = 1
    per_device_eval_batch_size = 64
    eval_accumulation_steps = 1
    logging_steps = 200 # frequency of updating progress bar
    learning_rate = 0.0001
    weight_decay = 0.0
    initializer_range = 0.02
    output_dir = "output/S1_d_[2-8]_D/"
    ckpt_dir = "/data/yingshac/tmlr/ckpts"
    load_from_dir = None #"0427_131257" # 
    init_from_ckpt = None
    shuffle_training_data = True

@dataclass
class Default_Config:
    date = "debug"
    dataset = ""
    in_vocab = []
    out_vocab = []
    train_files = []
    val_files = []
    test_files = []