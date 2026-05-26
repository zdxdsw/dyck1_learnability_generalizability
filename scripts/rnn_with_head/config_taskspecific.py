from dataclasses import dataclass
from config import Basic_Config

@dataclass
class S1_d_2_Config(Basic_Config):
    dataset = "S1"
    in_vocab = ['(', ')', '<eos>','<pad>']
    out_vocab = ['a', 'r']
    train_files = ["/data/yingshac/tmlr/train/fix_d_2.json"]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_2.json", f"d_2_l_{l}", l) for l in range(41, 121)]

@dataclass
class S1_d_3_Config(Basic_Config):
    dataset = "S1"
    in_vocab = ['(', ')', '<eos>','<pad>']
    out_vocab = ['a', 'r']
    train_files = ["/data/yingshac/tmlr/train/fix_d_3.json"]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_3.json", f"d_3_l_{l}", l) for l in range(41, 121)]

@dataclass
class S1_d_4_Config(Basic_Config):
    dataset = "S1"
    in_vocab = ['(', ')', '<eos>','<pad>']
    out_vocab = ['a', 'r']
    train_files = ["/data/yingshac/tmlr/train/fix_d_4.json"]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_4.json", f"d_4_l_{l}", l) for l in range(41, 121)]

@dataclass
class S1_d_5_Config(Basic_Config):
    dataset = "S1"
    in_vocab = ['(', ')', '<eos>','<pad>']
    out_vocab = ['a', 'r']
    train_files = ["/data/yingshac/tmlr/train/fix_d_5.json"]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_5.json", f"d_5_l_{l}", l) for l in range(41, 121)]

@dataclass
class S1_d_6_Config(Basic_Config):
    dataset = "S1"
    in_vocab = ['(', ')', '<eos>','<pad>']
    out_vocab = ['a', 'r']
    train_files = ["/data/yingshac/tmlr/train/fix_d_6.json"]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_6.json", f"d_6_l_{l}", l) for l in range(41, 121)]

@dataclass
class S1_d_7_Config(Basic_Config):
    dataset = "S1"
    in_vocab = ['(', ')', '<eos>','<pad>']
    out_vocab = ['a', 'r']
    train_files = ["/data/yingshac/tmlr/train/fix_d_7.json"]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_7.json", f"d_7_l_{l}", l) for l in range(41, 121)]

@dataclass
class S1_d_8_Config(Basic_Config):
    dataset = "S1"
    in_vocab = ['(', ')', '<eos>','<pad>']
    out_vocab = ['a', 'r']
    train_files = ["/data/yingshac/tmlr/train/fix_d_8.json"]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_8.json", f"d_8_l_{l}", l) for l in range(41, 121)]

@dataclass
class S1_d_leq2_Config(Basic_Config):
    dataset = "S1"
    in_vocab = ['(', ')', '<eos>','<pad>']
    out_vocab = ['a', 'r']
    train_files = ["/data/yingshac/tmlr/train/fix_d_2.json"]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_2.json", f"d_2_l_{l}", l) for l in range(41, 121)]

@dataclass
class S1_d_leq3_Config(Basic_Config):
    dataset = "S1"
    in_vocab = ['(', ')', '<eos>','<pad>']
    out_vocab = ['a', 'r']
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_2.json",
        "/data/yingshac/tmlr/train/fix_d_3.json"
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_3.json", f"d_3_l_{l}", l) for l in range(41, 121)]

@dataclass
class S1_d_leq4_Config(Basic_Config):
    dataset = "S1"
    in_vocab = ['(', ')', '<eos>','<pad>']
    out_vocab = ['a', 'r']
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_2.json",
        "/data/yingshac/tmlr/train/fix_d_3.json",
        "/data/yingshac/tmlr/train/fix_d_4.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_4.json", f"d_4_l_{l}", l) for l in range(41, 121)]

@dataclass
class S1_d_leq5_Config(Basic_Config):
    dataset = "S1"
    in_vocab = ['(', ')', '<eos>','<pad>']
    out_vocab = ['a', 'r']
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_2.json",
        "/data/yingshac/tmlr/train/fix_d_3.json",
        "/data/yingshac/tmlr/train/fix_d_4.json",
        "/data/yingshac/tmlr/train/fix_d_5.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_5.json", f"d_5_l_{l}", l) for l in range(41, 121)]

@dataclass
class S1_d_leq6_Config(Basic_Config):
    dataset = "S1"
    in_vocab = ['(', ')', '<eos>','<pad>']
    out_vocab = ['a', 'r']
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_2.json",
        "/data/yingshac/tmlr/train/fix_d_3.json",
        "/data/yingshac/tmlr/train/fix_d_4.json",
        "/data/yingshac/tmlr/train/fix_d_5.json",
        "/data/yingshac/tmlr/train/fix_d_6.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_6.json", f"d_6_l_{l}", l) for l in range(41, 121)]

@dataclass
class S1_d_leq7_Config(Basic_Config):
    dataset = "S1"
    in_vocab = ['(', ')', '<eos>','<pad>']
    out_vocab = ['a', 'r']
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_2.json",
        "/data/yingshac/tmlr/train/fix_d_3.json",
        "/data/yingshac/tmlr/train/fix_d_4.json",
        "/data/yingshac/tmlr/train/fix_d_5.json",
        "/data/yingshac/tmlr/train/fix_d_6.json",
        "/data/yingshac/tmlr/train/fix_d_7.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_7.json", f"d_7_l_{l}", l) for l in range(41, 121)]

@dataclass
class S1_d_leq8_Config(Basic_Config):
    dataset = "S1"
    in_vocab = ['(', ')', '<eos>','<pad>']
    out_vocab = ['a', 'r']
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_2.json",
        "/data/yingshac/tmlr/train/fix_d_3.json",
        "/data/yingshac/tmlr/train/fix_d_4.json",
        "/data/yingshac/tmlr/train/fix_d_5.json",
        "/data/yingshac/tmlr/train/fix_d_6.json",
        "/data/yingshac/tmlr/train/fix_d_7.json",
        "/data/yingshac/tmlr/train/fix_d_8.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_8.json", f"d_8_l_{l}", l) for l in range(41, 121)]

### without eos; cumulative
@dataclass
class S1a_d_leq2_Config(Basic_Config):
    dataset = "S1a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    train_files = ["/data/yingshac/tmlr/train/fix_d_2.json"]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_2.json", f"d_2_l_{l}", l) for l in range(41, 121)]

@dataclass
class S1a_d_leq3_Config(Basic_Config):
    dataset = "S1a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_2.json",
        "/data/yingshac/tmlr/train/fix_d_3.json"
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_3.json", f"d_3_l_{l}", l) for l in range(41, 121)]

@dataclass
class S1a_d_leq4_Config(Basic_Config):
    dataset = "S1a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_2.json",
        "/data/yingshac/tmlr/train/fix_d_3.json",
        "/data/yingshac/tmlr/train/fix_d_4.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_4.json", f"d_4_l_{l}", l) for l in range(41, 121)]

@dataclass
class S1a_d_leq5_Config(Basic_Config):
    dataset = "S1a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_2.json",
        "/data/yingshac/tmlr/train/fix_d_3.json",
        "/data/yingshac/tmlr/train/fix_d_4.json",
        "/data/yingshac/tmlr/train/fix_d_5.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_5.json", f"d_5_l_{l}", l) for l in range(41, 121)]

@dataclass
class S1a_d_leq6_Config(Basic_Config):
    dataset = "S1a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_2.json",
        "/data/yingshac/tmlr/train/fix_d_3.json",
        "/data/yingshac/tmlr/train/fix_d_4.json",
        "/data/yingshac/tmlr/train/fix_d_5.json",
        "/data/yingshac/tmlr/train/fix_d_6.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_6.json", f"d_6_l_{l}", l) for l in range(41, 121)]

@dataclass
class S1a_d_leq7_Config(Basic_Config):
    dataset = "S1a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_2.json",
        "/data/yingshac/tmlr/train/fix_d_3.json",
        "/data/yingshac/tmlr/train/fix_d_4.json",
        "/data/yingshac/tmlr/train/fix_d_5.json",
        "/data/yingshac/tmlr/train/fix_d_6.json",
        "/data/yingshac/tmlr/train/fix_d_7.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_7.json", f"d_7_l_{l}", l) for l in range(41, 121)]

@dataclass
class S1a_d_leq8_Config(Basic_Config):
    dataset = "S1a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_2.json",
        "/data/yingshac/tmlr/train/fix_d_3.json",
        "/data/yingshac/tmlr/train/fix_d_4.json",
        "/data/yingshac/tmlr/train/fix_d_5.json",
        "/data/yingshac/tmlr/train/fix_d_6.json",
        "/data/yingshac/tmlr/train/fix_d_7.json",
        "/data/yingshac/tmlr/train/fix_d_8.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_8.json", f"d_8_l_{l}", l) for l in range(41, 121)]

### without eos; cumulative; add d=1 data!!
@dataclass
class S1a_d_trulyleq2_Config(Basic_Config):
    dataset = "S1a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_1.json",
        "/data/yingshac/tmlr/train/fix_d_2.json"
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_1.json", "d_1", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
        ("/data/yingshac/tmlr/validation/fix_d_9.json", "d_9", None),
        ("/data/yingshac/tmlr/validation/fix_d_10.json", "d_10", None),
        ("/data/yingshac/tmlr/validation/fix_d_11.json", "d_11", None),
        ("/data/yingshac/tmlr/validation/fix_d_12.json", "d_12", None),
        ("/data/yingshac/tmlr/validation/fix_d_13.json", "d_13", None),
        ("/data/yingshac/tmlr/validation/fix_d_14.json", "d_14", None),
        ("/data/yingshac/tmlr/validation/fix_d_15.json", "d_15", None),
        ("/data/yingshac/tmlr/validation/fix_d_16.json", "d_16", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_2.json", f"d_2_l_{l}", l) for l in range(41, 121)]

@dataclass
class S1a_d_trulyleq3_Config(Basic_Config):
    dataset = "S1a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_1.json",
        "/data/yingshac/tmlr/train/fix_d_2.json",
        "/data/yingshac/tmlr/train/fix_d_3.json"
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_1.json", "d_1", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
        ("/data/yingshac/tmlr/validation/fix_d_9.json", "d_9", None),
        ("/data/yingshac/tmlr/validation/fix_d_10.json", "d_10", None),
        ("/data/yingshac/tmlr/validation/fix_d_11.json", "d_11", None),
        ("/data/yingshac/tmlr/validation/fix_d_12.json", "d_12", None),
        ("/data/yingshac/tmlr/validation/fix_d_13.json", "d_13", None),
        ("/data/yingshac/tmlr/validation/fix_d_14.json", "d_14", None),
        ("/data/yingshac/tmlr/validation/fix_d_15.json", "d_15", None),
        ("/data/yingshac/tmlr/validation/fix_d_16.json", "d_16", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_3.json", f"d_3_l_{l}", l) for l in range(41, 121)]

@dataclass
class S1a_d_trulyleq4_Config(Basic_Config):
    dataset = "S1a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_1.json",
        "/data/yingshac/tmlr/train/fix_d_2.json",
        "/data/yingshac/tmlr/train/fix_d_3.json",
        "/data/yingshac/tmlr/train/fix_d_4.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_1.json", "d_1", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
        ("/data/yingshac/tmlr/validation/fix_d_9.json", "d_9", None),
        ("/data/yingshac/tmlr/validation/fix_d_10.json", "d_10", None),
        ("/data/yingshac/tmlr/validation/fix_d_11.json", "d_11", None),
        ("/data/yingshac/tmlr/validation/fix_d_12.json", "d_12", None),
        ("/data/yingshac/tmlr/validation/fix_d_13.json", "d_13", None),
        ("/data/yingshac/tmlr/validation/fix_d_14.json", "d_14", None),
        ("/data/yingshac/tmlr/validation/fix_d_15.json", "d_15", None),
        ("/data/yingshac/tmlr/validation/fix_d_16.json", "d_16", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_4.json", f"d_4_l_{l}", l) for l in range(41, 121)]

@dataclass
class S1a_d_trulyleq5_Config(Basic_Config):
    dataset = "S1a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_1.json",
        "/data/yingshac/tmlr/train/fix_d_2.json",
        "/data/yingshac/tmlr/train/fix_d_3.json",
        "/data/yingshac/tmlr/train/fix_d_4.json",
        "/data/yingshac/tmlr/train/fix_d_5.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_1.json", "d_1", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
        ("/data/yingshac/tmlr/validation/fix_d_9.json", "d_9", None),
        ("/data/yingshac/tmlr/validation/fix_d_10.json", "d_10", None),
        ("/data/yingshac/tmlr/validation/fix_d_11.json", "d_11", None),
        ("/data/yingshac/tmlr/validation/fix_d_12.json", "d_12", None),
        ("/data/yingshac/tmlr/validation/fix_d_13.json", "d_13", None),
        ("/data/yingshac/tmlr/validation/fix_d_14.json", "d_14", None),
        ("/data/yingshac/tmlr/validation/fix_d_15.json", "d_15", None),
        ("/data/yingshac/tmlr/validation/fix_d_16.json", "d_16", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_5.json", f"d_5_l_{l}", l) for l in range(41, 121)]

@dataclass
class S1a_d_trulyleq6_Config(Basic_Config):
    dataset = "S1a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_1.json",
        "/data/yingshac/tmlr/train/fix_d_2.json",
        "/data/yingshac/tmlr/train/fix_d_3.json",
        "/data/yingshac/tmlr/train/fix_d_4.json",
        "/data/yingshac/tmlr/train/fix_d_5.json",
        "/data/yingshac/tmlr/train/fix_d_6.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_1.json", "d_1", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
        ("/data/yingshac/tmlr/validation/fix_d_9.json", "d_9", None),
        ("/data/yingshac/tmlr/validation/fix_d_10.json", "d_10", None),
        ("/data/yingshac/tmlr/validation/fix_d_11.json", "d_11", None),
        ("/data/yingshac/tmlr/validation/fix_d_12.json", "d_12", None),
        ("/data/yingshac/tmlr/validation/fix_d_13.json", "d_13", None),
        ("/data/yingshac/tmlr/validation/fix_d_14.json", "d_14", None),
        ("/data/yingshac/tmlr/validation/fix_d_15.json", "d_15", None),
        ("/data/yingshac/tmlr/validation/fix_d_16.json", "d_16", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_6.json", f"d_6_l_{l}", l) for l in range(41, 121)]

@dataclass
class S1a_d_trulyleq7_Config(Basic_Config):
    dataset = "S1a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_1.json",
        "/data/yingshac/tmlr/train/fix_d_2.json",
        "/data/yingshac/tmlr/train/fix_d_3.json",
        "/data/yingshac/tmlr/train/fix_d_4.json",
        "/data/yingshac/tmlr/train/fix_d_5.json",
        "/data/yingshac/tmlr/train/fix_d_6.json",
        "/data/yingshac/tmlr/train/fix_d_7.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_1.json", "d_1", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
        ("/data/yingshac/tmlr/validation/fix_d_9.json", "d_9", None),
        ("/data/yingshac/tmlr/validation/fix_d_10.json", "d_10", None),
        ("/data/yingshac/tmlr/validation/fix_d_11.json", "d_11", None),
        ("/data/yingshac/tmlr/validation/fix_d_12.json", "d_12", None),
        ("/data/yingshac/tmlr/validation/fix_d_13.json", "d_13", None),
        ("/data/yingshac/tmlr/validation/fix_d_14.json", "d_14", None),
        ("/data/yingshac/tmlr/validation/fix_d_15.json", "d_15", None),
        ("/data/yingshac/tmlr/validation/fix_d_16.json", "d_16", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_7.json", f"d_7_l_{l}", l) for l in range(41, 121)]

@dataclass
class S1a_d_trulyleq8_Config(Basic_Config):
    dataset = "S1a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_1.json",
        "/data/yingshac/tmlr/train/fix_d_2.json",
        "/data/yingshac/tmlr/train/fix_d_3.json",
        "/data/yingshac/tmlr/train/fix_d_4.json",
        "/data/yingshac/tmlr/train/fix_d_5.json",
        "/data/yingshac/tmlr/train/fix_d_6.json",
        "/data/yingshac/tmlr/train/fix_d_7.json",
        "/data/yingshac/tmlr/train/fix_d_8.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
        ("/data/yingshac/tmlr/validation/fix_d_1.json", "d_1", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_9.json", "d_9", None),
        ("/data/yingshac/tmlr/validation/fix_d_10.json", "d_10", None),
        ("/data/yingshac/tmlr/validation/fix_d_11.json", "d_11", None),
        ("/data/yingshac/tmlr/validation/fix_d_12.json", "d_12", None),
        ("/data/yingshac/tmlr/validation/fix_d_13.json", "d_13", None),
        ("/data/yingshac/tmlr/validation/fix_d_14.json", "d_14", None),
        ("/data/yingshac/tmlr/validation/fix_d_15.json", "d_15", None),
        ("/data/yingshac/tmlr/validation/fix_d_16.json", "d_16", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_8.json", f"d_8_l_{l}", l) for l in range(41, 121)]

### without eos; cumulative; SHORTER
@dataclass
class shorterS1a_d_leq2_Config(Basic_Config):
    dataset = "S1a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    train_files = ["/data/yingshac/tmlr/shorter/train/fix_d_2.json"]
    val_files = [
        ("/data/yingshac/tmlr/shorter/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_6.json", "d_6", None),
    ]
    test_files = [("/data/yingshac/tmlr/shorter/test/fix_d_2.json", f"d_2_l_{l}", l) for l in range(21, 61)]

@dataclass
class shorterS1a_d_leq3_Config(Basic_Config):
    dataset = "S1a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    train_files = [
        "/data/yingshac/tmlr/shorter/train/fix_d_2.json",
        "/data/yingshac/tmlr/shorter/train/fix_d_3.json"
    ]
    val_files = [
        ("/data/yingshac/tmlr/shorter/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_6.json", "d_6", None),
    ]
    test_files = [("/data/yingshac/tmlr/shorter/test/fix_d_3.json", f"d_3_l_{l}", l) for l in range(21, 61)]

@dataclass
class shorterS1a_d_leq4_Config(Basic_Config):
    dataset = "S1a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    train_files = [
        "/data/yingshac/tmlr/shorter/train/fix_d_2.json",
        "/data/yingshac/tmlr/shorter/train/fix_d_3.json",
        "/data/yingshac/tmlr/shorter/train/fix_d_4.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/shorter/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_6.json", "d_6", None),
    ]
    test_files = [("/data/yingshac/tmlr/shorter/test/fix_d_4.json", f"d_4_l_{l}", l) for l in range(21, 61)]

@dataclass
class shorterS1a_d_leq5_Config(Basic_Config):
    dataset = "S1a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    train_files = [
        "/data/yingshac/tmlr/shorter/train/fix_d_2.json",
        "/data/yingshac/tmlr/shorter/train/fix_d_3.json",
        "/data/yingshac/tmlr/shorter/train/fix_d_4.json",
        "/data/yingshac/tmlr/shorter/train/fix_d_5.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/shorter/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_6.json", "d_6", None),
    ]
    test_files = [("/data/yingshac/tmlr/shorter/test/fix_d_5.json", f"d_5_l_{l}", l) for l in range(21, 61)]

@dataclass
class shorterS1a_d_leq6_Config(Basic_Config):
    dataset = "S1a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    train_files = [
        "/data/yingshac/tmlr/shorter/train/fix_d_2.json",
        "/data/yingshac/tmlr/shorter/train/fix_d_3.json",
        "/data/yingshac/tmlr/shorter/train/fix_d_4.json",
        "/data/yingshac/tmlr/shorter/train/fix_d_5.json",
        "/data/yingshac/tmlr/shorter/train/fix_d_6.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/shorter/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_5.json", "d_5", None),
    ]
    test_files = [("/data/yingshac/tmlr/shorter/test/fix_d_6.json", f"d_6_l_{l}", l) for l in range(21, 61)]

### without eos; cumulative; SHORTER; add d=1 data!!
@dataclass
class shorterS1a_d_trulyleq2_Config(Basic_Config):
    dataset = "S1a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    train_files = [
        "/data/yingshac/tmlr/shorter/train/fix_d_1.json",
        "/data/yingshac/tmlr/shorter/train/fix_d_2.json"
    ]
    val_files = [
        ("/data/yingshac/tmlr/shorter/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_1.json", "d_1", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_6.json", "d_6", None),
    ]
    test_files = [("/data/yingshac/tmlr/shorter/test/fix_d_2.json", f"d_2_l_{l}", l) for l in range(21, 61)]

@dataclass
class shorterS1a_d_trulyleq3_Config(Basic_Config):
    dataset = "S1a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    train_files = [
        "/data/yingshac/tmlr/shorter/train/fix_d_1.json",
        "/data/yingshac/tmlr/shorter/train/fix_d_2.json",
        "/data/yingshac/tmlr/shorter/train/fix_d_3.json"
    ]
    val_files = [
        ("/data/yingshac/tmlr/shorter/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_1.json", "d_1", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_6.json", "d_6", None),
    ]
    test_files = [("/data/yingshac/tmlr/shorter/test/fix_d_3.json", f"d_3_l_{l}", l) for l in range(21, 61)]

@dataclass
class shorterS1a_d_trulyleq4_Config(Basic_Config):
    dataset = "S1a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    train_files = [
        "/data/yingshac/tmlr/shorter/train/fix_d_1.json",
        "/data/yingshac/tmlr/shorter/train/fix_d_2.json",
        "/data/yingshac/tmlr/shorter/train/fix_d_3.json",
        "/data/yingshac/tmlr/shorter/train/fix_d_4.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/shorter/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_1.json", "d_1", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_6.json", "d_6", None),
    ]
    test_files = [("/data/yingshac/tmlr/shorter/test/fix_d_4.json", f"d_4_l_{l}", l) for l in range(21, 61)]

@dataclass
class shorterS1a_d_trulyleq5_Config(Basic_Config):
    dataset = "S1a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    train_files = [
        "/data/yingshac/tmlr/shorter/train/fix_d_1.json",
        "/data/yingshac/tmlr/shorter/train/fix_d_2.json",
        "/data/yingshac/tmlr/shorter/train/fix_d_3.json",
        "/data/yingshac/tmlr/shorter/train/fix_d_4.json",
        "/data/yingshac/tmlr/shorter/train/fix_d_5.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/shorter/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_1.json", "d_1", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_6.json", "d_6", None),
    ]
    test_files = [("/data/yingshac/tmlr/shorter/test/fix_d_5.json", f"d_5_l_{l}", l) for l in range(21, 61)]

@dataclass
class shorterS1a_d_trulyleq6_Config(Basic_Config):
    dataset = "S1a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    train_files = [
        "/data/yingshac/tmlr/shorter/train/fix_d_1.json",
        "/data/yingshac/tmlr/shorter/train/fix_d_2.json",
        "/data/yingshac/tmlr/shorter/train/fix_d_3.json",
        "/data/yingshac/tmlr/shorter/train/fix_d_4.json",
        "/data/yingshac/tmlr/shorter/train/fix_d_5.json",
        "/data/yingshac/tmlr/shorter/train/fix_d_6.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/shorter/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_1.json", "d_1", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/shorter/validation/fix_d_5.json", "d_5", None),
    ]
    test_files = [("/data/yingshac/tmlr/shorter/test/fix_d_6.json", f"d_6_l_{l}", l) for l in range(21, 61)]

### S2
### without eos; cumulative; add d=1 data
@dataclass
class S2a_d_trulyleq2_Config(Basic_Config):
    dataset = "S2a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    out_S2_vocab = ['0', '1']
    S2_loss_weight = 0.5
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_1.json",
        "/data/yingshac/tmlr/train/fix_d_2.json"
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_1.json", "d_1", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
        ("/data/yingshac/tmlr/validation/fix_d_9.json", "d_9", None),
        ("/data/yingshac/tmlr/validation/fix_d_10.json", "d_10", None),
        ("/data/yingshac/tmlr/validation/fix_d_11.json", "d_11", None),
        ("/data/yingshac/tmlr/validation/fix_d_12.json", "d_12", None),
        ("/data/yingshac/tmlr/validation/fix_d_13.json", "d_13", None),
        ("/data/yingshac/tmlr/validation/fix_d_14.json", "d_14", None),
        ("/data/yingshac/tmlr/validation/fix_d_15.json", "d_15", None),
        ("/data/yingshac/tmlr/validation/fix_d_16.json", "d_16", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_2.json", f"d_2_l_{l}", l) for l in range(41, 121)]

@dataclass
class S2a_d_trulyleq3_Config(Basic_Config):
    dataset = "S2a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    out_S2_vocab = ['0', '1']
    S2_loss_weight = 0.5
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_1.json",
        "/data/yingshac/tmlr/train/fix_d_2.json",
        "/data/yingshac/tmlr/train/fix_d_3.json"
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_1.json", "d_1", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
        ("/data/yingshac/tmlr/validation/fix_d_9.json", "d_9", None),
        ("/data/yingshac/tmlr/validation/fix_d_10.json", "d_10", None),
        ("/data/yingshac/tmlr/validation/fix_d_11.json", "d_11", None),
        ("/data/yingshac/tmlr/validation/fix_d_12.json", "d_12", None),
        ("/data/yingshac/tmlr/validation/fix_d_13.json", "d_13", None),
        ("/data/yingshac/tmlr/validation/fix_d_14.json", "d_14", None),
        ("/data/yingshac/tmlr/validation/fix_d_15.json", "d_15", None),
        ("/data/yingshac/tmlr/validation/fix_d_16.json", "d_16", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_3.json", f"d_3_l_{l}", l) for l in range(41, 121)]

@dataclass
class S2a_d_trulyleq4_Config(Basic_Config):
    dataset = "S2a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    out_S2_vocab = ['0', '1']
    S2_loss_weight = 0.5
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_1.json",
        "/data/yingshac/tmlr/train/fix_d_2.json",
        "/data/yingshac/tmlr/train/fix_d_3.json",
        "/data/yingshac/tmlr/train/fix_d_4.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_1.json", "d_1", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
        ("/data/yingshac/tmlr/validation/fix_d_9.json", "d_9", None),
        ("/data/yingshac/tmlr/validation/fix_d_10.json", "d_10", None),
        ("/data/yingshac/tmlr/validation/fix_d_11.json", "d_11", None),
        ("/data/yingshac/tmlr/validation/fix_d_12.json", "d_12", None),
        ("/data/yingshac/tmlr/validation/fix_d_13.json", "d_13", None),
        ("/data/yingshac/tmlr/validation/fix_d_14.json", "d_14", None),
        ("/data/yingshac/tmlr/validation/fix_d_15.json", "d_15", None),
        ("/data/yingshac/tmlr/validation/fix_d_16.json", "d_16", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_4.json", f"d_4_l_{l}", l) for l in range(41, 121)]

@dataclass
class S2a_d_trulyleq5_Config(Basic_Config):
    dataset = "S2a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    out_S2_vocab = ['0', '1']
    S2_loss_weight = 0.5
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_1.json",
        "/data/yingshac/tmlr/train/fix_d_2.json",
        "/data/yingshac/tmlr/train/fix_d_3.json",
        "/data/yingshac/tmlr/train/fix_d_4.json",
        "/data/yingshac/tmlr/train/fix_d_5.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_1.json", "d_1", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
        ("/data/yingshac/tmlr/validation/fix_d_9.json", "d_9", None),
        ("/data/yingshac/tmlr/validation/fix_d_10.json", "d_10", None),
        ("/data/yingshac/tmlr/validation/fix_d_11.json", "d_11", None),
        ("/data/yingshac/tmlr/validation/fix_d_12.json", "d_12", None),
        ("/data/yingshac/tmlr/validation/fix_d_13.json", "d_13", None),
        ("/data/yingshac/tmlr/validation/fix_d_14.json", "d_14", None),
        ("/data/yingshac/tmlr/validation/fix_d_15.json", "d_15", None),
        ("/data/yingshac/tmlr/validation/fix_d_16.json", "d_16", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_5.json", f"d_5_l_{l}", l) for l in range(41, 121)]

@dataclass
class S2a_d_trulyleq6_Config(Basic_Config):
    dataset = "S2a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    out_S2_vocab = ['0', '1']
    S2_loss_weight = 0.5
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_1.json",
        "/data/yingshac/tmlr/train/fix_d_2.json",
        "/data/yingshac/tmlr/train/fix_d_3.json",
        "/data/yingshac/tmlr/train/fix_d_4.json",
        "/data/yingshac/tmlr/train/fix_d_5.json",
        "/data/yingshac/tmlr/train/fix_d_6.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_1.json", "d_1", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
        ("/data/yingshac/tmlr/validation/fix_d_9.json", "d_9", None),
        ("/data/yingshac/tmlr/validation/fix_d_10.json", "d_10", None),
        ("/data/yingshac/tmlr/validation/fix_d_11.json", "d_11", None),
        ("/data/yingshac/tmlr/validation/fix_d_12.json", "d_12", None),
        ("/data/yingshac/tmlr/validation/fix_d_13.json", "d_13", None),
        ("/data/yingshac/tmlr/validation/fix_d_14.json", "d_14", None),
        ("/data/yingshac/tmlr/validation/fix_d_15.json", "d_15", None),
        ("/data/yingshac/tmlr/validation/fix_d_16.json", "d_16", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_6.json", f"d_6_l_{l}", l) for l in range(41, 121)]

@dataclass
class S2a_d_trulyleq7_Config(Basic_Config):
    dataset = "S2a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    out_S2_vocab = ['0', '1']
    S2_loss_weight = 0.5
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_1.json",
        "/data/yingshac/tmlr/train/fix_d_2.json",
        "/data/yingshac/tmlr/train/fix_d_3.json",
        "/data/yingshac/tmlr/train/fix_d_4.json",
        "/data/yingshac/tmlr/train/fix_d_5.json",
        "/data/yingshac/tmlr/train/fix_d_6.json",
        "/data/yingshac/tmlr/train/fix_d_7.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_1.json", "d_1", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
        ("/data/yingshac/tmlr/validation/fix_d_9.json", "d_9", None),
        ("/data/yingshac/tmlr/validation/fix_d_10.json", "d_10", None),
        ("/data/yingshac/tmlr/validation/fix_d_11.json", "d_11", None),
        ("/data/yingshac/tmlr/validation/fix_d_12.json", "d_12", None),
        ("/data/yingshac/tmlr/validation/fix_d_13.json", "d_13", None),
        ("/data/yingshac/tmlr/validation/fix_d_14.json", "d_14", None),
        ("/data/yingshac/tmlr/validation/fix_d_15.json", "d_15", None),
        ("/data/yingshac/tmlr/validation/fix_d_16.json", "d_16", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_7.json", f"d_7_l_{l}", l) for l in range(41, 121)]

@dataclass
class S2a_d_trulyleq8_Config(Basic_Config):
    dataset = "S2a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    out_S2_vocab = ['0', '1']
    S2_loss_weight = 0.5
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_1.json",
        "/data/yingshac/tmlr/train/fix_d_2.json",
        "/data/yingshac/tmlr/train/fix_d_3.json",
        "/data/yingshac/tmlr/train/fix_d_4.json",
        "/data/yingshac/tmlr/train/fix_d_5.json",
        "/data/yingshac/tmlr/train/fix_d_6.json",
        "/data/yingshac/tmlr/train/fix_d_7.json",
        "/data/yingshac/tmlr/train/fix_d_8.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
        ("/data/yingshac/tmlr/validation/fix_d_1.json", "d_1", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_9.json", "d_9", None),
        ("/data/yingshac/tmlr/validation/fix_d_10.json", "d_10", None),
        ("/data/yingshac/tmlr/validation/fix_d_11.json", "d_11", None),
        ("/data/yingshac/tmlr/validation/fix_d_12.json", "d_12", None),
        ("/data/yingshac/tmlr/validation/fix_d_13.json", "d_13", None),
        ("/data/yingshac/tmlr/validation/fix_d_14.json", "d_14", None),
        ("/data/yingshac/tmlr/validation/fix_d_15.json", "d_15", None),
        ("/data/yingshac/tmlr/validation/fix_d_16.json", "d_16", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_8.json", f"d_8_l_{l}", l) for l in range(41, 121)]

### S3
### without eos; cumulative; add d=1 data
@dataclass
class S3a_d_trulyleq2_Config(Basic_Config):
    dataset = "S3a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    out_S3_vocab = [f'd{d}' for d in range(-1, 18)]
    S3_loss_weight = 1.0
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_1.json",
        "/data/yingshac/tmlr/train/fix_d_2.json"
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_1.json", "d_1", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
        ("/data/yingshac/tmlr/validation/fix_d_9.json", "d_9", None),
        ("/data/yingshac/tmlr/validation/fix_d_10.json", "d_10", None),
        ("/data/yingshac/tmlr/validation/fix_d_11.json", "d_11", None),
        ("/data/yingshac/tmlr/validation/fix_d_12.json", "d_12", None),
        ("/data/yingshac/tmlr/validation/fix_d_13.json", "d_13", None),
        ("/data/yingshac/tmlr/validation/fix_d_14.json", "d_14", None),
        ("/data/yingshac/tmlr/validation/fix_d_15.json", "d_15", None),
        ("/data/yingshac/tmlr/validation/fix_d_16.json", "d_16", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_2.json", f"d_2_l_{l}", l) for l in range(41, 121)]

@dataclass
class S3a_d_trulyleq3_Config(Basic_Config):
    dataset = "S3a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    out_S3_vocab = [f'd{d}' for d in range(-1, 18)]
    S3_loss_weight = 1.0
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_1.json",
        "/data/yingshac/tmlr/train/fix_d_2.json",
        "/data/yingshac/tmlr/train/fix_d_3.json"
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_1.json", "d_1", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
        ("/data/yingshac/tmlr/validation/fix_d_9.json", "d_9", None),
        ("/data/yingshac/tmlr/validation/fix_d_10.json", "d_10", None),
        ("/data/yingshac/tmlr/validation/fix_d_11.json", "d_11", None),
        ("/data/yingshac/tmlr/validation/fix_d_12.json", "d_12", None),
        ("/data/yingshac/tmlr/validation/fix_d_13.json", "d_13", None),
        ("/data/yingshac/tmlr/validation/fix_d_14.json", "d_14", None),
        ("/data/yingshac/tmlr/validation/fix_d_15.json", "d_15", None),
        ("/data/yingshac/tmlr/validation/fix_d_16.json", "d_16", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_3.json", f"d_3_l_{l}", l) for l in range(41, 121)]

@dataclass
class S3a_d_trulyleq4_Config(Basic_Config):
    dataset = "S3a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    out_S3_vocab = [f'd{d}' for d in range(-1, 18)]
    S3_loss_weight = 1.0
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_1.json",
        "/data/yingshac/tmlr/train/fix_d_2.json",
        "/data/yingshac/tmlr/train/fix_d_3.json",
        "/data/yingshac/tmlr/train/fix_d_4.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_1.json", "d_1", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
        ("/data/yingshac/tmlr/validation/fix_d_9.json", "d_9", None),
        ("/data/yingshac/tmlr/validation/fix_d_10.json", "d_10", None),
        ("/data/yingshac/tmlr/validation/fix_d_11.json", "d_11", None),
        ("/data/yingshac/tmlr/validation/fix_d_12.json", "d_12", None),
        ("/data/yingshac/tmlr/validation/fix_d_13.json", "d_13", None),
        ("/data/yingshac/tmlr/validation/fix_d_14.json", "d_14", None),
        ("/data/yingshac/tmlr/validation/fix_d_15.json", "d_15", None),
        ("/data/yingshac/tmlr/validation/fix_d_16.json", "d_16", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_4.json", f"d_4_l_{l}", l) for l in range(41, 121)]

@dataclass
class S3a_d_trulyleq5_Config(Basic_Config):
    dataset = "S3a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    out_S3_vocab = [f'd{d}' for d in range(-1, 18)]
    S3_loss_weight = 1.0
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_1.json",
        "/data/yingshac/tmlr/train/fix_d_2.json",
        "/data/yingshac/tmlr/train/fix_d_3.json",
        "/data/yingshac/tmlr/train/fix_d_4.json",
        "/data/yingshac/tmlr/train/fix_d_5.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_1.json", "d_1", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
        ("/data/yingshac/tmlr/validation/fix_d_9.json", "d_9", None),
        ("/data/yingshac/tmlr/validation/fix_d_10.json", "d_10", None),
        ("/data/yingshac/tmlr/validation/fix_d_11.json", "d_11", None),
        ("/data/yingshac/tmlr/validation/fix_d_12.json", "d_12", None),
        ("/data/yingshac/tmlr/validation/fix_d_13.json", "d_13", None),
        ("/data/yingshac/tmlr/validation/fix_d_14.json", "d_14", None),
        ("/data/yingshac/tmlr/validation/fix_d_15.json", "d_15", None),
        ("/data/yingshac/tmlr/validation/fix_d_16.json", "d_16", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_5.json", f"d_5_l_{l}", l) for l in range(41, 121)]

@dataclass
class S3a_d_trulyleq6_Config(Basic_Config):
    dataset = "S3a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    out_S3_vocab = [f'd{d}' for d in range(-1, 18)]
    S3_loss_weight = 1.0
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_1.json",
        "/data/yingshac/tmlr/train/fix_d_2.json",
        "/data/yingshac/tmlr/train/fix_d_3.json",
        "/data/yingshac/tmlr/train/fix_d_4.json",
        "/data/yingshac/tmlr/train/fix_d_5.json",
        "/data/yingshac/tmlr/train/fix_d_6.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_1.json", "d_1", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
        ("/data/yingshac/tmlr/validation/fix_d_9.json", "d_9", None),
        ("/data/yingshac/tmlr/validation/fix_d_10.json", "d_10", None),
        ("/data/yingshac/tmlr/validation/fix_d_11.json", "d_11", None),
        ("/data/yingshac/tmlr/validation/fix_d_12.json", "d_12", None),
        ("/data/yingshac/tmlr/validation/fix_d_13.json", "d_13", None),
        ("/data/yingshac/tmlr/validation/fix_d_14.json", "d_14", None),
        ("/data/yingshac/tmlr/validation/fix_d_15.json", "d_15", None),
        ("/data/yingshac/tmlr/validation/fix_d_16.json", "d_16", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_6.json", f"d_6_l_{l}", l) for l in range(41, 121)]

@dataclass
class S3a_d_trulyleq7_Config(Basic_Config):
    dataset = "S3a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    out_S3_vocab = [f'd{d}' for d in range(-1, 18)]
    S3_loss_weight = 1.0
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_1.json",
        "/data/yingshac/tmlr/train/fix_d_2.json",
        "/data/yingshac/tmlr/train/fix_d_3.json",
        "/data/yingshac/tmlr/train/fix_d_4.json",
        "/data/yingshac/tmlr/train/fix_d_5.json",
        "/data/yingshac/tmlr/train/fix_d_6.json",
        "/data/yingshac/tmlr/train/fix_d_7.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_1.json", "d_1", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
        ("/data/yingshac/tmlr/validation/fix_d_9.json", "d_9", None),
        ("/data/yingshac/tmlr/validation/fix_d_10.json", "d_10", None),
        ("/data/yingshac/tmlr/validation/fix_d_11.json", "d_11", None),
        ("/data/yingshac/tmlr/validation/fix_d_12.json", "d_12", None),
        ("/data/yingshac/tmlr/validation/fix_d_13.json", "d_13", None),
        ("/data/yingshac/tmlr/validation/fix_d_14.json", "d_14", None),
        ("/data/yingshac/tmlr/validation/fix_d_15.json", "d_15", None),
        ("/data/yingshac/tmlr/validation/fix_d_16.json", "d_16", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_7.json", f"d_7_l_{l}", l) for l in range(41, 121)]

@dataclass
class S3a_d_trulyleq8_Config(Basic_Config):
    dataset = "S3a_"
    in_vocab = ['(', ')', '<pad>']
    out_vocab = ['a', 'r']
    out_S3_vocab = [f'd{d}' for d in range(-1, 18)]
    S3_loss_weight = 1.0
    train_files = [
        "/data/yingshac/tmlr/train/fix_d_1.json",
        "/data/yingshac/tmlr/train/fix_d_2.json",
        "/data/yingshac/tmlr/train/fix_d_3.json",
        "/data/yingshac/tmlr/train/fix_d_4.json",
        "/data/yingshac/tmlr/train/fix_d_5.json",
        "/data/yingshac/tmlr/train/fix_d_6.json",
        "/data/yingshac/tmlr/train/fix_d_7.json",
        "/data/yingshac/tmlr/train/fix_d_8.json",
    ]
    val_files = [
        ("/data/yingshac/tmlr/validation/fix_d_8.json", "d_8", None),
        ("/data/yingshac/tmlr/validation/fix_d_1.json", "d_1", None),
        ("/data/yingshac/tmlr/validation/fix_d_2.json", "d_2", None),
        ("/data/yingshac/tmlr/validation/fix_d_3.json", "d_3", None),
        ("/data/yingshac/tmlr/validation/fix_d_4.json", "d_4", None),
        ("/data/yingshac/tmlr/validation/fix_d_5.json", "d_5", None),
        ("/data/yingshac/tmlr/validation/fix_d_6.json", "d_6", None),
        ("/data/yingshac/tmlr/validation/fix_d_7.json", "d_7", None),
        ("/data/yingshac/tmlr/validation/fix_d_9.json", "d_9", None),
        ("/data/yingshac/tmlr/validation/fix_d_10.json", "d_10", None),
        ("/data/yingshac/tmlr/validation/fix_d_11.json", "d_11", None),
        ("/data/yingshac/tmlr/validation/fix_d_12.json", "d_12", None),
        ("/data/yingshac/tmlr/validation/fix_d_13.json", "d_13", None),
        ("/data/yingshac/tmlr/validation/fix_d_14.json", "d_14", None),
        ("/data/yingshac/tmlr/validation/fix_d_15.json", "d_15", None),
        ("/data/yingshac/tmlr/validation/fix_d_16.json", "d_16", None),
    ]
    test_files = [("/data/yingshac/tmlr/test/fix_d_8.json", f"d_8_l_{l}", l) for l in range(41, 121)]