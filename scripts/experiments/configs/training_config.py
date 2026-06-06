from dataclasses import dataclass

@dataclass
class TrainingConfig:
    num_epochs: int = 10
    per_device_train_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    per_device_eval_batch_size: int = 32
    eval_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    shuffle_training_data: bool = True


@dataclass
class LoggerConfig:
    logging_steps: int = 200
    more_eval_th_by_epoch: int = 1
    more_eval_th_by_val_acc: float = -1