from dataclasses import dataclass, field

from configs import (
    RunConfig,
    TrainingConfig,
    LoggerConfig,
    TaskConfig,
    CollatorConfig,
    RNNConfig,
    TransformerConfig,
)

@dataclass
class EverythingConfig:
    run_config: RunConfig = field(default_factory=RunConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    logger_config: LoggerConfig = field(default_factory=LoggerConfig)
    task_config: TaskConfig = field(default_factory=TaskConfig)
    collator_config: CollatorConfig = field(default_factory=CollatorConfig)
    model_config: RNNConfig | TransformerConfig = field(default_factory=RNNConfig)