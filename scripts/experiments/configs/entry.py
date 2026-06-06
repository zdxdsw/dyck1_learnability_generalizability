from dataclasses import dataclass, field

from configs import (
    RunConfig,
    TrainingConfig,
    LoggerConfig,
    CollatorConfig,
    TaskConfig,
    task_config_builder, 
    RNNConfig,
    TransformerConfig,
    model_config_builder,
    flatten_config,
)

@dataclass
class EverythingConfig:
    run_config: RunConfig = field(default_factory=RunConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    logger_config: LoggerConfig = field(default_factory=LoggerConfig)
    task_config: TaskConfig = field(default_factory=TaskConfig)
    collator_config: CollatorConfig = field(default_factory=CollatorConfig)
    model_config: RNNConfig | TransformerConfig = field(default_factory=RNNConfig)


def flat_everything_config_builder(task: str):
    config = EverythingConfig()
    config.run_config.task = task # Passed from terminal when launching run.py

    config.task_config = task_config_builder(task)
    config.model_config = model_config_builder(config.run_config.model)
    
    return flatten_config(config)