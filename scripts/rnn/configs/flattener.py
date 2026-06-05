from dataclasses import asdict, is_dataclass
from types import SimpleNamespace


def flatten_config(config):
    """
    Convert a nested dataclass config into a flat SimpleNamespace.
    Different chunks of the config don't share any field names, so we can just merge all the sections together.
    This facilitates dumping config into json. 

    For example:
        config.run.task
        config.training.learning_rate
        config.task.in_vocab
        config.model.hidden_size

    become:
        config.task
        config.learning_rate
        config.in_vocab
        config.hidden_size
    """

    if not is_dataclass(config):
        raise TypeError(f"Expected a dataclass instance, got {type(config)}")

    flat = {}

    for section_name, section_value in vars(config).items():
        if is_dataclass(section_value):
            section_dict = asdict(section_value)
            overlap = set(flat).intersection(section_dict)
            if overlap:
                raise ValueError(
                    f"Duplicate config fields found while flattening section "
                    f"{section_name}: {sorted(overlap)}"
                )
            flat.update(section_dict)
        else:
            if section_name in flat:
                raise ValueError(f"Duplicate config field: {section_name}")
            flat[section_name] = section_value

    return SimpleNamespace(**flat)