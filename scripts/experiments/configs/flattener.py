from dataclasses import asdict, is_dataclass
from types import SimpleNamespace


from dataclasses import fields, is_dataclass
from types import SimpleNamespace


def flatten_config(config):
    """
    Convert a nested dataclass config into a flat SimpleNamespace.

    Recursively unravels nested dataclasses.

    Example:
        config.run.task
        config.training.learning_rate
        config.task.in_vocab
        config.model.hidden_size
        config.model.position_encoding.pe_type

    become:
        config.task
        config.learning_rate
        config.in_vocab
        config.hidden_size
        config.pe_type
    """

    if not is_dataclass(config):
        raise TypeError(f"Expected a dataclass instance, got {type(config)}")

    flat = {}

    def _flatten(obj, path="config"):
        if is_dataclass(obj):
            for f in fields(obj):
                name = f.name
                value = getattr(obj, name)
                _flatten(value, path=f"{path}.{name}")
        else:
            key = path.split(".")[-1]

            if key in flat:
                raise ValueError(
                    f"Duplicate config field found while flattening: {key!r}. "
                    f"Current path: {path}. "
                    f"Existing value: {flat[key]!r}. "
                    f"New value: {obj!r}."
                )

            flat[key] = obj

    _flatten(config)

    return SimpleNamespace(**flat)