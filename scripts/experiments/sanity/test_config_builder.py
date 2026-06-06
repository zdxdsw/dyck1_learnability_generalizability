import pprint, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from configs import flat_everything_config_builder
from contextlib import redirect_stdout

# Change this manually when testing.
TASK = "S1_d_trulyleq8"

# Long fields to print at the end.
LONG_FIELDS = {
    "train_files",
    "val_files",
    "test_files",
}

def format_list_multiline(xs, indent: str = "    "):
    """
    Format a list as:

        [
         item1,
         item2
        ]

    The caller is responsible for printing `field_name = ` before it.
    """
    if len(xs) == 0:
        return "[]"

    lines = ["["]

    for i, item in enumerate(xs):
        suffix = "," if i < len(xs) - 1 else ""
        lines.append(f" {pprint.pformat(item, width=120)}{suffix}")

    lines.append("]")

    return "\n".join(lines)

def format_value(k, v, max_width: int = 120):
    """
    Pretty-format values so long lists are easier to read.
    Only LONG_FIELDS list values are expanded multiline. Other fields that are lists stay inline.
    """
    if k in LONG_FIELDS and isinstance(v, list):
        return format_list_multiline(v)

    return pprint.pformat(v, width=max_width, compact=False)

def ordered_config_items(config):
    """
    Return config items with long file-list fields moved to the end.
    """
    items = list(vars(config).items())

    short_items = [
        (k, v)
        for k, v in items
        if k not in LONG_FIELDS
    ]

    long_items = [
        (k, v)
        for k, v in items
        if k in LONG_FIELDS
    ]

    return short_items + long_items


def print_namespace_as_dataclass_like(config, class_name: str = "FlatConfig"):
    """
    Print a SimpleNamespace/object in a dataclass-like format:

        @dataclass
        class FlatConfig:
            key = value
            ...

    This is only for visual inspection.
    """
    print("@dataclass")
    print(f"class {class_name}:")
    for k, v in ordered_config_items(config):
        formatted = format_value(k, v)

        # Indent multiline values nicely.
        formatted = formatted.replace("\n", "\n    ")
        print(f"    {k} = {formatted}")


def run_test():
    config = flat_everything_config_builder(TASK)

    print("=" * 80)
    print(f"Built flat config for task: {TASK}")
    print("=" * 80)
    print_namespace_as_dataclass_like(config, class_name="FlatEverythingConfig")

def main():
    output_path = Path(__file__).with_name("test_config_builder_output.txt")

    # 1. Print to console.
    run_test()

    # 2. Save the same output to file.
    with open(output_path, "w") as f:
        with redirect_stdout(f):
            run_test()


if __name__ == "__main__":
    main()