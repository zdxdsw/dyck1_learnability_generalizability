from dataclasses import dataclass, field

@dataclass
class TaskConfig:
    dataset: str = ""

    in_vocab: list[str] = field(default_factory=list)
    out_vocab: list[str] = field(default_factory=list)

    train_files: list[str] = field(default_factory=list)
    val_files: list[tuple[str, str, int | None]] = field(default_factory=list)
    test_files: list[tuple[str, str, int]] = field(default_factory=list)

    out_S2_vocab: list[str] | None = None
    S2_loss_weight: float = 0.0

    out_S3_vocab: list[str] | None = None
    S3_loss_weight: float = 0.0