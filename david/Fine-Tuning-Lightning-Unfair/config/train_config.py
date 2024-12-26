import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    pretrained_model: str = "roberta-base"
    num_classes: int = 2
    lr: float = 2e-4
    max_length: int = 128
    batch_size: int = 512
    num_workers: int = os.cpu_count()
    max_epochs: int = 100
    debug_mode_sample: int | None = None
    max_time: dict[str, float] = field(default_factory=lambda: {"hours": 3})
    model_checkpoint_dir: str = os.path.join(
        Path(__file__).parents[1],
        "model-checkpoints",
    )
    min_delta: float = 0.005
    patience: int = 40

    # MLflow
    mlflow_experiment_name: str = "ExperimentV3.2"
    mlflow_run_name: str = "onnx-gpu-run12"
    mlflow_description: str = (
        f"PEFT tune roberta-base to classify {mlflow_experiment_name}."
    )
