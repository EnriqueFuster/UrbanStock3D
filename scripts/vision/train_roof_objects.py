"""Train the first roof-object segmentation baseline on canonical RID2 data."""

import argparse
import os
from pathlib import Path

from urbanstock3d.vision.common import load_roof_objects_config, load_taxonomy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/vision/roof_objects.yaml"))
    parser.add_argument("--dataset", type=Path, default=Path("data/vision/rid2/yolo/dataset.yaml"))
    parser.add_argument("--run-name", default="yolo26s-seg-rid2-v1")
    parser.add_argument("--epochs", type=int, default=None, help="Override the configured epochs.")
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of training images to use, between 0 and 1.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Train one epoch on 2%% of the data to verify the pipeline.",
    )
    arguments = parser.parse_args()
    if arguments.epochs is not None and arguments.epochs <= 0:
        parser.error("--epochs must be greater than zero")
    if not 0 < arguments.fraction <= 1:
        parser.error("--fraction must be greater than zero and at most one")

    config = load_roof_objects_config(arguments.config)
    taxonomy = load_taxonomy(Path("config/vision/taxonomy.yaml"))
    config.validate_taxonomy(taxonomy)
    if not arguments.dataset.exists():
        raise SystemExit(f"Missing converted dataset: {arguments.dataset}")
    if config.dataset_id is None:
        raise SystemExit("The training configuration must identify its dataset")

    # Keep optional heavy imports out of the normal application startup.
    import torch
    from ultralytics import YOLO, settings  # type: ignore[attr-defined]

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Run `uv sync --group dev --group vision` first.")

    database_path = Path("mlflow.db").resolve().as_posix()
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{database_path}"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = config.tracking.experiment
    os.environ["MLFLOW_RUN"] = arguments.run_name
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    settings.update(  # type: ignore[no-untyped-call]
        {"mlflow": True, "weights_dir": str(Path("models/pretrained").resolve())}
    )

    epochs = 1 if arguments.smoke_test else arguments.epochs or config.training.epochs
    fraction = 0.02 if arguments.smoke_test else arguments.fraction
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {config.model.checkpoint}")
    print(f"Dataset: {config.dataset_id}")
    print(f"Run: {arguments.run_name} ({epochs} epoch(s), fraction={fraction})")

    model = YOLO(config.model.checkpoint)
    model.train(
        data=str(arguments.dataset.resolve()),
        epochs=epochs,
        imgsz=config.training.image_size,
        batch=config.training.batch_size,
        patience=config.training.early_stopping_patience,
        seed=config.training.seed,
        deterministic=True,
        device=0,
        workers=0,
        fraction=fraction,
        project=str(Path("models/roof_objects").resolve()),
        name=arguments.run_name,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
