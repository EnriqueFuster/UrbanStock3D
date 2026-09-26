"""Evaluate one roof-object checkpoint and render sample predictions."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path, help="Path to a trained best.pt checkpoint.")
    parser.add_argument("--dataset", type=Path, default=Path("data/vision/rid2/yolo/dataset.yaml"))
    parser.add_argument("--source-dir", type=Path, default=Path("data/vision/rid2/yolo/images/val"))
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=None)
    arguments = parser.parse_args()
    if not arguments.weights.exists():
        raise SystemExit(f"Missing checkpoint: {arguments.weights}")
    if arguments.samples <= 0:
        parser.error("--samples must be greater than zero")

    from PIL import Image
    from ultralytics import YOLO  # type: ignore[attr-defined]

    output_dir = arguments.output_dir or arguments.weights.parents[1] / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(arguments.weights)

    metrics = model.val(data=str(arguments.dataset.resolve()), imgsz=512, device=0, workers=0)
    metric_values = {name: float(value) for name, value in metrics.results_dict.items()}
    (output_dir / "metrics.json").write_text(
        json.dumps(metric_values, indent=2) + "\n", encoding="utf-8"
    )

    image_paths = sorted(arguments.source_dir.glob("*.png"))[: arguments.samples]
    results = model.predict(image_paths, imgsz=512, conf=0.25, device=0, verbose=False)
    tiles = [
        Image.fromarray(result.plot()[..., ::-1])  # type: ignore[union-attr]
        for result in results
    ]
    columns = 4
    rows = (len(tiles) + columns - 1) // columns
    tile_width, tile_height = tiles[0].size
    montage = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    for index, tile in enumerate(tiles):
        montage.paste(tile, ((index % columns) * tile_width, (index // columns) * tile_height))
    montage.save(output_dir / "prediction_montage.jpg", quality=92)

    print(json.dumps(metric_values, indent=2))
    print(f"Wrote evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
