"""Export a trained roof-object segmenter to ONNX and validate the file."""

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path, help="Path to a trained best.pt checkpoint.")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    if not arguments.weights.exists():
        raise SystemExit(f"Missing checkpoint: {arguments.weights}")
    if arguments.image_size <= 0:
        parser.error("--image-size must be greater than zero")

    import onnx
    from ultralytics import YOLO  # type: ignore[attr-defined]

    model = YOLO(arguments.weights)
    exported = Path(
        model.export(
            format="onnx",
            imgsz=arguments.image_size,
            dynamic=True,
            simplify=False,
            opset=18,
        )
    )
    output = arguments.output or arguments.weights.parents[1] / "export" / "model.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != output.resolve():
        shutil.move(exported, output)

    onnx.checker.check_model(onnx.load(output))
    print(f"Validated ONNX model: {output}")


if __name__ == "__main__":
    main()
