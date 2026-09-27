"""Evaluate the RID2 baseline over the selected PNOA building sample."""

import argparse
import json
from pathlib import Path

from urbanstock3d.vision.roof_objects.batch import evaluate_selected_buildings
from urbanstock3d.vision.roof_objects.service import UltralyticsRoofObjectSegmenter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--selection", type=Path, default=Path("data/selected_buildings.json"))
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--report", type=Path, default=Path("outputs/vision/pnoa_feasibility.json"))
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--no-download", action="store_true")
    arguments = parser.parse_args()

    from ultralytics import YOLO  # type: ignore[attr-defined]

    shared_model = YOLO(arguments.weights)

    def segmenter_factory(mask_dir: Path) -> UltralyticsRoofObjectSegmenter:
        return UltralyticsRoofObjectSegmenter(
            arguments.weights,
            mask_dir,
            confidence=arguments.confidence,
            model=shared_model,
        )

    report = evaluate_selected_buildings(
        arguments.selection,
        arguments.outputs_root,
        segmenter_factory,
        acquire_missing=not arguments.no_download,
    )
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote PNOA feasibility report to {arguments.report}")


if __name__ == "__main__":
    main()
