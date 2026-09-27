"""Prepare prioritized PNOA images and label templates for human review."""

import argparse
from pathlib import Path

from urbanstock3d.vision.roof_objects.annotation import build_pnoa_annotation_queue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=Path("data/selected_buildings.json"))
    parser.add_argument(
        "--feasibility", type=Path, default=Path("outputs/vision/pnoa_feasibility.json")
    )
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--queue-root", type=Path, default=Path("data/vision/local_pnoa/review"))
    arguments = parser.parse_args()
    for path in (arguments.selection, arguments.feasibility):
        if not path.exists():
            raise SystemExit(f"Missing input: {path}")

    manifest = build_pnoa_annotation_queue(
        arguments.selection,
        arguments.feasibility,
        arguments.outputs_root,
        arguments.queue_root,
    )
    print(f"Wrote PNOA annotation queue to {manifest}")


if __name__ == "__main__":
    main()
