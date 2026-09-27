"""Run the RID2 roof-object baseline on one PNOA orthophoto crop."""

import argparse
from pathlib import Path

from urbanstock3d.vision.roof_objects.service import (
    UltralyticsRoofObjectSegmenter,
    infer_roof_objects,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--building-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--confidence", type=float, default=0.25)
    arguments = parser.parse_args()
    if not arguments.image.exists():
        raise SystemExit(f"Missing orthophoto crop: {arguments.image}")

    output_dir = arguments.output_dir or arguments.image.with_name("roof_objects")
    segmenter = UltralyticsRoofObjectSegmenter(
        arguments.weights,
        output_dir / "masks",
        confidence=arguments.confidence,
    )
    result_path = output_dir / "result.json"
    result = infer_roof_objects(
        arguments.building_id,
        arguments.image,
        segmenter,
        result_path,
    )
    print(f"Detected {len(result.objects)} candidate roof object(s)")
    print(f"Wrote structured result to {result_path}")


if __name__ == "__main__":
    main()
