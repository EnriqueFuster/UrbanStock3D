"""Vectorize roof-object masks and clip them to a cadastral footprint."""

import argparse
from pathlib import Path

from urbanstock3d.vision.models import RoofVisionResult
from urbanstock3d.vision.roof_objects.vectorization import vectorize_and_clip_roof_objects


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="Raw roof-object result.json.")
    parser.add_argument("building_geojson", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    arguments = parser.parse_args()
    for path in (arguments.result, arguments.building_geojson):
        if not path.exists():
            raise SystemExit(f"Missing input: {path}")

    result = RoofVisionResult.model_validate_json(arguments.result.read_text(encoding="utf-8"))
    output_dir = arguments.output_dir or arguments.result.parent / "vectorized"
    vectorized = vectorize_and_clip_roof_objects(result, arguments.building_geojson, output_dir)
    output_path = output_dir / "result.json"
    output_path.write_text(vectorized.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"Retained {len(vectorized.objects)} footprint-clipped object(s)")
    print(f"Wrote vector result to {output_path}")


if __name__ == "__main__":
    main()
