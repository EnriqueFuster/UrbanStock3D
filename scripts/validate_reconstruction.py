"""Validate one reconstructed CityJSONSeq model against its source footprint."""

import argparse
import json
from pathlib import Path
from typing import Any

from urbanstock3d.providers.pnoa_lidar import footprint_polygons_utm
from urbanstock3d.reconstruction.validation import assess_cityjson_lidar_fit, validate_cityjsonseq


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("building_geojson", type=Path)
    parser.add_argument("--lidar", type=Path, help="Optional LiDAR crop for point-to-roof fit.")
    parser.add_argument("--lod", default="2.2")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_report(
    model: Path,
    building_path: Path,
    output: Path,
    *,
    lidar_path: Path | None = None,
    lod: str = "2.2",
) -> Path:
    building: dict[str, Any] = json.loads(building_path.read_text(encoding="utf-8"))
    footprint = footprint_polygons_utm(building["geometry"])
    report = validate_cityjsonseq(model, footprint, lod=lod)
    lidar_fit = (
        assess_cityjson_lidar_fit(model, lidar_path, footprint, lod=lod)
        if lidar_path is not None
        else None
    )
    payload = {
        "schema_version": 1,
        "kind": "reconstruction_geometry_quality",
        "building_id": building["id"],
        "source": {"model": model.as_posix(), "footprint": building_path.as_posix()},
        "quality": report.to_dict(),
        "lidar_fit": lidar_fit.to_dict() if lidar_fit else None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    arguments = parse_arguments()
    output = build_report(
        arguments.model,
        arguments.building_geojson,
        arguments.output,
        lidar_path=arguments.lidar,
        lod=arguments.lod,
    )
    print(f"Wrote reconstruction quality report to {output}")


if __name__ == "__main__":
    main()
