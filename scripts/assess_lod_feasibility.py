"""Assess defensible reconstruction LoDs without selecting a backend."""

import argparse
import json
from pathlib import Path
from typing import Any

from urbanstock3d.reconstruction.evidence import HeightEvidenceMetrics
from urbanstock3d.reconstruction.models import LidarQualityReport
from urbanstock3d.reconstruction.quality.lod_feasibility import (
    LodEvidenceAvailability,
    assess_lod_feasibility,
)
from urbanstock3d.reconstruction.quality.roof_complexity import (
    roof_complexity_report_from_dict,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lidar_quality", type=Path)
    parser.add_argument("roof_complexity", type=Path)
    parser.add_argument("--height-quality", type=Path)
    parser.add_argument("--height-grid", choices=("fine", "coarse"), default="fine")
    parser.add_argument("--reliable-footprint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ground-elevation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--representative-height", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_report(
    lidar_path: Path,
    complexity_path: Path,
    output: Path,
    *,
    availability: LodEvidenceAvailability,
    height_path: Path | None = None,
    height_grid: str = "fine",
) -> Path:
    lidar_payload: dict[str, Any] = json.loads(lidar_path.read_text(encoding="utf-8"))
    lidar_values = lidar_payload["quality"]
    lidar_values["warnings"] = tuple(lidar_values.get("warnings", ()))
    lidar = LidarQualityReport(**lidar_values)
    complexity_payload = json.loads(complexity_path.read_text(encoding="utf-8"))
    complexity = roof_complexity_report_from_dict(complexity_payload["result"])
    height = None
    if height_path is not None:
        height_payload = json.loads(height_path.read_text(encoding="utf-8"))
        height = HeightEvidenceMetrics(**height_payload[height_grid])
    report = assess_lod_feasibility(availability, lidar, complexity, height=height)
    payload = {
        "schema_version": 1,
        "kind": "pre_reconstruction_lod_feasibility",
        "building_id": lidar_payload.get("building_id"),
        "availability": {
            "reliable_footprint": availability.reliable_footprint,
            "ground_elevation": availability.ground_elevation,
            "representative_height": availability.representative_height,
        },
        "sources": {
            "lidar_quality": lidar_path.as_posix(),
            "roof_complexity": complexity_path.as_posix(),
            "height_quality": height_path.as_posix() if height_path else None,
            "height_grid": height_grid if height_path else None,
        },
        "result": report.to_dict(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    arguments = parse_arguments()
    availability = LodEvidenceAvailability(
        reliable_footprint=arguments.reliable_footprint,
        ground_elevation=arguments.ground_elevation,
        representative_height=arguments.representative_height,
    )
    output = build_report(
        arguments.lidar_quality,
        arguments.roof_complexity,
        arguments.output,
        availability=availability,
        height_path=arguments.height_quality,
        height_grid=arguments.height_grid,
    )
    print(f"Wrote LoD feasibility report to {output}")


if __name__ == "__main__":
    main()
