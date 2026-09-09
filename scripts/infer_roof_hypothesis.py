"""Infer a coarse roof-form hypothesis from LiDAR and nDSM quality reports."""

import argparse
import json
from pathlib import Path
from typing import Any

from urbanstock3d.reconstruction.evidence import HeightEvidenceMetrics
from urbanstock3d.reconstruction.models import LidarQualityReport
from urbanstock3d.reconstruction.quality.roof_hypothesis import infer_roof_hypothesis


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lidar_quality", type=Path)
    parser.add_argument("height_quality", type=Path)
    parser.add_argument("--height-grid", choices=("fine", "coarse"), default="fine")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_report(
    lidar_quality_path: Path,
    height_quality_path: Path,
    output: Path,
    *,
    height_grid: str = "fine",
) -> Path:
    lidar_payload: dict[str, Any] = json.loads(lidar_quality_path.read_text(encoding="utf-8"))
    height_payload: dict[str, Any] = json.loads(height_quality_path.read_text(encoding="utf-8"))
    lidar_values = lidar_payload["quality"]
    lidar_values["warnings"] = tuple(lidar_values.get("warnings", ()))
    lidar = LidarQualityReport(**lidar_values)
    height = HeightEvidenceMetrics(**height_payload[height_grid])
    hypothesis = infer_roof_hypothesis(lidar, height)
    payload = {
        "schema_version": 1,
        "kind": "pre_reconstruction_roof_hypothesis",
        "building_id": lidar_payload.get("building_id"),
        "sources": {
            "lidar_quality": lidar_quality_path.as_posix(),
            "height_quality": height_quality_path.as_posix(),
            "height_grid": height_grid,
        },
        "result": hypothesis.to_dict(),
        "interpretation": "routing evidence only; not reconstructed roof topology",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    arguments = parse_arguments()
    report = build_report(
        arguments.lidar_quality,
        arguments.height_quality,
        arguments.output,
        height_grid=arguments.height_grid,
    )
    print(f"Wrote roof hypothesis to {report}")


if __name__ == "__main__":
    main()
