"""Combine footprint, RANSAC, and optional nDSM evidence into roof complexity."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from urbanstock3d.providers.pnoa_lidar import footprint_polygons_utm
from urbanstock3d.reconstruction.evidence import (
    HeightEvidenceMetrics,
    load_normalized_height_raster,
)
from urbanstock3d.reconstruction.quality.ransac import roof_plane_evidence_from_dict
from urbanstock3d.reconstruction.quality.roof_complexity import (
    RoofComplexityParameters,
    assess_roof_complexity,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("building_geojson", type=Path)
    parser.add_argument("plane_evidence", type=Path)
    parser.add_argument("--height-quality", type=Path)
    parser.add_argument("--height-grid", choices=("fine", "coarse"), default="fine")
    parser.add_argument("--ndsm", type=Path, help="Optional nDSM used for robust height range.")
    parser.add_argument("--building-parts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_report(
    building_path: Path,
    plane_path: Path,
    output: Path,
    *,
    height_quality_path: Path | None = None,
    height_grid: str = "fine",
    ndsm_path: Path | None = None,
    building_parts_path: Path | None = None,
) -> Path:
    building: dict[str, Any] = json.loads(building_path.read_text(encoding="utf-8"))
    plane_payload: dict[str, Any] = json.loads(plane_path.read_text(encoding="utf-8"))
    planes = roof_plane_evidence_from_dict(plane_payload["evidence"])
    height = None
    if height_quality_path is not None:
        height_payload = json.loads(height_quality_path.read_text(encoding="utf-8"))
        height = HeightEvidenceMetrics(**height_payload[height_grid])
    building_part_count = None
    if building_parts_path is not None:
        parts_payload = json.loads(building_parts_path.read_text(encoding="utf-8"))
        building_part_count = len(parts_payload["features"])
    height_range_m = None
    if ndsm_path is not None:
        ndsm = load_normalized_height_raster(ndsm_path)
        values = ndsm.height_m[ndsm.valid_mask]
        height_range_m = float(np.percentile(values, 95) - np.percentile(values, 5))
    parameters = RoofComplexityParameters()
    report = assess_roof_complexity(
        footprint_polygons_utm(building["geometry"]),
        planes,
        height=height,
        building_part_count=building_part_count,
        height_range_m=height_range_m,
        parameters=parameters,
    )
    payload = {
        "schema_version": 1,
        "kind": "pre_reconstruction_roof_complexity",
        "building_id": building["id"],
        "sources": {
            "footprint": building_path.as_posix(),
            "planes": plane_path.as_posix(),
            "height_quality": height_quality_path.as_posix() if height_quality_path else None,
            "ndsm": ndsm_path.as_posix() if ndsm_path else None,
            "building_parts": building_parts_path.as_posix() if building_parts_path else None,
        },
        "parameters": asdict(parameters),
        "result": report.to_dict(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    arguments = parse_arguments()
    output = build_report(
        arguments.building_geojson,
        arguments.plane_evidence,
        arguments.output,
        height_quality_path=arguments.height_quality,
        height_grid=arguments.height_grid,
        ndsm_path=arguments.ndsm,
        building_parts_path=arguments.building_parts,
    )
    print(f"Wrote roof complexity report to {output}")


if __name__ == "__main__":
    main()
