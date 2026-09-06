"""Record the PNOA-LiDAR grid cells required by one building."""

import argparse
import json
from pathlib import Path
from typing import Any

from urbanstock3d.providers.pnoa_lidar import (
    PENINSULA_UTM,
    cnig_search_url,
    footprint_bbox_utm,
    required_grid_cells,
)


def parse_arguments() -> argparse.Namespace:
    """Read the building artifact and discovery buffer."""
    parser = argparse.ArgumentParser(description="Locate PNOA-LiDAR grid cells.")
    parser.add_argument("building_geojson", type=Path)
    parser.add_argument("--buffer-m", type=float, default=25.0)
    return parser.parse_args()


def audit_lidar_discovery(building_geojson: Path, *, buffer_m: float) -> Path:
    """Write a reproducible LiDAR grid-discovery report."""
    geojson: dict[str, Any] = json.loads(building_geojson.read_text(encoding="utf-8"))
    geometry = geojson["geometry"]
    if geometry["type"] != "Polygon":
        raise ValueError("LiDAR discovery currently supports Polygon buildings only")

    coordinates = geometry["coordinates"]
    projected_bbox = footprint_bbox_utm(coordinates)
    cells = required_grid_cells(projected_bbox, buffer_m=buffer_m)
    positions = [position for ring in coordinates for position in ring]
    xs = [position[0] for position in positions]
    ys = [position[1] for position in positions]
    geographic_bbox = min(xs), min(ys), max(xs), max(ys)

    report = {
        "provider": "IGN-CNIG",
        "product": "PNOA-LiDAR third coverage (2022-2025)",
        "status": "catalog_verification_required",
        "building_id": geojson["id"],
        "request": {
            "buffer_m": buffer_m,
            "crs": PENINSULA_UTM,
            "building_bbox": projected_bbox,
        },
        "grid_cells": [cell.identifier for cell in cells],
        "official_catalog_search_url": cnig_search_url(geographic_bbox),
        "source_assets": [],
        "warnings": [
            "CNIG does not currently document a stable LAZ search API; source asset URLs "
            "must be verified in the official catalogue before download."
        ],
    }
    report_path = building_geojson.with_name("lidar_discovery.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    """Run LiDAR grid discovery from the command line."""
    args = parse_arguments()
    report_path = audit_lidar_discovery(args.building_geojson, buffer_m=args.buffer_m)
    print(f"Wrote LiDAR discovery audit to {report_path}")


if __name__ == "__main__":
    main()
