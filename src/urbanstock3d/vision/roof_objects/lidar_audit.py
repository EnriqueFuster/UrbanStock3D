"""Audit PNOA-LiDAR catalogue coverage for the selected building sample."""

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from urbanstock3d.providers.pnoa_lidar import (
    CnigLidarAsset,
    LidarGridCell,
    footprint_geometry_bbox_utm,
    required_grid_cells,
)

DiscoverCell = Callable[[LidarGridCell], CnigLidarAsset]


def audit_selected_lidar_catalog(
    selection_path: Path,
    outputs_root: Path,
    discover_cell: DiscoverCell,
    *,
    buffer_m: float = 25.0,
) -> dict[str, Any]:
    """Discover each unique LiDAR cell once and map it back to buildings."""
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    building_cells: dict[str, tuple[LidarGridCell, ...]] = {}
    unique_cells: dict[str, LidarGridCell] = {}

    for building in selection["buildings"]:
        building_id = str(building["building_id"])
        footprint_path = outputs_root / building_id / "building.geojson"
        footprint = json.loads(footprint_path.read_text(encoding="utf-8"))
        cells = required_grid_cells(
            footprint_geometry_bbox_utm(footprint["geometry"]),
            buffer_m=buffer_m,
        )
        building_cells[building_id] = cells
        unique_cells.update({cell.identifier: cell for cell in cells})

    assets: dict[str, CnigLidarAsset] = {}
    errors: dict[str, str] = {}
    for identifier, cell in sorted(unique_cells.items()):
        try:
            assets[identifier] = discover_cell(cell)
        except Exception as error:  # catalogue failures must remain visible per cell
            errors[identifier] = f"{type(error).__name__}: {error}"

    buildings = []
    for building in selection["buildings"]:
        building_id = str(building["building_id"])
        identifiers = [cell.identifier for cell in building_cells[building_id]]
        available = [identifier for identifier in identifiers if identifier in assets]
        missing = [identifier for identifier in identifiers if identifier in errors]
        if len(available) == len(identifiers):
            status = "catalog_verified"
        elif available:
            status = "partial_coverage"
        else:
            status = "catalog_unavailable"
        buildings.append(
            {
                "building_id": building_id,
                "cadastral_root_id": building["cadastral_root_id"],
                "grid_cells": identifiers,
                "asset_detail_urls": [assets[cell].detail_url for cell in available],
                "flight_years": sorted({assets[cell].flight_year for cell in available}),
                "catalog_density_points_m2": sorted(
                    {assets[cell].density_points_m2 for cell in available}
                ),
                "status": status,
                "failed_cells": missing,
            }
        )

    verified = sum(item["status"] == "catalog_verified" for item in buildings)
    partial = sum(item["status"] == "partial_coverage" for item in buildings)
    return {
        "schema_version": 1,
        "provider": "IGN-CNIG",
        "product": "PNOA-LiDAR third coverage",
        "scope": {
            "selection": str(selection_path),
            "buffer_m": buffer_m,
            "selected_buildings": len(buildings),
            "unique_grid_cells": len(unique_cells),
        },
        "summary": {
            "catalog_verified_buildings": verified,
            "partial_coverage_buildings": partial,
            "catalog_unavailable_buildings": len(buildings) - verified - partial,
            "verified_cells": len(assets),
            "failed_cells": len(errors),
        },
        "assets": [asdict(assets[cell]) for cell in sorted(assets)],
        "cell_errors": errors,
        "buildings": buildings,
        "limitations": [
            "Catalogue density is product metadata, not measured density over each roof.",
            "Point dimensions and class 67 require downloading and opening each LAZ.",
            "A user-reported photovoltaic candidate is not a verified annotation.",
        ],
    }
