import json
from pathlib import Path

from urbanstock3d.providers.pnoa_lidar import CnigLidarAsset, LidarGridCell
from urbanstock3d.vision.roof_objects.lidar_audit import audit_selected_lidar_catalog


def test_audit_reuses_grid_assets_across_buildings(tmp_path: Path) -> None:
    selection = {
        "buildings": [
            {"building_id": "ES.SDGC.BU.A", "cadastral_root_id": "A"},
            {"building_id": "ES.SDGC.BU.B", "cadastral_root_id": "B"},
        ]
    }
    selection_path = tmp_path / "selected.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    geometry = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-0.3912, 39.4770],
                    [-0.3911, 39.4770],
                    [-0.3911, 39.4771],
                    [-0.3912, 39.4770],
                ]
            ],
        },
    }
    for building in selection["buildings"]:
        directory = tmp_path / "outputs" / building["building_id"]
        directory.mkdir(parents=True)
        (directory / "building.geojson").write_text(json.dumps(geometry), encoding="utf-8")

    calls: list[str] = []

    def discover(cell: LidarGridCell) -> CnigLidarAsset:
        calls.append(cell.identifier)
        return CnigLidarAsset(
            detail_url="https://example.test/asset",
            sequential_id="1",
            filename=f"{cell.identifier}.laz",
            flight_year=2023,
            density_points_m2=5.0,
            size_mb=100.0,
            grid_cell=cell.identifier,
            processing_level="NPC03",
        )

    report = audit_selected_lidar_catalog(
        selection_path,
        tmp_path / "outputs",
        discover,
        buffer_m=0,
    )

    assert len(calls) == 1
    assert report["summary"] == {
        "catalog_verified_buildings": 2,
        "partial_coverage_buildings": 0,
        "catalog_unavailable_buildings": 0,
        "verified_cells": 1,
        "failed_cells": 0,
    }
    assert all(item["status"] == "catalog_verified" for item in report["buildings"])
