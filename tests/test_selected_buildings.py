import json
from pathlib import Path
from typing import Any


def test_selected_buildings_are_complete_and_unique() -> None:
    registry_path = Path("data/selected_buildings.json")
    registry: dict[str, Any] = json.loads(registry_path.read_text(encoding="utf-8"))
    buildings = registry["buildings"]

    building_ids = [building["building_id"] for building in buildings]
    profiles = [building["profile"] for building in buildings]

    assert registry["schema_version"] == 1
    assert len(buildings) == 5
    assert len(building_ids) == len(set(building_ids))
    assert len(profiles) == len(set(profiles))

    for building in buildings:
        assert building["building_id"].endswith(building["cadastral_root_id"])
        assert building["observed"]["part_count"] > 0
        assert building["observed"]["lidar_grid_cells"]
        assert building["coverage"]["catastro"] == "verified"
        assert building["coverage"]["pnoa"] == "verified"
        assert building["coverage"]["facade"] in {
            "verified",
            "verified_with_warning",
        }
        if building["coverage"]["lidar"] == "asset_verified":
            asset_urls = building["observed"]["lidar_asset_detail_urls"]
            assert len(asset_urls) == len(building["observed"]["lidar_grid_cells"])
            assert all(url.startswith("https://centrodedescargas.cnig.es/") for url in asset_urls)
