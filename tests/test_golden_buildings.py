import json
from pathlib import Path
from typing import Any


def test_golden_building_selection_is_complete_and_unique() -> None:
    registry_path = Path("data/golden_buildings.json")
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
        assert building["coverage"]["catastro"] == "verified"
        assert building["coverage"]["pnoa"] == "verified"
