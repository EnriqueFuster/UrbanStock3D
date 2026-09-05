import json
from pathlib import Path

import pytest

from urbanstock3d.domain.cadastre import (
    CadastralBuilding,
    CadastralBuildingPart,
    CadastralIdentity,
    Footprint2D,
)
from urbanstock3d.outputs.cadastre import write_cadastral_artifacts


@pytest.fixture
def building() -> CadastralBuilding:
    footprint = Footprint2D(
        crs="EPSG:25830",
        rings=(
            (
                (724385.52, 4372941.57),
                (724406.57, 4372941.57),
                (724406.57, 4372966.60),
                (724385.52, 4372941.57),
            ),
        ),
    )
    return CadastralBuilding(
        identity=CadastralIdentity(
            cadastral_root_id="4531917YJ2743B",
            local_id="4531917YJ2743B",
            namespace="ES.SDGC.BU",
        ),
        footprint=footprint,
        parts=(
            CadastralBuildingPart(
                local_id="4531917YJ2743B_part1",
                floors_above_ground=10,
                footprint=footprint,
            ),
        ),
        current_use="1_residential",
    )


def test_writer_creates_expected_artifacts(
    building: CadastralBuilding,
    tmp_path: Path,
) -> None:
    building_dir = write_cadastral_artifacts(building, tmp_path)

    assert building_dir.name == "ES.SDGC.BU.4531917YJ2743B"
    assert {path.name for path in building_dir.iterdir()} == {
        "identity.json",
        "building.geojson",
        "building_parts.geojson",
    }


def test_geojson_coordinates_are_exported_in_wgs84(
    building: CadastralBuilding,
    tmp_path: Path,
) -> None:
    building_dir = write_cadastral_artifacts(building, tmp_path)
    geojson = json.loads((building_dir / "building.geojson").read_text(encoding="utf-8"))

    longitude, latitude = geojson["geometry"]["coordinates"][0][0]
    assert longitude == pytest.approx(-0.3913, abs=0.001)
    assert latitude == pytest.approx(39.4767, abs=0.001)
    assert geojson["properties"]["current_use"] == "1_residential"


def test_parts_are_written_as_individual_features(
    building: CadastralBuilding,
    tmp_path: Path,
) -> None:
    building_dir = write_cadastral_artifacts(building, tmp_path)
    collection = json.loads((building_dir / "building_parts.geojson").read_text(encoding="utf-8"))

    assert collection["type"] == "FeatureCollection"
    assert len(collection["features"]) == 1
    assert collection["features"][0]["properties"]["floors_above_ground"] == 10
