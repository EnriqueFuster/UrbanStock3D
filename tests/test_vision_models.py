import pytest
from pydantic import ValidationError

from urbanstock3d.vision import GeoPolygon, RoofObjectInstance
from urbanstock3d.vision.models import ModelProvenance


def test_roof_object_retains_georeferenced_geometry_and_provenance() -> None:
    instance = RoofObjectInstance(
        object_id="pv-001",
        class_name="pv_panel",
        confidence=0.92,
        mask_artifact="masks/pv-001.tif",
        geometry_map=GeoPolygon(
            coordinates=(
                (
                    (724390.0, 4372950.0),
                    (724392.0, 4372950.0),
                    (724392.0, 4372951.0),
                    (724390.0, 4372950.0),
                ),
            ),
            crs="EPSG:25830",
        ),
        pixel_area=200,
        area_m2=2.0,
        provenance=ModelProvenance(
            model_name="urbanstock-roof-objects",
            model_version="0.1.0",
            taxonomy_version="1.0",
            source_image="pnoa_crop.tif",
        ),
    )

    assert instance.geometry_map is not None
    assert instance.geometry_map.crs == "EPSG:25830"


def test_rejects_open_map_polygon_ring() -> None:
    with pytest.raises(ValidationError, match="closed"):
        GeoPolygon(
            coordinates=(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),),
            crs="EPSG:25830",
        )
