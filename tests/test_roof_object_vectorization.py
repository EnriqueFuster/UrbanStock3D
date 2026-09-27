import json
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from urbanstock3d.vision.models import (
    AssessmentStatus,
    ModelProvenance,
    RoofObjectInstance,
    RoofVisionResult,
)
from urbanstock3d.vision.roof_objects.vectorization import vectorize_and_clip_roof_objects


def test_vectorizes_mask_and_clips_it_to_footprint(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.tif"
    transform = from_origin(724000, 4373000, 1, 1)
    mask = np.ones((4, 4), dtype=np.uint8)
    with rasterio.open(
        mask_path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="uint8",
        crs="EPSG:25830",
        transform=transform,
    ) as raster:
        raster.write(mask, 1)

    to_wgs84 = Transformer.from_crs("EPSG:25830", "EPSG:4326", always_xy=True)
    ring_utm = [(724000, 4373000), (724002, 4373000), (724002, 4372996), (724000, 4372996)]
    ring = [to_wgs84.transform(x, y) for x, y in ring_utm]
    ring.append(ring[0])
    building = tmp_path / "building.geojson"
    building.write_text(
        json.dumps(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {},
            }
        ),
        encoding="utf-8",
    )
    result = RoofVisionResult(
        building_id="ES.SDGC.BU.TEST",
        status=AssessmentStatus.EXPERIMENTAL,
        objects=(
            RoofObjectInstance(
                object_id="pv_panel-001",
                class_name="pv_panel",
                confidence=0.9,
                mask_artifact=mask_path.as_posix(),
                pixel_area=16,
                area_m2=16,
                provenance=ModelProvenance(
                    model_name="test",
                    model_version="1",
                    taxonomy_version="1.0",
                    source_image="orthophoto.tif",
                ),
                quality_flags=("PNOA_DOMAIN_NOT_VALIDATED", "NOT_CLIPPED_TO_FOOTPRINT"),
            ),
        ),
    )

    vectorized = vectorize_and_clip_roof_objects(result, building, tmp_path / "vectorized")

    assert len(vectorized.objects) == 1
    instance = vectorized.objects[0]
    assert instance.pixel_area == 8
    assert instance.area_m2 == 8
    assert instance.geometry_map is not None
    assert instance.geometry_map.crs == "EPSG:25830"
    assert "NOT_CLIPPED_TO_FOOTPRINT" not in instance.quality_flags
    geojson = json.loads((tmp_path / "vectorized" / "objects.geojson").read_text())
    assert len(geojson["features"]) == 1
