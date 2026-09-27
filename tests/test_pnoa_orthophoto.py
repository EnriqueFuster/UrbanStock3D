import json
from io import BytesIO
from pathlib import Path

import httpx
import pytest
import rasterio
from PIL import Image

from urbanstock3d.providers.pnoa_orthophoto import (
    acquire_pnoa_crop,
    image_dimensions,
    projected_building_bbox,
)


def test_projected_bbox_supports_multipolygon() -> None:
    geometry = {
        "type": "MultiPolygon",
        "coordinates": [
            [[[-0.3911, 39.4770], [-0.3910, 39.4770], [-0.3911, 39.4770]]],
            [[[-0.3909, 39.4769], [-0.3908, 39.4769], [-0.3909, 39.4769]]],
        ],
    }
    base = projected_building_bbox(geometry, buffer_m=0)
    buffered = projected_building_bbox(geometry, buffer_m=10)

    assert buffered == pytest.approx((base[0] - 10, base[1] - 10, base[2] + 10, base[3] + 10))


def test_image_dimensions_rejects_oversized_request() -> None:
    with pytest.raises(ValueError, match="size limit"):
        image_dimensions((0, 0, 2000, 2000), 0.25)


def test_acquires_georeferenced_rgb_crop(tmp_path: Path) -> None:
    building = tmp_path / "building.geojson"
    building.write_text(
        json.dumps(
            {
                "type": "Feature",
                "id": "ES.SDGC.BU.TEST",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-0.39110, 39.47700],
                            [-0.39109, 39.47700],
                            [-0.39109, 39.47699],
                            [-0.39110, 39.47700],
                        ]
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    def respond(request: httpx.Request) -> httpx.Response:
        width = int(request.url.params["WIDTH"])
        height = int(request.url.params["HEIGHT"])
        buffer = BytesIO()
        Image.new("RGB", (width, height), (20, 40, 60)).save(buffer, format="JPEG")
        return httpx.Response(200, content=buffer.getvalue(), request=request)

    output = tmp_path / "orthophoto_crop.tif"
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        metadata_path = acquire_pnoa_crop(
            building,
            output,
            buffer_m=1,
            pixel_size_m=0.25,
            client=client,
        )

    with rasterio.open(output) as raster:
        assert raster.crs.to_string() == "EPSG:25830"
        assert raster.count == 3
        assert raster.width > 0
        assert raster.height > 0
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["building_id"] == "ES.SDGC.BU.TEST"
    assert metadata["raster"]["pixel_size_m"] == 0.25
