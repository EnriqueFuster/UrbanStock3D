import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.transform import from_origin

from urbanstock3d.vision.models import AssessmentStatus
from urbanstock3d.vision.roof_objects.service import (
    UltralyticsRoofObjectSegmenter,
    infer_roof_objects,
)


class FakeModel:
    def predict(self, image: np.ndarray, **_: object) -> list[SimpleNamespace]:
        assert image.shape == (4, 5, 3)
        mask = np.zeros((1, 4, 5), dtype=np.float32)
        mask[0, 1:3, 2:4] = 1
        return [
            SimpleNamespace(
                masks=SimpleNamespace(data=mask),
                boxes=SimpleNamespace(
                    cls=np.asarray([0], dtype=np.float32),
                    conf=np.asarray([0.9], dtype=np.float32),
                ),
                names={0: "pv_panel"},
            )
        ]


def test_inference_writes_georeferenced_masks_and_result(tmp_path: Path) -> None:
    image = tmp_path / "orthophoto_crop.tif"
    with rasterio.open(
        image,
        "w",
        driver="GTiff",
        width=5,
        height=4,
        count=3,
        dtype="uint8",
        crs="EPSG:25830",
        transform=from_origin(724000, 4373000, 0.25, 0.25),
    ) as raster:
        raster.write(np.zeros((3, 4, 5), dtype=np.uint8))

    weights = tmp_path / "baseline" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.touch()
    segmenter = UltralyticsRoofObjectSegmenter(
        weights,
        tmp_path / "masks",
        model=FakeModel(),
    )
    result_path = tmp_path / "result.json"
    result = infer_roof_objects("ES.SDGC.BU.TEST", image, segmenter, result_path)

    assert result.status is AssessmentStatus.EXPERIMENTAL
    assert len(result.objects) == 1
    instance = result.objects[0]
    assert instance.class_name == "pv_panel"
    assert instance.pixel_area == 4
    assert instance.area_m2 == 0.25
    with rasterio.open(instance.mask_artifact) as mask:
        assert mask.crs.to_string() == "EPSG:25830"
        assert int(mask.read(1).sum()) == 4
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["building_id"] == "ES.SDGC.BU.TEST"


def test_inference_returns_empty_tuple_without_masks(tmp_path: Path) -> None:
    class EmptyModel:
        def predict(self, *_: object, **__: object) -> list[SimpleNamespace]:
            return [SimpleNamespace(masks=None, boxes=None)]

    image = tmp_path / "image.tif"
    with rasterio.open(
        image,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=3,
        dtype="uint8",
        crs="EPSG:25830",
        transform=from_origin(0, 1, 0.25, 0.25),
    ) as raster:
        raster.write(np.zeros((3, 2, 2), dtype=np.uint8))
    weights = tmp_path / "run" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.touch()

    segmenter = UltralyticsRoofObjectSegmenter(weights, tmp_path / "masks", model=EmptyModel())
    assert segmenter.predict(image) == ()
