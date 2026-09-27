import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from urbanstock3d.vision.roof_objects.annotation import build_pnoa_annotation_queue


def test_prioritizes_user_reported_pv_missed_by_model(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "buildings": [
                    {
                        "building_id": "ES.SDGC.BU.TEST",
                        "cadastral_root_id": "TEST",
                        "observed": {"photovoltaic_panels": "user_reported"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    feasibility = tmp_path / "feasibility.json"
    feasibility.write_text(
        json.dumps(
            {
                "buildings": [
                    {
                        "building_id": "ES.SDGC.BU.TEST",
                        "status": "completed",
                        "retained_objects": 2,
                        "retained_classes": {"skylight": 2},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "outputs" / "ES.SDGC.BU.TEST"
    output_root.mkdir(parents=True)
    with rasterio.open(
        output_root / "orthophoto_crop.tif",
        "w",
        driver="GTiff",
        width=3,
        height=2,
        count=3,
        dtype="uint8",
        crs="EPSG:25830",
        transform=from_origin(0, 1, 0.25, 0.25),
    ) as raster:
        raster.write(np.zeros((3, 2, 3), dtype=np.uint8))

    queue_root = tmp_path / "queue"
    manifest_path = build_pnoa_annotation_queue(
        selection, feasibility, tmp_path / "outputs", queue_root
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["summary"] == {
        "tasks": 1,
        "high_priority": 1,
        "medium_priority": 0,
        "low_priority": 0,
    }
    task = manifest["tasks"][0]
    assert task["review_reason"] == "user_reported_pv_not_detected_as_pv_panel"
    assert (queue_root / task["image"]).exists()
    label = json.loads((queue_root / task["labels"]).read_text(encoding="utf-8"))
    assert label["status"] == "pending"
    assert label["instances"] == []
