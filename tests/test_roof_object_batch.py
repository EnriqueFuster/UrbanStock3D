import json
from pathlib import Path

from urbanstock3d.vision.roof_objects.batch import evaluate_selected_buildings


class EmptySegmenter:
    model_name = "empty"
    model_version = "1"

    def predict(self, image: Path) -> tuple[()]:
        return ()


def test_batch_records_completed_and_missing_buildings(tmp_path: Path) -> None:
    selection = tmp_path / "selected.json"
    selection.write_text(
        json.dumps(
            {
                "buildings": [
                    {"building_id": "ready", "profile": "available inputs"},
                    {"building_id": "missing", "profile": "missing inputs"},
                ]
            }
        ),
        encoding="utf-8",
    )
    ready_root = tmp_path / "outputs" / "ready"
    ready_root.mkdir(parents=True)
    (ready_root / "building.geojson").write_text("{}", encoding="utf-8")
    (ready_root / "orthophoto_crop.tif").touch()

    # Empty predictions do not read raster contents during vectorization.
    report = evaluate_selected_buildings(
        selection,
        tmp_path / "outputs",
        lambda _: EmptySegmenter(),
        acquire_missing=False,
    )

    assert report["summary"] == {
        "selected_buildings": 2,
        "completed_buildings": 1,
        "failed_buildings": 1,
        "raw_candidates": 0,
        "retained_objects": 0,
    }
    assert report["buildings"][0]["status"] == "completed"
    assert report["buildings"][1]["status"] == "failed"
