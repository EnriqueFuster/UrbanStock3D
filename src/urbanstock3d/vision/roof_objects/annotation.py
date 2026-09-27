"""Build a human-review queue from PNOA feasibility results."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image

from urbanstock3d.vision.common import load_taxonomy


def _priority_and_reason(building: dict[str, Any], evaluation: dict[str, Any]) -> tuple[str, str]:
    photovoltaic_reported = building["observed"].get("photovoltaic_panels") == "user_reported"
    predicted_pv = evaluation["retained_classes"].get("pv_panel", 0) > 0
    if photovoltaic_reported and not predicted_pv:
        return "high", "user_reported_pv_not_detected_as_pv_panel"
    if evaluation["retained_objects"] > 0:
        return "medium", "unverified_retained_predictions"
    return "low", "no_retained_predictions"


def build_pnoa_annotation_queue(
    selection_path: Path,
    feasibility_path: Path,
    outputs_root: Path,
    queue_root: Path,
) -> Path:
    """Create raw review PNGs, empty label templates, and a prioritized manifest."""
    selection: dict[str, Any] = json.loads(selection_path.read_text(encoding="utf-8"))
    feasibility: dict[str, Any] = json.loads(feasibility_path.read_text(encoding="utf-8"))
    evaluations = {
        item["building_id"]: item
        for item in feasibility["buildings"]
        if item["status"] == "completed"
    }
    taxonomy = load_taxonomy(Path("config/vision/taxonomy.yaml"))
    images_dir = queue_root / "images"
    labels_dir = queue_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[dict[str, Any]] = []
    for building in selection["buildings"]:
        building_id = str(building["building_id"])
        evaluation = evaluations.get(building_id)
        if evaluation is None:
            continue
        source_raster = outputs_root / building_id / "orthophoto_crop.tif"
        if not source_raster.exists():
            continue
        cadastral_root = str(building["cadastral_root_id"])
        image_path = images_dir / f"{cadastral_root}.png"
        with rasterio.open(source_raster) as raster:
            image = np.moveaxis(raster.read((1, 2, 3)), 0, 2)
        Image.fromarray(image).save(image_path)

        priority, reason = _priority_and_reason(building, evaluation)
        label_path = labels_dir / f"{cadastral_root}.json"
        if not label_path.exists():
            label_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "building_id": building_id,
                        "image": image_path.relative_to(queue_root).as_posix(),
                        "status": "pending",
                        "instances": [],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        tasks.append(
            {
                "task_id": cadastral_root,
                "building_id": building_id,
                "priority": priority,
                "review_reason": reason,
                "image": image_path.relative_to(queue_root).as_posix(),
                "labels": label_path.relative_to(queue_root).as_posix(),
                "user_observation": building["observed"].get("photovoltaic_panels"),
                "model_predictions": evaluation["retained_classes"],
                "annotation_status": json.loads(label_path.read_text(encoding="utf-8"))["status"],
            }
        )

    priority_order = {"high": 0, "medium": 1, "low": 2}
    tasks.sort(key=lambda task: (priority_order[task["priority"]], task["task_id"]))
    manifest_path = queue_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pnoa_roof_object_annotation_queue",
                "taxonomy_version": taxonomy.version,
                "classes": taxonomy.roof_objects,
                "coordinate_format": "pixel_polygon_xy",
                "summary": {
                    "tasks": len(tasks),
                    "high_priority": sum(task["priority"] == "high" for task in tasks),
                    "medium_priority": sum(task["priority"] == "medium" for task in tasks),
                    "low_priority": sum(task["priority"] == "low" for task in tasks),
                },
                "tasks": tasks,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path
