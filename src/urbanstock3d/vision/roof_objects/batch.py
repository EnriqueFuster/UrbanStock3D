"""Run the roof-object feasibility pipeline across selected buildings."""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from urbanstock3d.providers.pnoa_orthophoto import acquire_pnoa_crop
from urbanstock3d.vision.roof_objects.service import RoofObjectSegmenter, infer_roof_objects
from urbanstock3d.vision.roof_objects.vectorization import vectorize_and_clip_roof_objects


@dataclass(frozen=True)
class RoofObjectBatchEntry:
    building_id: str
    profile: str
    status: str
    raw_candidates: int
    retained_objects: int
    retained_classes: dict[str, int]
    error: str | None = None


def evaluate_selected_buildings(
    selection_path: Path,
    outputs_root: Path,
    segmenter_factory: Callable[[Path], RoofObjectSegmenter],
    *,
    acquire_missing: bool = True,
) -> dict[str, Any]:
    """Run the existing pipeline independently for every selected building."""
    selection: dict[str, Any] = json.loads(selection_path.read_text(encoding="utf-8"))
    entries: list[RoofObjectBatchEntry] = []
    for building in selection["buildings"]:
        building_id = str(building["building_id"])
        root = outputs_root / building_id
        footprint_path = root / "building.geojson"
        orthophoto_path = root / "orthophoto_crop.tif"
        try:
            if not footprint_path.exists():
                raise FileNotFoundError(f"Missing resolved footprint: {footprint_path}")
            if not orthophoto_path.exists():
                if not acquire_missing:
                    raise FileNotFoundError(f"Missing PNOA crop: {orthophoto_path}")
                acquire_pnoa_crop(footprint_path, orthophoto_path)

            result_root = root / "roof_objects"
            raw_path = result_root / "result.json"
            raw = infer_roof_objects(
                building_id,
                orthophoto_path,
                segmenter_factory(result_root / "masks"),
                raw_path,
            )
            vector_root = result_root / "vectorized"
            vectorized = vectorize_and_clip_roof_objects(raw, footprint_path, vector_root)
            (vector_root / "result.json").write_text(
                vectorized.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            classes: dict[str, int] = {}
            for instance in vectorized.objects:
                classes[instance.class_name] = classes.get(instance.class_name, 0) + 1
            entries.append(
                RoofObjectBatchEntry(
                    building_id=building_id,
                    profile=str(building["profile"]),
                    status="completed",
                    raw_candidates=len(raw.objects),
                    retained_objects=len(vectorized.objects),
                    retained_classes=classes,
                )
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            entries.append(
                RoofObjectBatchEntry(
                    building_id=building_id,
                    profile=str(building["profile"]),
                    status="failed",
                    raw_candidates=0,
                    retained_objects=0,
                    retained_classes={},
                    error=str(error),
                )
            )

    completed = [entry for entry in entries if entry.status == "completed"]
    return {
        "schema_version": 1,
        "kind": "pnoa_roof_object_feasibility",
        "summary": {
            "selected_buildings": len(entries),
            "completed_buildings": len(completed),
            "failed_buildings": len(entries) - len(completed),
            "raw_candidates": sum(entry.raw_candidates for entry in completed),
            "retained_objects": sum(entry.retained_objects for entry in completed),
        },
        "buildings": [asdict(entry) for entry in entries],
    }
