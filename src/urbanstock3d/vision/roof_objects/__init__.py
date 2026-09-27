"""Roof-object instance-segmentation task boundary."""

from urbanstock3d.vision.roof_objects.annotation import build_pnoa_annotation_queue
from urbanstock3d.vision.roof_objects.batch import evaluate_selected_buildings
from urbanstock3d.vision.roof_objects.datasets import (
    audit_rid2,
    inspect_zip_archive,
    load_rid2_source,
)
from urbanstock3d.vision.roof_objects.service import (
    RoofObjectSegmenter,
    UltralyticsRoofObjectSegmenter,
    infer_roof_objects,
)
from urbanstock3d.vision.roof_objects.vectorization import vectorize_and_clip_roof_objects

__all__ = [
    "RoofObjectSegmenter",
    "UltralyticsRoofObjectSegmenter",
    "audit_rid2",
    "build_pnoa_annotation_queue",
    "evaluate_selected_buildings",
    "infer_roof_objects",
    "inspect_zip_archive",
    "load_rid2_source",
    "vectorize_and_clip_roof_objects",
]
