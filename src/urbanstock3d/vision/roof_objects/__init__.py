"""Roof-object instance-segmentation task boundary."""

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

__all__ = [
    "RoofObjectSegmenter",
    "UltralyticsRoofObjectSegmenter",
    "audit_rid2",
    "infer_roof_objects",
    "inspect_zip_archive",
    "load_rid2_source",
]
