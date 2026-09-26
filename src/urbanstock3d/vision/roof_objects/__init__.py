"""Roof-object instance-segmentation task boundary."""

from urbanstock3d.vision.roof_objects.datasets import (
    audit_rid2,
    inspect_zip_archive,
    load_rid2_source,
)
from urbanstock3d.vision.roof_objects.service import RoofObjectSegmenter

__all__ = ["RoofObjectSegmenter", "audit_rid2", "inspect_zip_archive", "load_rid2_source"]
