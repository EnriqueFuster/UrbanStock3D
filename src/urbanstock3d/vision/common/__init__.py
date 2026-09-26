"""Shared configuration and dataset contracts for vision tasks."""

from urbanstock3d.vision.common.config import RoofObjectsConfig, load_roof_objects_config
from urbanstock3d.vision.common.taxonomy import CanonicalTaxonomy, load_taxonomy

__all__ = [
    "CanonicalTaxonomy",
    "RoofObjectsConfig",
    "load_roof_objects_config",
    "load_taxonomy",
]
