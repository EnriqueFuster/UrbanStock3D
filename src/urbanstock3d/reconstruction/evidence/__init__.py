"""Derive reconstruction-ready evidence without inventing missing observations."""

from urbanstock3d.reconstruction.evidence.height_metrics import (
    HeightEvidenceMetrics,
    HeightResolutionAgreement,
    compare_height_resolutions,
    evaluate_height_evidence,
)
from urbanstock3d.reconstruction.evidence.lidar_height import (
    ObservedHeightRaster,
    build_lidar_height_rasters,
    load_height_raster,
    save_height_raster,
)
from urbanstock3d.reconstruction.evidence.terrain import (
    NormalizedHeightRaster,
    build_normalized_height_raster,
    load_normalized_height_raster,
    save_normalized_height_raster,
)

__all__ = [
    "NormalizedHeightRaster",
    "ObservedHeightRaster",
    "HeightEvidenceMetrics",
    "HeightResolutionAgreement",
    "build_lidar_height_rasters",
    "build_normalized_height_raster",
    "compare_height_resolutions",
    "evaluate_height_evidence",
    "load_height_raster",
    "load_normalized_height_raster",
    "save_height_raster",
    "save_normalized_height_raster",
]
