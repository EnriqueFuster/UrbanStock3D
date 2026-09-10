"""Independent validation of reconstructed building geometry."""

from urbanstock3d.reconstruction.validation.cityjson import (
    CityJsonQualityParameters,
    CityJsonQualityReport,
    validate_cityjsonseq,
)
from urbanstock3d.reconstruction.validation.lidar_fit import (
    LidarFitParameters,
    LidarFitReport,
    LidarResiduals,
    assess_cityjson_lidar_fit,
    compute_cityjson_lidar_residuals,
)
from urbanstock3d.reconstruction.validation.obj import (
    ObjQualityReport,
    assess_obj_lidar_fit,
    compute_obj_lidar_residuals,
    read_obj,
    validate_obj,
)
from urbanstock3d.reconstruction.validation.quality import (
    QualityConfidence,
    ReconstructionQualityParameters,
    ReconstructionQualityReport,
    evaluate_reconstruction_quality,
)
from urbanstock3d.reconstruction.validation.roof_observations import (
    RoofObservationParameters,
    RoofObservationReport,
    select_roof_observations,
)

__all__ = [
    "CityJsonQualityParameters",
    "CityJsonQualityReport",
    "LidarFitParameters",
    "LidarFitReport",
    "LidarResiduals",
    "ObjQualityReport",
    "QualityConfidence",
    "ReconstructionQualityParameters",
    "ReconstructionQualityReport",
    "RoofObservationParameters",
    "RoofObservationReport",
    "assess_cityjson_lidar_fit",
    "assess_obj_lidar_fit",
    "compute_cityjson_lidar_residuals",
    "compute_obj_lidar_residuals",
    "evaluate_reconstruction_quality",
    "read_obj",
    "select_roof_observations",
    "validate_cityjsonseq",
    "validate_obj",
]
