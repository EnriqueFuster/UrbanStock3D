"""Final decision from currently implemented reconstruction quality gates."""

from dataclasses import asdict, dataclass
from enum import StrEnum

from urbanstock3d.reconstruction.validation.cityjson import CityJsonQualityReport
from urbanstock3d.reconstruction.validation.lidar_fit import LidarFitReport


class QualityConfidence(StrEnum):
    """Engineering confidence label, not a calibrated probability."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass(frozen=True)
class ReconstructionQualityParameters:
    """Explicit initial acceptance thresholds for LiDAR roof fit."""

    maximum_rmse_m: float = 1.0
    maximum_p95_m: float = 2.0
    minimum_within_050m_ratio: float = 0.75
    high_confidence_rmse_m: float = 0.30
    high_confidence_p95_m: float = 0.60
    high_confidence_within_050m_ratio: float = 0.90

    def __post_init__(self) -> None:
        distances = (
            self.maximum_rmse_m,
            self.maximum_p95_m,
            self.high_confidence_rmse_m,
            self.high_confidence_p95_m,
        )
        if any(value <= 0 for value in distances):
            raise ValueError("Quality distance thresholds must be positive")
        ratios = (self.minimum_within_050m_ratio, self.high_confidence_within_050m_ratio)
        if any(not 0 <= value <= 1 for value in ratios):
            raise ValueError("Quality support ratios must lie between zero and one")


@dataclass(frozen=True)
class ReconstructionQualityReport:
    """Backend-independent accept/reject decision and its causes."""

    geometry_valid: bool
    plausibility_pass: bool
    point_surface_rmse_m: float
    point_surface_p95_m: float
    within_050m_ratio: float
    accepted: bool
    confidence_class: QualityConfidence
    failures: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["confidence_class"] = self.confidence_class.value
        return payload


def evaluate_reconstruction_quality(
    geometry: CityJsonQualityReport,
    lidar_fit: LidarFitReport,
    *,
    parameters: ReconstructionQualityParameters | None = None,
) -> ReconstructionQualityReport:
    """Apply structural and metric gates without backend-specific success flags."""
    parameters = parameters or ReconstructionQualityParameters()
    failures = list(geometry.failures)
    if lidar_fit.distance_rmse_m > parameters.maximum_rmse_m:
        failures.append("point-to-surface RMSE exceeds acceptance threshold")
    if lidar_fit.distance_p95_m > parameters.maximum_p95_m:
        failures.append("point-to-surface P95 exceeds acceptance threshold")
    if lidar_fit.within_050m_ratio < parameters.minimum_within_050m_ratio:
        failures.append("insufficient roof points lie within 0.5 m of the model")
    accepted = not failures
    high_confidence = (
        accepted
        and lidar_fit.distance_rmse_m <= parameters.high_confidence_rmse_m
        and lidar_fit.distance_p95_m <= parameters.high_confidence_p95_m
        and lidar_fit.within_050m_ratio >= parameters.high_confidence_within_050m_ratio
    )
    confidence = (
        QualityConfidence.HIGH
        if high_confidence
        else QualityConfidence.MEDIUM
        if accepted
        else QualityConfidence.LOW
    )
    return ReconstructionQualityReport(
        geometry_valid=geometry.geometry_valid,
        plausibility_pass=geometry.plausibility_pass,
        point_surface_rmse_m=lidar_fit.distance_rmse_m,
        point_surface_p95_m=lidar_fit.distance_p95_m,
        within_050m_ratio=lidar_fit.within_050m_ratio,
        accepted=accepted,
        confidence_class=confidence,
        failures=tuple(failures),
        warnings=(*geometry.warnings, *lidar_fit.warnings),
    )
