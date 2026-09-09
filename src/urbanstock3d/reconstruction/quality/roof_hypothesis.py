"""Infer a cautious roof-form hypothesis from cheap measured evidence."""

from dataclasses import asdict, dataclass
from enum import StrEnum

import numpy as np

from urbanstock3d.reconstruction.evidence import HeightEvidenceMetrics
from urbanstock3d.reconstruction.models import LidarQualityReport


class RoofFormHypothesis(StrEnum):
    """Coarse roof forms used before choosing a reconstruction backend."""

    FLAT = "flat"
    PITCHED = "pitched"
    COMPOUND = "compound"
    UNCERTAIN = "uncertain"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class RoofHypothesisParameters:
    """Provisional, explicit thresholds to calibrate on selected buildings."""

    minimum_density_per_m2: float = 1.0
    minimum_valid_ratio: float = 0.55
    minimum_planar_support: float = 0.60
    flat_gradient_p95_max: float = 0.25
    pitched_gradient_p50_min: float = 0.08
    compound_gradient_p95_min: float = 1.0
    flat_discontinuity_ratio_max: float = 0.05
    compound_discontinuity_ratio_min: float = 0.15

    def __post_init__(self) -> None:
        positive = (
            self.minimum_density_per_m2,
            self.flat_gradient_p95_max,
            self.pitched_gradient_p50_min,
            self.compound_gradient_p95_min,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Density and gradient thresholds must be positive")
        ratios = (
            self.minimum_valid_ratio,
            self.minimum_planar_support,
            self.flat_discontinuity_ratio_max,
            self.compound_discontinuity_ratio_min,
        )
        if any(not 0 <= value <= 1 for value in ratios):
            raise ValueError("Ratio thresholds must lie between zero and one")


@dataclass(frozen=True)
class RoofHypothesisReport:
    """A derived hypothesis that keeps its evidence and reasons visible."""

    hypothesis: RoofFormHypothesis
    evidence_score: float
    density_roof_per_m2: float | None
    lidar_planar_support_ratio: float | None
    ndsm_valid_ratio: float
    gradient_p50: float
    gradient_p95: float
    discontinuity_ratio: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation."""
        payload = asdict(self)
        payload["hypothesis"] = self.hypothesis.value
        return payload


def infer_roof_hypothesis(
    lidar: LidarQualityReport,
    height: HeightEvidenceMetrics,
    *,
    parameters: RoofHypothesisParameters | None = None,
) -> RoofHypothesisReport:
    """Combine LiDAR support and nDSM morphology without claiming final topology."""
    parameters = parameters or RoofHypothesisParameters()
    density = lidar.density_roof_per_m2
    planar_support = lidar.planar_support_ratio
    evidence_score = _evidence_score(density, planar_support, height.valid_ratio)
    missing_reasons: list[str] = []
    if density is None or density < parameters.minimum_density_per_m2:
        missing_reasons.append("roof-point density is below the minimum support threshold")
    if height.valid_ratio < parameters.minimum_valid_ratio:
        missing_reasons.append("normalized-height coverage is below the minimum threshold")
    if planar_support is None:
        missing_reasons.append("local planar support is unavailable")
    if missing_reasons:
        return _report(
            RoofFormHypothesis.INSUFFICIENT_EVIDENCE,
            evidence_score,
            lidar,
            height,
            missing_reasons,
        )
    assert planar_support is not None

    if (
        height.gradient_p95 <= parameters.flat_gradient_p95_max
        and height.discontinuity_ratio <= parameters.flat_discontinuity_ratio_max
    ):
        return _report(
            RoofFormHypothesis.FLAT,
            evidence_score,
            lidar,
            height,
            ["low nDSM gradients and few vertical discontinuities"],
        )
    if (
        height.gradient_p95 >= parameters.compound_gradient_p95_min
        or height.discontinuity_ratio >= parameters.compound_discontinuity_ratio_min
    ):
        return _report(
            RoofFormHypothesis.COMPOUND,
            evidence_score,
            lidar,
            height,
            ["large gradients or frequent height discontinuities indicate multiple roof regimes"],
        )
    if (
        planar_support >= parameters.minimum_planar_support
        and height.gradient_p50 >= parameters.pitched_gradient_p50_min
    ):
        return _report(
            RoofFormHypothesis.PITCHED,
            evidence_score,
            lidar,
            height,
            ["stable local planes and sustained nDSM slope support a pitched form"],
        )
    return _report(
        RoofFormHypothesis.UNCERTAIN,
        evidence_score,
        lidar,
        height,
        ["available evidence does not separate flat, pitched, and compound forms"],
    )


def _evidence_score(
    density: float | None,
    planar_support: float | None,
    valid_ratio: float,
) -> float:
    components = [valid_ratio]
    if density is not None:
        components.append(min(density / 4.0, 1.0))
    if planar_support is not None:
        components.append(planar_support)
    return float(np.mean(np.clip(components, 0.0, 1.0)))


def _report(
    hypothesis: RoofFormHypothesis,
    evidence_score: float,
    lidar: LidarQualityReport,
    height: HeightEvidenceMetrics,
    reasons: list[str],
) -> RoofHypothesisReport:
    return RoofHypothesisReport(
        hypothesis=hypothesis,
        evidence_score=evidence_score,
        density_roof_per_m2=lidar.density_roof_per_m2,
        lidar_planar_support_ratio=lidar.planar_support_ratio,
        ndsm_valid_ratio=height.valid_ratio,
        gradient_p50=height.gradient_p50,
        gradient_p95=height.gradient_p95,
        discontinuity_ratio=height.discontinuity_ratio,
        reasons=tuple(reasons),
    )
