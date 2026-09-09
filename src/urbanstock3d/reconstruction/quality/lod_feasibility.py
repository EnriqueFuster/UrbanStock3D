"""Pure evidence-based feasibility assessment for supported CityJSON LoDs."""

from dataclasses import asdict, dataclass

import numpy as np

from urbanstock3d.reconstruction.enums import LodRequest
from urbanstock3d.reconstruction.evidence import HeightEvidenceMetrics
from urbanstock3d.reconstruction.models import LidarQualityReport
from urbanstock3d.reconstruction.quality.roof_complexity import (
    RoofComplexityClass,
    RoofComplexityReport,
)


@dataclass(frozen=True)
class LodEvidenceAvailability:
    """Explicit availability of evidence not represented by quality metrics."""

    reliable_footprint: bool
    ground_elevation: bool
    representative_height: bool


@dataclass(frozen=True)
class LodFeasibilityParameters:
    """Provisional gates to calibrate against the selected-building benchmark."""

    lod13_minimum_coverage: float = 0.65
    lod13_step_ratio: float = 0.05
    lod22_minimum_density_per_m2: float = 2.5
    lod22_minimum_coverage: float = 0.75
    lod22_maximum_hole_ratio: float = 0.30
    lod22_minimum_planar_support: float = 0.50
    lod22_minimum_alignment: float = 0.70
    lod22_minimum_plane_explained_ratio: float = 0.40
    lod22_minimum_score: float = 0.68

    def __post_init__(self) -> None:
        if self.lod22_minimum_density_per_m2 <= 0:
            raise ValueError("LoD2.2 density threshold must be positive")
        ratios = (
            self.lod13_minimum_coverage,
            self.lod13_step_ratio,
            self.lod22_minimum_coverage,
            self.lod22_maximum_hole_ratio,
            self.lod22_minimum_planar_support,
            self.lod22_minimum_alignment,
            self.lod22_minimum_plane_explained_ratio,
            self.lod22_minimum_score,
        )
        if any(not 0 <= value <= 1 for value in ratios):
            raise ValueError("LoD feasibility ratios must lie between zero and one")


@dataclass(frozen=True)
class LodFeasibility:
    """Support decision for one level of detail."""

    lod: LodRequest
    supported: bool
    score: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["lod"] = self.lod.value
        return payload


@dataclass(frozen=True)
class LodFeasibilityReport:
    """Independent feasibility decisions ordered from LoD1.2 to LoD2.2."""

    lod12: LodFeasibility
    lod13: LodFeasibility
    lod22: LodFeasibility

    @property
    def highest_supported_lod(self) -> LodRequest | None:
        for result in (self.lod22, self.lod13, self.lod12):
            if result.supported:
                return result.lod
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "lod12": self.lod12.to_dict(),
            "lod13": self.lod13.to_dict(),
            "lod22": self.lod22.to_dict(),
            "highest_supported_lod": (
                self.highest_supported_lod.value if self.highest_supported_lod else None
            ),
        }


def assess_lod_feasibility(
    availability: LodEvidenceAvailability,
    lidar: LidarQualityReport,
    complexity: RoofComplexityReport,
    *,
    height: HeightEvidenceMetrics | None = None,
    parameters: LodFeasibilityParameters | None = None,
) -> LodFeasibilityReport:
    """Determine defensible LoDs without choosing or executing a backend."""
    parameters = parameters or LodFeasibilityParameters()
    lod12 = _assess_lod12(availability, lidar)
    lod13 = _assess_lod13(lod12, lidar, complexity, height, parameters)
    lod22 = _assess_lod22(lod12, lidar, complexity, height, parameters)
    return LodFeasibilityReport(lod12=lod12, lod13=lod13, lod22=lod22)


def _assess_lod12(
    availability: LodEvidenceAvailability,
    lidar: LidarQualityReport,
) -> LodFeasibility:
    checks = {
        "reliable cadastral footprint": availability.reliable_footprint,
        "ground elevation": availability.ground_elevation,
        "representative building height": availability.representative_height,
        "building observations": lidar.available and lidar.roof_point_count > 0,
    }
    score = float(np.mean(tuple(checks.values())))
    missing = tuple(f"missing {name}" for name, passed in checks.items() if not passed)
    reasons = missing or ("footprint, ground, and representative height are available",)
    return LodFeasibility(LodRequest.LOD12, not missing, score, reasons)


def _assess_lod13(
    lod12: LodFeasibility,
    lidar: LidarQualityReport,
    complexity: RoofComplexityReport,
    height: HeightEvidenceMetrics | None,
    parameters: LodFeasibilityParameters,
) -> LodFeasibility:
    coverage = lidar.coverage_100m or 0.0
    parts_support = (complexity.building_part_count or 0) > 1
    step_support = height is not None and height.discontinuity_ratio >= parameters.lod13_step_ratio
    checks = {
        "LoD1.2 prerequisites": lod12.supported,
        "spatial roof coverage": coverage >= parameters.lod13_minimum_coverage,
        "major height-part evidence": parts_support or step_support,
    }
    score = float(np.mean(tuple(checks.values())))
    failed = tuple(name for name, passed in checks.items() if not passed)
    reasons = (
        tuple(f"insufficient {name}" for name in failed)
        if failed
        else ("building parts or nDSM steps support differentiated height blocks",)
    )
    return LodFeasibility(LodRequest.LOD13, not failed, score, reasons)


def _assess_lod22(
    lod12: LodFeasibility,
    lidar: LidarQualityReport,
    complexity: RoofComplexityReport,
    height: HeightEvidenceMetrics | None,
    parameters: LodFeasibilityParameters,
) -> LodFeasibility:
    density = lidar.density_roof_per_m2 or 0.0
    coverage = lidar.coverage_100m or 0.0
    hole_support = 1.0 - (lidar.largest_hole_ratio if lidar.largest_hole_ratio is not None else 1.0)
    planar = lidar.planar_support_ratio or 0.0
    alignment = lidar.footprint_alignment_score or 0.0
    height_continuity = height.valid_ratio if height is not None else coverage
    complexity_support = {
        RoofComplexityClass.SIMPLE: 1.0,
        RoofComplexityClass.MODERATE: 0.85,
        RoofComplexityClass.COMPLEX: 0.65,
        RoofComplexityClass.VERY_COMPLEX: 0.40,
    }[complexity.complexity_class]
    components = (
        min(density / 5.0, 1.0),
        coverage,
        hole_support,
        planar,
        alignment,
        complexity.plane_explained_ratio,
        height_continuity,
        complexity_support,
    )
    score = float(np.mean(np.clip(components, 0.0, 1.0)))
    gates = {
        "LoD1.2 prerequisites": lod12.supported,
        "LiDAR density": density >= parameters.lod22_minimum_density_per_m2,
        "spatial roof coverage": coverage >= parameters.lod22_minimum_coverage,
        "hole control": hole_support >= 1.0 - parameters.lod22_maximum_hole_ratio,
        "local planar support": planar >= parameters.lod22_minimum_planar_support,
        "footprint alignment": alignment >= parameters.lod22_minimum_alignment,
        "dominant-plane explanation": (
            complexity.plane_explained_ratio >= parameters.lod22_minimum_plane_explained_ratio
        ),
        "aggregate evidence score": score >= parameters.lod22_minimum_score,
    }
    failed = tuple(name for name, passed in gates.items() if not passed)
    reasons = (
        tuple(f"insufficient {name}" for name in failed)
        if failed
        else ("measured geometry supports explicit roof topology",)
    )
    return LodFeasibility(LodRequest.LOD22, not failed, score, reasons)
