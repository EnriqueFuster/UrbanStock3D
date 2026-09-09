"""Interpretable roof-complexity assessment from pre-reconstruction evidence."""

from dataclasses import asdict, dataclass
from enum import StrEnum
from math import hypot, pi

import numpy as np

from urbanstock3d.processors.lidar import polygon_area
from urbanstock3d.reconstruction.evidence import HeightEvidenceMetrics
from urbanstock3d.reconstruction.quality.lidar import Footprint
from urbanstock3d.reconstruction.quality.ransac import RoofPlaneEvidence


class RoofComplexityClass(StrEnum):
    """Coarse complexity bands used by later feasibility and routing logic."""

    SIMPLE = "SIMPLE"
    MODERATE = "MODERATE"
    COMPLEX = "COMPLEX"
    VERY_COMPLEX = "VERY_COMPLEX"


@dataclass(frozen=True)
class RoofComplexityParameters:
    """Provisional scaling and weighting values to calibrate on selected buildings."""

    vertex_excess_scale: float = 12.0
    building_part_excess_scale: float = 4.0
    plane_excess_scale: float = 5.0
    normal_excess_scale: float = 3.0
    discontinuity_ratio_scale: float = 0.20
    simple_score_max: float = 0.25
    moderate_score_max: float = 0.50
    complex_score_max: float = 0.75
    footprint_weight: float = 0.15
    building_parts_weight: float = 0.10
    planes_weight: float = 0.25
    normals_weight: float = 0.15
    discontinuities_weight: float = 0.20
    unexplained_weight: float = 0.15

    def __post_init__(self) -> None:
        scales = (
            self.vertex_excess_scale,
            self.building_part_excess_scale,
            self.plane_excess_scale,
            self.normal_excess_scale,
            self.discontinuity_ratio_scale,
        )
        if any(value <= 0 for value in scales):
            raise ValueError("Complexity scaling parameters must be positive")
        thresholds = (self.simple_score_max, self.moderate_score_max, self.complex_score_max)
        if not 0 < thresholds[0] < thresholds[1] < thresholds[2] < 1:
            raise ValueError("Complexity score thresholds must be increasing within (0, 1)")
        if any(weight < 0 for weight in self.weights):
            raise ValueError("Complexity weights cannot be negative")
        if sum(self.weights) <= 0:
            raise ValueError("At least one complexity weight must be positive")

    @property
    def weights(self) -> tuple[float, ...]:
        return (
            self.footprint_weight,
            self.building_parts_weight,
            self.planes_weight,
            self.normals_weight,
            self.discontinuities_weight,
            self.unexplained_weight,
        )


@dataclass(frozen=True)
class RoofComplexityReport:
    """Measured inputs and a transparent preliminary complexity classification."""

    footprint_area_m2: float
    footprint_vertices: int
    footprint_compactness: float
    building_part_count: int | None
    height_range_m: float | None
    preliminary_plane_count: int
    dominant_normal_count: int
    plane_explained_ratio: float
    ndsm_discontinuity_ratio: float | None
    complexity_class: RoofComplexityClass
    score: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["complexity_class"] = self.complexity_class.value
        return payload


def assess_roof_complexity(
    footprint: Footprint,
    planes: RoofPlaneEvidence,
    *,
    height: HeightEvidenceMetrics | None = None,
    building_part_count: int | None = None,
    height_range_m: float | None = None,
    parameters: RoofComplexityParameters | None = None,
) -> RoofComplexityReport:
    """Combine independent morphology, plane, and nDSM complexity cues."""
    parameters = parameters or RoofComplexityParameters()
    if not footprint:
        raise ValueError("Roof complexity requires a building footprint")
    if building_part_count is not None and building_part_count < 0:
        raise ValueError("Building part count cannot be negative")
    area = sum(polygon_area(polygon) for polygon in footprint)
    perimeter = sum(_ring_perimeter(ring) for polygon in footprint for ring in polygon)
    vertices = sum(max(len(ring) - 1, 0) for polygon in footprint for ring in polygon)
    compactness = float(np.clip(4 * pi * area / perimeter**2, 0.0, 1.0))

    components: list[tuple[str, float, float]] = [
        (
            "irregular footprint",
            np.mean(
                (
                    1.0 - compactness,
                    _scaled_excess(vertices, 4, parameters.vertex_excess_scale),
                )
            ),
            parameters.footprint_weight,
        ),
        (
            "multiple preliminary planes",
            _scaled_excess(planes.preliminary_plane_count, 1, parameters.plane_excess_scale),
            parameters.planes_weight,
        ),
        (
            "multiple dominant orientations",
            _scaled_excess(planes.dominant_normal_count, 1, parameters.normal_excess_scale),
            parameters.normals_weight,
        ),
        (
            "roof observations not explained by dominant planes",
            1.0 - planes.explained_ratio,
            parameters.unexplained_weight,
        ),
    ]
    if building_part_count is not None:
        components.append(
            (
                "multiple cadastral building parts",
                _scaled_excess(
                    building_part_count,
                    1,
                    parameters.building_part_excess_scale,
                ),
                parameters.building_parts_weight,
            )
        )
    if height is not None:
        components.append(
            (
                "frequent nDSM height discontinuities",
                min(height.discontinuity_ratio / parameters.discontinuity_ratio_scale, 1.0),
                parameters.discontinuities_weight,
            )
        )
    active_weight = sum(weight for _, _, weight in components)
    score = sum(value * weight for _, value, weight in components) / active_weight
    complexity_class = _classify(score, parameters)
    ranked = sorted(components, key=lambda component: component[1] * component[2], reverse=True)
    reasons = tuple(name for name, value, _ in ranked if value >= 0.25)[:3]
    if not reasons:
        reasons = ("all available complexity indicators are low",)
    return RoofComplexityReport(
        footprint_area_m2=area,
        footprint_vertices=vertices,
        footprint_compactness=compactness,
        building_part_count=building_part_count,
        height_range_m=height_range_m,
        preliminary_plane_count=planes.preliminary_plane_count,
        dominant_normal_count=planes.dominant_normal_count,
        plane_explained_ratio=planes.explained_ratio,
        ndsm_discontinuity_ratio=height.discontinuity_ratio if height is not None else None,
        complexity_class=complexity_class,
        score=float(score),
        reasons=reasons,
    )


def _ring_perimeter(ring: tuple[tuple[float, float], ...]) -> float:
    return sum(hypot(end[0] - start[0], end[1] - start[1]) for start, end in zip(ring, ring[1:]))


def _scaled_excess(value: int, baseline: int, scale: float) -> float:
    return float(np.clip((value - baseline) / scale, 0.0, 1.0))


def _classify(
    score: float,
    parameters: RoofComplexityParameters,
) -> RoofComplexityClass:
    if score < parameters.simple_score_max:
        return RoofComplexityClass.SIMPLE
    if score < parameters.moderate_score_max:
        return RoofComplexityClass.MODERATE
    if score < parameters.complex_score_max:
        return RoofComplexityClass.COMPLEX
    return RoofComplexityClass.VERY_COMPLEX
