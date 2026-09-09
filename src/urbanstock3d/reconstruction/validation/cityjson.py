"""Cheap structural and footprint checks for reconstructed CityJSON solids."""

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from urbanstock3d.formats.cityjson import lod_surfaces, read_cityjsonseq, transformed_vertices
from urbanstock3d.processors.lidar import polygon_area
from urbanstock3d.reconstruction.quality.lidar import Footprint


@dataclass(frozen=True)
class CityJsonQualityParameters:
    """Initial sanity thresholds, intentionally separate from scientific accuracy metrics."""

    minimum_footprint_bbox_iou: float = 0.80
    minimum_height_m: float = 1.0
    maximum_height_m: float = 150.0
    tiny_face_area_m2: float = 0.25


@dataclass(frozen=True)
class CityJsonQualityReport:
    """Structural validity and coarse plausibility for one reconstructed LoD."""

    lod: str
    vertex_count: int
    face_count: int
    semantic_counts: dict[str, int]
    watertight: bool
    invalid_ring_count: int
    tiny_face_fraction: float
    height_range_m: float
    footprint_bbox_iou: float
    footprint_area_ratio: float | None
    geometry_valid: bool
    plausibility_pass: bool
    accepted: bool
    failures: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_cityjsonseq(
    model_path: Path,
    footprint: Footprint,
    *,
    lod: str = "2.2",
    parameters: CityJsonQualityParameters | None = None,
) -> CityJsonQualityReport:
    """Validate one Roofer CityJSONSeq artifact against its input footprint."""
    parameters = parameters or CityJsonQualityParameters()
    metadata, feature = read_cityjsonseq(model_path)
    vertices = transformed_vertices(metadata, feature)
    surfaces = lod_surfaces(feature, lod)
    invalid_rings = 0
    edge_counts: Counter[tuple[int, int]] = Counter()
    face_areas: list[float] = []
    semantic_counts: Counter[str] = Counter()
    ground_area = 0.0
    for surface in surfaces:
        semantic_counts[surface.semantic_type] += 1
        for ring in surface.rings:
            if len(ring) < 3 or any(index < 0 or index >= len(vertices) for index in ring):
                invalid_rings += 1
                continue
            closed = (*ring, ring[0])
            edge_counts.update(
                (min(start, end), max(start, end)) for start, end in zip(closed, closed[1:])
            )
            area = _polygon_area_3d(vertices[np.asarray(ring)])
            face_areas.append(area)
            if surface.semantic_type == "GroundSurface":
                ground_area += _polygon_area_xy(vertices[np.asarray(ring)])
    finite_vertices = bool(len(vertices)) and bool(np.all(np.isfinite(vertices)))
    watertight = bool(edge_counts) and all(count == 2 for count in edge_counts.values())
    height_range = float(np.ptp(vertices[:, 2])) if finite_vertices else 0.0
    footprint_bbox = _footprint_bbox(footprint)
    model_bbox = (
        float(np.min(vertices[:, 0])),
        float(np.min(vertices[:, 1])),
        float(np.max(vertices[:, 0])),
        float(np.max(vertices[:, 1])),
    )
    bbox_iou = _bbox_iou(footprint_bbox, model_bbox) if finite_vertices else 0.0
    footprint_area = sum(polygon_area(polygon) for polygon in footprint)
    area_ratio = ground_area / footprint_area if ground_area > 0 else None
    tiny_fraction = (
        float(np.mean(np.asarray(face_areas) < parameters.tiny_face_area_m2)) if face_areas else 1.0
    )
    required_semantics = {"GroundSurface", "WallSurface", "RoofSurface"}
    failures: list[str] = []
    if not finite_vertices:
        failures.append("vertices are empty or non-finite")
    if invalid_rings:
        failures.append("surface rings contain invalid vertex references")
    if not watertight:
        failures.append("solid boundary is not watertight")
    if not required_semantics.issubset(semantic_counts):
        failures.append("required semantic surface types are missing")
    geometry_valid = not failures
    plausibility_failures: list[str] = []
    if bbox_iou < parameters.minimum_footprint_bbox_iou:
        plausibility_failures.append("model footprint bbox does not align with source footprint")
    if not parameters.minimum_height_m <= height_range <= parameters.maximum_height_m:
        plausibility_failures.append("model height is outside configured plausibility bounds")
    failures.extend(plausibility_failures)
    warnings = (
        ("ground-surface area is unavailable",)
        if area_ratio is None
        else (() if 0.8 <= area_ratio <= 1.2 else ("ground area differs from source footprint",))
    )
    return CityJsonQualityReport(
        lod=lod,
        vertex_count=len(vertices),
        face_count=len(face_areas),
        semantic_counts=dict(sorted(semantic_counts.items())),
        watertight=watertight,
        invalid_ring_count=invalid_rings,
        tiny_face_fraction=tiny_fraction,
        height_range_m=height_range,
        footprint_bbox_iou=bbox_iou,
        footprint_area_ratio=area_ratio,
        geometry_valid=geometry_valid,
        plausibility_pass=not plausibility_failures,
        accepted=not failures,
        failures=tuple(failures),
        warnings=warnings,
    )


def _polygon_area_3d(points: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    shifted = np.roll(points, -1, axis=0)
    return float(np.linalg.norm(np.sum(np.cross(points, shifted), axis=0)) / 2)


def _polygon_area_xy(points: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    shifted = np.roll(points, -1, axis=0)
    return float(abs(np.sum(points[:, 0] * shifted[:, 1] - shifted[:, 0] * points[:, 1])) / 2)


def _footprint_bbox(footprint: Footprint) -> tuple[float, float, float, float]:
    positions = [position for polygon in footprint for ring in polygon for position in ring]
    x, y = zip(*positions, strict=True)
    return min(x), min(y), max(x), max(y)


def _bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0
