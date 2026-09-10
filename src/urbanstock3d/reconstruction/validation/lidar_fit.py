"""Independent point-to-roof fit metrics for reconstructed CityJSON geometry."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import laspy
import numpy as np

from urbanstock3d.formats.cityjson import lod_surfaces, read_cityjsonseq, transformed_vertices
from urbanstock3d.processors.lidar import BUILDING_CLASS, points_in_polygon
from urbanstock3d.reconstruction.quality.lidar import Footprint
from urbanstock3d.reconstruction.validation.roof_observations import (
    RoofObservationParameters,
    select_roof_observations,
)


@dataclass(frozen=True)
class LidarFitParameters:
    """Bounds for a reproducible and inexpensive fit calculation."""

    sample_size: int = 5_000
    random_seed: int = 42
    observations: RoofObservationParameters = RoofObservationParameters()

    def __post_init__(self) -> None:
        if self.sample_size <= 0:
            raise ValueError("Sample size must be positive")


@dataclass(frozen=True)
class LidarFitReport:
    """Distances from observed roof observations to reconstructed roof triangles."""

    source_roof_point_count: int
    selected_roof_point_count: int
    sampled_roof_point_count: int
    roof_triangle_count: int
    distance_median_m: float
    distance_rmse_m: float
    distance_p95_m: float
    within_020m_ratio: float
    within_050m_ratio: float
    within_100m_ratio: float
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready metrics."""
        return asdict(self)


def assess_cityjson_lidar_fit(
    model_path: Path,
    lidar_path: Path,
    footprint: Footprint,
    *,
    lod: str = "2.2",
    parameters: LidarFitParameters | None = None,
) -> LidarFitReport:
    """Measure exact 3D distances from sampled roof points to roof triangles."""
    parameters = parameters or LidarFitParameters()
    metadata, feature = read_cityjsonseq(model_path)
    vertices = transformed_vertices(metadata, feature)
    triangles, used_fan_triangulation = _roof_triangles(feature, vertices, lod)
    cloud = laspy.read(lidar_path)
    classification = np.asarray(cloud.classification, dtype=np.uint8)
    building = classification == BUILDING_CLASS
    x = np.asarray(cloud.x)[building]
    y = np.asarray(cloud.y)[building]
    z = np.asarray(cloud.z)[building]
    inside = np.zeros(x.shape, dtype=np.bool_)
    for polygon in footprint:
        inside |= points_in_polygon(x, y, polygon)
    points = np.column_stack((x[inside], y[inside], z[inside]))
    source_count = len(points)
    if source_count == 0:
        raise ValueError("No classified roof points exist inside the footprint")
    selected, selection = select_roof_observations(points, parameters=parameters.observations)
    points = points[selected]
    if len(points) > parameters.sample_size:
        indices = np.random.default_rng(parameters.random_seed).choice(
            len(points), parameters.sample_size, replace=False
        )
        points = points[indices]
    distances = np.array(
        [
            min(point_triangle_distance(point, triangle) for triangle in triangles)
            for point in points
        ]
    )
    warnings = (
        ("non-triangular roof faces used fan triangulation",) if used_fan_triangulation else ()
    )
    return LidarFitReport(
        source_roof_point_count=source_count,
        selected_roof_point_count=selection.selected_point_count,
        sampled_roof_point_count=len(points),
        roof_triangle_count=len(triangles),
        distance_median_m=float(np.median(distances)),
        distance_rmse_m=float(np.sqrt(np.mean(np.square(distances)))),
        distance_p95_m=float(np.percentile(distances, 95)),
        within_020m_ratio=_within_ratio(distances, 0.2),
        within_050m_ratio=_within_ratio(distances, 0.5),
        within_100m_ratio=_within_ratio(distances, 1.0),
        warnings=(
            *warnings,
            f"roof observations selected with {selection.method}: "
            f"{selection.selected_point_count}/{selection.source_point_count}",
        ),
    )


def _within_ratio(
    distances: np.ndarray[Any, np.dtype[np.floating[Any]]], threshold: float
) -> float:
    return float(np.mean(distances <= threshold + 1e-9))


def _roof_triangles(
    feature: dict[str, Any],
    vertices: np.ndarray[Any, np.dtype[np.float64]],
    lod: str,
) -> tuple[list[np.ndarray[Any, np.dtype[np.float64]]], bool]:
    triangles: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    used_fan = False
    for surface in lod_surfaces(feature, lod):
        if surface.semantic_type != "RoofSurface":
            continue
        if len(surface.rings) != 1:
            raise ValueError("Roof fit does not support surfaces with interior rings")
        ring = surface.rings[0]
        if len(ring) > 3:
            used_fan = True
        for index in range(1, len(ring) - 1):
            triangles.append(vertices[np.asarray((ring[0], ring[index], ring[index + 1]))])
    if not triangles:
        raise ValueError(f"CityJSON feature contains no LoD {lod} roof triangles")
    return triangles, used_fan


def point_triangle_distance(
    point: np.ndarray[Any, np.dtype[np.floating[Any]]],
    triangle: np.ndarray[Any, np.dtype[np.float64]],
) -> float:
    """Return exact Euclidean distance using triangle Voronoi regions."""
    a, b, c = triangle
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = float(ab @ ap), float(ac @ ap)
    if d1 <= 0 and d2 <= 0:
        return float(np.linalg.norm(ap))
    bp = point - b
    d3, d4 = float(ab @ bp), float(ac @ bp)
    if d3 >= 0 and d4 <= d3:
        return float(np.linalg.norm(bp))
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        projection = a + (d1 / (d1 - d3)) * ab
        return float(np.linalg.norm(point - projection))
    cp = point - c
    d5, d6 = float(ab @ cp), float(ac @ cp)
    if d6 >= 0 and d5 <= d6:
        return float(np.linalg.norm(cp))
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        projection = a + (d2 / (d2 - d6)) * ac
        return float(np.linalg.norm(point - projection))
    va = d3 * d6 - d5 * d4
    if va <= 0 and d4 - d3 >= 0 and d5 - d6 >= 0:
        edge = c - b
        projection = b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * edge
        return float(np.linalg.norm(point - projection))
    normal = np.cross(ab, ac)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm == 0:
        raise ValueError("Reconstructed roof contains a degenerate triangle")
    return abs(float((point - a) @ normal)) / normal_norm
