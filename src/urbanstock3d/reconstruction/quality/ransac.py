"""Bounded RANSAC diagnostics for preliminary roof-plane evidence."""

from dataclasses import asdict, dataclass
from math import atan2, degrees
from pathlib import Path
from typing import Any

import laspy
import numpy as np

from urbanstock3d.processors.lidar import BUILDING_CLASS, points_in_polygon
from urbanstock3d.reconstruction.quality.lidar import Footprint


@dataclass(frozen=True)
class RansacParameters:
    """Explicit bounds for inexpensive pre-routing plane detection."""

    sample_size: int = 5_000
    max_iterations: int = 200
    max_planes: int = 8
    distance_threshold_m: float = 0.20
    minimum_support_points: int = 30
    minimum_support_ratio: float = 0.05
    normal_cluster_angle_deg: float = 10.0
    random_seed: int = 42

    def __post_init__(self) -> None:
        if min(self.sample_size, self.max_iterations, self.max_planes) <= 0:
            raise ValueError("RANSAC execution bounds must be positive")
        if self.minimum_support_points < 3:
            raise ValueError("A supported plane requires at least three points")
        if self.distance_threshold_m <= 0:
            raise ValueError("RANSAC distance threshold must be positive")
        if not 0 < self.minimum_support_ratio <= 1:
            raise ValueError("Minimum support ratio must lie in (0, 1]")
        if not 0 < self.normal_cluster_angle_deg <= 90:
            raise ValueError("Normal cluster angle must lie in (0, 90]")


@dataclass(frozen=True)
class PreliminaryPlane:
    """One locally fitted plane and its measured support."""

    normal: tuple[float, float, float]
    centroid: tuple[float, float, float]
    support_point_count: int
    support_ratio: float
    residual_median_m: float
    residual_p95_m: float
    slope_deg: float
    aspect_deg: float | None


@dataclass(frozen=True)
class RoofPlaneEvidence:
    """Bounded plane inventory used for complexity assessment, not reconstruction."""

    source_point_count: int
    analysed_point_count: int
    explained_point_count: int
    explained_ratio: float
    preliminary_plane_count: int
    dominant_normal_count: int
    planes: tuple[PreliminaryPlane, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready report."""
        return asdict(self)


def assess_roof_planes(
    path: Path,
    footprint: Footprint,
    *,
    parameters: RansacParameters | None = None,
) -> RoofPlaneEvidence:
    """Read classified building points inside a footprint and detect dominant planes."""
    cloud = laspy.read(path)
    classification = np.asarray(cloud.classification, dtype=np.uint8)
    building = classification == BUILDING_CLASS
    x = np.asarray(cloud.x)[building]
    y = np.asarray(cloud.y)[building]
    z = np.asarray(cloud.z)[building]
    inside = np.zeros(x.shape, dtype=np.bool_)
    for polygon in footprint:
        inside |= points_in_polygon(x, y, polygon)
    return detect_roof_planes(
        np.column_stack((x[inside], y[inside], z[inside])),
        parameters=parameters,
    )


def detect_roof_planes(
    points: np.ndarray[Any, np.dtype[np.floating[Any]]],
    *,
    parameters: RansacParameters | None = None,
) -> RoofPlaneEvidence:
    """Sequentially extract supported planes from a bounded deterministic sample."""
    parameters = parameters or RansacParameters()
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Roof points must have shape (n, 3)")
    source_count = len(points)
    if source_count < 3:
        return RoofPlaneEvidence(source_count, source_count, 0, 0.0, 0, 0, (), ("too few points",))
    rng = np.random.default_rng(parameters.random_seed)
    if source_count > parameters.sample_size:
        sample_indices = rng.choice(source_count, parameters.sample_size, replace=False)
        analysed = np.asarray(points[sample_indices], dtype=np.float64)
    else:
        analysed = np.asarray(points, dtype=np.float64)
    remaining = analysed.copy()
    planes: list[PreliminaryPlane] = []
    minimum_support = max(
        parameters.minimum_support_points,
        int(np.ceil(len(analysed) * parameters.minimum_support_ratio)),
    )
    for _ in range(parameters.max_planes):
        if len(remaining) < minimum_support:
            break
        inliers = _best_inliers(remaining, parameters, rng)
        if inliers is None or int(np.count_nonzero(inliers)) < minimum_support:
            break
        normal, centroid = _fit_plane_pca(remaining[inliers])
        residuals = _plane_distances(remaining, normal, centroid)
        refined = residuals <= parameters.distance_threshold_m
        support_count = int(np.count_nonzero(refined))
        if support_count < minimum_support:
            break
        supported_residuals = residuals[refined]
        planes.append(
            _plane_report(
                normal,
                centroid,
                support_count,
                len(analysed),
                supported_residuals,
            )
        )
        remaining = remaining[~refined]
    explained = len(analysed) - len(remaining)
    warnings = () if planes else ("no plane met the configured support thresholds",)
    return RoofPlaneEvidence(
        source_point_count=source_count,
        analysed_point_count=len(analysed),
        explained_point_count=explained,
        explained_ratio=explained / len(analysed),
        preliminary_plane_count=len(planes),
        dominant_normal_count=_count_normal_clusters(planes, parameters.normal_cluster_angle_deg),
        planes=tuple(planes),
        warnings=warnings,
    )


def _best_inliers(
    points: np.ndarray[Any, np.dtype[np.floating[Any]]],
    parameters: RansacParameters,
    rng: np.random.Generator,
) -> np.ndarray[Any, np.dtype[np.bool_]] | None:
    best: np.ndarray[Any, np.dtype[np.bool_]] | None = None
    best_count = 0
    for _ in range(parameters.max_iterations):
        indices = rng.choice(len(points), 3, replace=False)
        first, second, third = points[indices]
        normal = np.cross(second - first, third - first)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-10:
            continue
        normal /= norm
        inliers = _plane_distances(points, normal, first) <= parameters.distance_threshold_m
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best, best_count = inliers, count
    return best


def _fit_plane_pca(
    points: np.ndarray[Any, np.dtype[np.floating[Any]]],
) -> tuple[
    np.ndarray[Any, np.dtype[np.floating[Any]]], np.ndarray[Any, np.dtype[np.floating[Any]]]
]:
    centroid = np.mean(points, axis=0)
    _, _, vectors = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vectors[-1]
    if normal[2] < 0:
        normal = -normal
    return np.asarray(normal, dtype=np.float64), np.asarray(centroid, dtype=np.float64)


def _plane_distances(
    points: np.ndarray[Any, np.dtype[np.floating[Any]]],
    normal: np.ndarray[Any, np.dtype[np.floating[Any]]],
    origin: np.ndarray[Any, np.dtype[np.floating[Any]]],
) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
    return np.asarray(np.abs((points - origin) @ normal), dtype=np.float64)


def _plane_report(
    normal: np.ndarray[Any, np.dtype[np.floating[Any]]],
    centroid: np.ndarray[Any, np.dtype[np.floating[Any]]],
    support_count: int,
    total_count: int,
    residuals: np.ndarray[Any, np.dtype[np.floating[Any]]],
) -> PreliminaryPlane:
    slope = degrees(np.arccos(np.clip(normal[2], -1.0, 1.0)))
    horizontal = float(np.hypot(normal[0], normal[1]))
    aspect = None if horizontal < 1e-8 else (degrees(atan2(-normal[0], -normal[1])) + 360) % 360
    return PreliminaryPlane(
        normal=(float(normal[0]), float(normal[1]), float(normal[2])),
        centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
        support_point_count=support_count,
        support_ratio=support_count / total_count,
        residual_median_m=float(np.median(residuals)),
        residual_p95_m=float(np.percentile(residuals, 95)),
        slope_deg=slope,
        aspect_deg=aspect,
    )


def _count_normal_clusters(planes: list[PreliminaryPlane], angle_threshold_deg: float) -> int:
    representatives: list[np.ndarray[Any, np.dtype[np.floating[Any]]]] = []
    cosine_threshold = float(np.cos(np.deg2rad(angle_threshold_deg)))
    for plane in sorted(planes, key=lambda item: item.support_point_count, reverse=True):
        normal = np.asarray(plane.normal)
        if not any(
            abs(float(normal @ existing)) >= cosine_threshold for existing in representatives
        ):
            representatives.append(normal)
    return len(representatives)
