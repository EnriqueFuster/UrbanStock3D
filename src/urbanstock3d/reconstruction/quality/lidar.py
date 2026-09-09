"""Cheap, interpretable LiDAR quality metrics computed before reconstruction."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from scipy.spatial import cKDTree

from urbanstock3d.processors.lidar import BUILDING_CLASS, points_in_polygon, polygon_area
from urbanstock3d.reconstruction.models import LidarQualityReport

PolygonRings = tuple[tuple[tuple[float, float], ...], ...]
Footprint = tuple[PolygonRings, ...]


@dataclass(frozen=True)
class CoverageGrid:
    """Rasterized roof occupancy constrained to the cadastral footprint."""

    resolution_m: float
    min_x: float
    min_y: float
    footprint_mask: np.ndarray[Any, np.dtype[np.bool_]]
    occupied_mask: np.ndarray[Any, np.dtype[np.bool_]]

    @property
    def coverage(self) -> float:
        valid_cells = int(np.count_nonzero(self.footprint_mask))
        return float(np.count_nonzero(self.occupied_mask & self.footprint_mask) / valid_cells)


@dataclass(frozen=True)
class LidarQualityParameters:
    """Explicit tunables for bounded pre-reconstruction LiDAR analysis."""

    pca_sample_size: int = 2_000
    pca_neighbours: int = 16
    planar_residual_max_m: float = 0.20
    planarity_min: float = 0.30

    def __post_init__(self) -> None:
        if self.pca_sample_size <= 0:
            raise ValueError("PCA sample size must be positive")
        if self.pca_neighbours < 3:
            raise ValueError("PCA requires at least three neighbours")
        if self.planar_residual_max_m <= 0:
            raise ValueError("Planar residual threshold must be positive")
        if not 0 <= self.planarity_min <= 1:
            raise ValueError("Minimum planarity must be between zero and one")


def assess_lidar_quality(
    path: Path,
    footprint: Footprint,
    *,
    parameters: LidarQualityParameters | None = None,
) -> tuple[LidarQualityReport, CoverageGrid]:
    """Measure spatial support without modifying or filtering source points."""
    if not footprint:
        raise ValueError("LiDAR quality assessment requires a footprint")
    parameters = parameters or LidarQualityParameters()
    cloud = laspy.read(path)
    x = np.asarray(cloud.x)
    y = np.asarray(cloud.y)
    z = np.asarray(cloud.z)
    classification = np.asarray(cloud.classification, dtype=np.uint8)
    inside = _points_in_footprint(x, y, footprint)
    roof = inside & (classification == BUILDING_CLASS)
    area = sum(polygon_area(polygon) for polygon in footprint)
    grid_050 = rasterize_coverage(x[roof], y[roof], footprint, resolution_m=0.5)
    grid_100 = rasterize_coverage(x[roof], y[roof], footprint, resolution_m=1.0)
    spacing_median, spacing_p90 = _nearest_neighbour_spacing(x[roof], y[roof])
    planar_support, local_residual = local_planar_support(
        np.column_stack((x[roof], y[roof], z[roof])),
        parameters,
    )
    outlier_ratio = _robust_elevation_outlier_ratio(z[roof])
    alignment_score, shift_x, shift_y = _estimate_alignment(
        x[classification == BUILDING_CLASS],
        y[classification == BUILDING_CLASS],
        footprint,
    )
    warnings: list[str] = []
    if np.count_nonzero(roof) == 0:
        warnings.append("no classified building points inside footprint")
    if shift_x is not None and shift_y is not None and max(abs(shift_x), abs(shift_y)) == 2.0:
        warnings.append("alignment optimum reached the bounded search limit")
    report = LidarQualityReport(
        available=len(cloud.points) > 0,
        point_count=int(np.count_nonzero(inside)),
        roof_point_count=int(np.count_nonzero(roof)),
        density_all_per_m2=float(np.count_nonzero(inside) / area),
        density_roof_per_m2=float(np.count_nonzero(roof) / area),
        coverage_050m=grid_050.coverage,
        coverage_100m=grid_100.coverage,
        largest_hole_ratio=_largest_empty_component_ratio(grid_100),
        nn_spacing_median_m=spacing_median,
        nn_spacing_p90_m=spacing_p90,
        planar_support_ratio=planar_support,
        local_residual_median_m=local_residual,
        outlier_ratio=outlier_ratio,
        footprint_alignment_score=alignment_score,
        estimated_shift_x_m=shift_x,
        estimated_shift_y_m=shift_y,
        quality_class="UNCLASSIFIED",
        score=0.0,
        warnings=tuple(warnings),
    )
    return report, grid_100


def rasterize_coverage(
    roof_x: np.ndarray[Any, np.dtype[np.floating[Any]]],
    roof_y: np.ndarray[Any, np.dtype[np.floating[Any]]],
    footprint: Footprint,
    *,
    resolution_m: float,
) -> CoverageGrid:
    """Rasterize footprint cells and mark those supported by roof points."""
    if resolution_m <= 0:
        raise ValueError("Coverage resolution must be positive")
    positions = [position for polygon in footprint for ring in polygon for position in ring]
    xs, ys = zip(*positions, strict=True)
    min_x, min_y = min(xs), min(ys)
    cols = max(1, int(np.ceil((max(xs) - min_x) / resolution_m)))
    rows = max(1, int(np.ceil((max(ys) - min_y) / resolution_m)))
    center_x = min_x + (np.arange(cols) + 0.5) * resolution_m
    center_y = min_y + (np.arange(rows) + 0.5) * resolution_m
    mesh_x, mesh_y = np.meshgrid(center_x, center_y)
    footprint_mask = _points_in_footprint(mesh_x.ravel(), mesh_y.ravel(), footprint).reshape(
        rows, cols
    )
    occupied = np.zeros((rows, cols), dtype=np.bool_)
    if len(roof_x):
        point_cols = np.floor((roof_x - min_x) / resolution_m).astype(int)
        point_rows = np.floor((roof_y - min_y) / resolution_m).astype(int)
        valid = (point_cols >= 0) & (point_cols < cols) & (point_rows >= 0) & (point_rows < rows)
        occupied[point_rows[valid], point_cols[valid]] = True
    if not np.any(footprint_mask):
        raise ValueError("Footprint is smaller than the requested coverage grid")
    return CoverageGrid(resolution_m, min_x, min_y, footprint_mask, occupied)


def _largest_empty_component_ratio(grid: CoverageGrid) -> float:
    empty = grid.footprint_mask & ~grid.occupied_mask
    visited = np.zeros(empty.shape, dtype=np.bool_)
    largest = 0
    rows, cols = empty.shape
    for row, col in np.argwhere(empty):
        if visited[row, col]:
            continue
        size = 0
        queue = deque([(int(row), int(col))])
        visited[row, col] = True
        while queue:
            current_row, current_col = queue.popleft()
            size += 1
            for next_row, next_col in (
                (current_row - 1, current_col),
                (current_row + 1, current_col),
                (current_row, current_col - 1),
                (current_row, current_col + 1),
            ):
                if (
                    0 <= next_row < rows
                    and 0 <= next_col < cols
                    and empty[next_row, next_col]
                    and not visited[next_row, next_col]
                ):
                    visited[next_row, next_col] = True
                    queue.append((next_row, next_col))
        largest = max(largest, size)
    footprint_cells = int(np.count_nonzero(grid.footprint_mask))
    return largest / footprint_cells


def _nearest_neighbour_spacing(
    x: np.ndarray[Any, np.dtype[np.floating[Any]]],
    y: np.ndarray[Any, np.dtype[np.floating[Any]]],
) -> tuple[float | None, float | None]:
    if len(x) < 2:
        return None, None
    points = np.column_stack((x, y))
    if len(points) > 10_000:
        indices = np.random.default_rng(42).choice(len(points), 10_000, replace=False)
        points = points[indices]
    distances, _ = cKDTree(points).query(points, k=2)
    nearest = distances[:, 1]
    return float(np.median(nearest)), float(np.percentile(nearest, 90))


def _robust_elevation_outlier_ratio(
    z: np.ndarray[Any, np.dtype[np.floating[Any]]],
) -> float | None:
    if len(z) < 3:
        return None
    median = np.median(z)
    mad = np.median(np.abs(z - median))
    if mad == 0:
        return float(np.count_nonzero(z != median) / len(z))
    robust_z = np.abs(z - median) / (1.4826 * mad)
    return float(np.count_nonzero(robust_z > 6.0) / len(z))


def local_planar_support(
    points: np.ndarray[Any, np.dtype[np.floating[Any]]],
    parameters: LidarQualityParameters,
) -> tuple[float | None, float | None]:
    """Estimate local surface planarity from bounded k-neighbourhood PCA."""
    if len(points) < 3:
        return None, None
    sample_size = min(len(points), parameters.pca_sample_size)
    if sample_size == len(points):
        sample_indices = np.arange(len(points))
    else:
        sample_indices = np.random.default_rng(42).choice(len(points), sample_size, replace=False)
    neighbours = min(parameters.pca_neighbours, len(points))
    tree = cKDTree(points)
    _, neighbour_indices = tree.query(points[sample_indices], k=neighbours)
    neighbourhoods = points[neighbour_indices]
    centered = neighbourhoods - neighbourhoods.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / neighbours
    eigenvalues = np.linalg.eigvalsh(covariance)
    largest = eigenvalues[:, 2]
    planarity = np.divide(
        eigenvalues[:, 1] - eigenvalues[:, 0],
        largest,
        out=np.zeros_like(largest),
        where=largest > 0,
    )
    residuals = np.sqrt(np.maximum(eigenvalues[:, 0], 0.0))
    supported = (residuals <= parameters.planar_residual_max_m) & (
        planarity >= parameters.planarity_min
    )
    return float(np.mean(supported)), float(np.median(residuals))


def _estimate_alignment(
    x: np.ndarray[Any, np.dtype[np.floating[Any]]],
    y: np.ndarray[Any, np.dtype[np.floating[Any]]],
    footprint: Footprint,
) -> tuple[float | None, float | None, float | None]:
    if len(x) == 0:
        return None, None, None
    positions = [position for polygon in footprint for ring in polygon for position in ring]
    xs, ys = zip(*positions, strict=True)
    nearby = (
        (x >= min(xs) - 2.0) & (x <= max(xs) + 2.0) & (y >= min(ys) - 2.0) & (y <= max(ys) + 2.0)
    )
    x, y = x[nearby], y[nearby]
    if len(x) == 0:
        return None, None, None
    candidates = np.arange(-2.0, 2.01, 0.5)
    best_score, best_x, best_y = -1.0, 0.0, 0.0
    for shift_x in candidates:
        for shift_y in candidates:
            score = float(np.mean(_points_in_footprint(x + shift_x, y + shift_y, footprint)))
            if score > best_score:
                best_score, best_x, best_y = score, float(shift_x), float(shift_y)
    return best_score, best_x, best_y


def _points_in_footprint(
    x: np.ndarray[Any, np.dtype[np.floating[Any]]],
    y: np.ndarray[Any, np.dtype[np.floating[Any]]],
    footprint: Footprint,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    inside = np.zeros(x.shape, dtype=np.bool_)
    for polygon in footprint:
        inside |= points_in_polygon(x, y, polygon)
    return inside
