"""Inspect and process local LiDAR point-cloud artifacts."""

from collections import Counter
from dataclasses import asdict, dataclass
from math import inf
from pathlib import Path
from typing import Any

import laspy
import numpy as np

GROUND_CLASS = 2
VEGETATION_CLASSES = frozenset({3, 4, 5})
BUILDING_CLASS = 6


@dataclass(frozen=True)
class LidarCropSummary:
    """Quality statistics for points inside a rectangular context crop."""

    bbox: tuple[float, float, float, float]
    area_m2: float
    point_count: int
    point_density_m2: float
    z_min_m: float
    z_p01_m: float
    z_p50_m: float
    z_p95_m: float
    z_p99_m: float
    z_max_m: float
    class_histogram: dict[str, int]
    ground_point_count: int
    vegetation_point_count: int
    building_point_count: int
    overlap_flag_point_count: int
    legacy_overlap_class_point_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def inspect_lidar_header(path: Path) -> dict[str, Any]:
    """Read reproducibility and quality metadata without loading all points."""
    with laspy.open(path) as reader:
        header = reader.header
        crs = header.parse_crs()
        return {
            "las_version": str(header.version),
            "point_format": header.point_format.id,
            "point_count": header.point_count,
            "scales": header.scales.tolist(),
            "offsets": header.offsets.tolist(),
            "minimums": header.mins.tolist(),
            "maximums": header.maxs.tolist(),
            "dimensions": list(header.point_format.dimension_names),
            "crs": crs.to_string() if crs is not None else None,
        }


def summarize_lidar_bbox(
    path: Path,
    bbox: tuple[float, float, float, float],
    *,
    chunk_size: int = 1_000_000,
) -> LidarCropSummary:
    """Summarize a bbox crop while reading the point cloud in chunks."""
    min_x, min_y, max_x, max_y = bbox
    if min_x >= max_x or min_y >= max_y:
        raise ValueError("LiDAR crop bbox must have positive area")
    if chunk_size <= 0:
        raise ValueError("LiDAR chunk size must be positive")

    histogram: Counter[int] = Counter()
    point_count = 0
    overlap_flag_point_count = 0
    z_min = inf
    z_max = -inf
    z_chunks: list[np.ndarray[Any, np.dtype[np.floating[Any]]]] = []

    with laspy.open(path) as reader:
        for points in reader.chunk_iterator(chunk_size):
            x = np.asarray(points.x)
            y = np.asarray(points.y)
            inside = (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)
            if not np.any(inside):
                continue

            z = np.asarray(points.z)[inside]
            classifications = np.asarray(points.classification, dtype=np.uint8)[inside]
            overlap_flag_point_count += int(np.count_nonzero(np.asarray(points.overlap)[inside]))
            values, counts = np.unique(classifications, return_counts=True)
            histogram.update(
                {int(value): int(count) for value, count in zip(values, counts, strict=True)}
            )
            point_count += len(z)
            z_min = min(z_min, float(np.min(z)))
            z_max = max(z_max, float(np.max(z)))
            z_chunks.append(z)

    if point_count == 0:
        raise ValueError("LiDAR crop contains no points")

    area_m2 = (max_x - min_x) * (max_y - min_y)
    elevations = np.concatenate(z_chunks)
    z_p01, z_p50, z_p95, z_p99 = np.percentile(elevations, [1, 50, 95, 99])
    return LidarCropSummary(
        bbox=bbox,
        area_m2=area_m2,
        point_count=point_count,
        point_density_m2=point_count / area_m2,
        z_min_m=z_min,
        z_p01_m=float(z_p01),
        z_p50_m=float(z_p50),
        z_p95_m=float(z_p95),
        z_p99_m=float(z_p99),
        z_max_m=z_max,
        class_histogram={str(key): histogram[key] for key in sorted(histogram)},
        ground_point_count=histogram[GROUND_CLASS],
        vegetation_point_count=sum(histogram[key] for key in VEGETATION_CLASSES),
        building_point_count=histogram[BUILDING_CLASS],
        overlap_flag_point_count=overlap_flag_point_count,
        legacy_overlap_class_point_count=histogram[12],
    )
