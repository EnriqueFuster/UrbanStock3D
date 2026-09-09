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


@dataclass(frozen=True)
class BuildingLidarSummary:
    """Ground-normalized LiDAR measurements for one building footprint."""

    footprint_area_m2: float
    footprint_point_count: int
    usable_point_count: int
    usable_point_density_m2: float
    class_histogram: dict[str, int]
    building_point_count: int
    vegetation_point_count: int
    legacy_overlap_class_point_count: int
    context_ground_point_count: int
    ground_elevation_p50_m: float
    roof_elevation_p50_m: float
    roof_elevation_p95_m: float
    height_p50_m: float
    height_p95_m: float

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


def write_lidar_bbox_crop(
    source: Path,
    destination: Path,
    bbox: tuple[float, float, float, float],
    *,
    chunk_size: int = 1_000_000,
) -> int:
    """Write a spatial subset while preserving the source point format and metadata."""
    min_x, min_y, max_x, max_y = bbox
    if min_x >= max_x or min_y >= max_y:
        raise ValueError("LiDAR crop bbox must have positive area")
    if chunk_size <= 0:
        raise ValueError("LiDAR chunk size must be positive")
    if source.resolve() == destination.resolve():
        raise ValueError("LiDAR crop destination must differ from its source")

    destination.parent.mkdir(parents=True, exist_ok=True)
    point_count = 0
    with laspy.open(source) as reader:
        with laspy.open(destination, mode="w", header=reader.header) as writer:
            for points in reader.chunk_iterator(chunk_size):
                x = np.asarray(points.x)
                y = np.asarray(points.y)
                inside = (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)
                if not np.any(inside):
                    continue
                selected = points[inside]
                writer.write_points(selected)
                point_count += len(selected)

    if point_count == 0:
        destination.unlink(missing_ok=True)
        raise ValueError("LiDAR crop contains no points")
    return point_count


def write_lidar_bbox_crop_from_sources(
    sources: tuple[Path, ...],
    destination: Path,
    bbox: tuple[float, float, float, float],
    *,
    chunk_size: int = 1_000_000,
) -> int:
    """Crop and merge compatible LAS/LAZ sources into one context point cloud."""
    if not sources:
        raise ValueError("At least one LiDAR source is required")
    min_x, min_y, max_x, max_y = bbox
    if min_x >= max_x or min_y >= max_y:
        raise ValueError("LiDAR crop bbox must have positive area")
    if chunk_size <= 0:
        raise ValueError("LiDAR chunk size must be positive")
    resolved_destination = destination.resolve()
    if any(source.resolve() == resolved_destination for source in sources):
        raise ValueError("LiDAR crop destination must differ from its sources")

    destination.parent.mkdir(parents=True, exist_ok=True)
    point_count = 0
    with laspy.open(sources[0]) as first_reader:
        output_header = first_reader.header
    with laspy.open(destination, mode="w", header=output_header) as writer:
        for source in sources:
            with laspy.open(source) as reader:
                if reader.header.point_format != output_header.point_format:
                    raise ValueError("LiDAR source point formats are incompatible")
                for points in reader.chunk_iterator(chunk_size):
                    x = np.asarray(points.x)
                    y = np.asarray(points.y)
                    inside = (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)
                    if not np.any(inside):
                        continue
                    selected = points[inside]
                    selected.change_scaling(
                        scales=output_header.scales,
                        offsets=output_header.offsets,
                    )
                    writer.write_points(selected)
                    point_count += len(selected)

    if point_count == 0:
        destination.unlink(missing_ok=True)
        raise ValueError("LiDAR crop contains no points")
    return point_count


def summarize_building_lidar(
    path: Path,
    rings: tuple[tuple[tuple[float, float], ...], ...],
    context_bbox: tuple[float, float, float, float],
    *,
    chunk_size: int = 1_000_000,
) -> BuildingLidarSummary:
    """Measure classified building points against nearby classified ground."""
    if not rings or len(rings[0]) < 4:
        raise ValueError("Building footprint must contain a valid exterior ring")

    histogram: Counter[int] = Counter()
    footprint_point_count = 0
    ground_chunks: list[np.ndarray[Any, np.dtype[np.floating[Any]]]] = []
    roof_chunks: list[np.ndarray[Any, np.dtype[np.floating[Any]]]] = []
    min_x, min_y, max_x, max_y = context_bbox

    with laspy.open(path) as reader:
        for points in reader.chunk_iterator(chunk_size):
            x = np.asarray(points.x)
            y = np.asarray(points.y)
            in_context = (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)
            if not np.any(in_context):
                continue

            x = x[in_context]
            y = y[in_context]
            z = np.asarray(points.z)[in_context]
            classifications = np.asarray(points.classification, dtype=np.uint8)[in_context]
            ground_chunks.append(z[classifications == GROUND_CLASS])

            in_footprint = points_in_polygon(x, y, rings)
            footprint_classes = classifications[in_footprint]
            footprint_z = z[in_footprint]
            values, counts = np.unique(footprint_classes, return_counts=True)
            histogram.update(
                {int(value): int(count) for value, count in zip(values, counts, strict=True)}
            )
            footprint_point_count += len(footprint_z)
            roof_chunks.append(footprint_z[footprint_classes == BUILDING_CLASS])

    ground = _non_empty_values(ground_chunks, "context contains no classified ground points")
    roof = _non_empty_values(roof_chunks, "footprint contains no classified building points")
    ground_p50 = float(np.percentile(ground, 50))
    roof_p50, roof_p95 = np.percentile(roof, [50, 95])
    footprint_area = polygon_area(rings)
    unusable_count = histogram[12]
    usable_count = footprint_point_count - unusable_count
    return BuildingLidarSummary(
        footprint_area_m2=footprint_area,
        footprint_point_count=footprint_point_count,
        usable_point_count=usable_count,
        usable_point_density_m2=usable_count / footprint_area,
        class_histogram={str(key): histogram[key] for key in sorted(histogram)},
        building_point_count=histogram[BUILDING_CLASS],
        vegetation_point_count=sum(histogram[key] for key in VEGETATION_CLASSES),
        legacy_overlap_class_point_count=unusable_count,
        context_ground_point_count=len(ground),
        ground_elevation_p50_m=ground_p50,
        roof_elevation_p50_m=float(roof_p50),
        roof_elevation_p95_m=float(roof_p95),
        height_p50_m=float(roof_p50) - ground_p50,
        height_p95_m=float(roof_p95) - ground_p50,
    )


def summarize_multipolygon_building_lidar(
    path: Path,
    polygons: tuple[tuple[tuple[tuple[float, float], ...], ...], ...],
    context_bbox: tuple[float, float, float, float],
    *,
    chunk_size: int = 1_000_000,
) -> BuildingLidarSummary:
    """Measure one building whose footprint can have multiple polygons."""
    if len(polygons) == 1:
        return summarize_building_lidar(path, polygons[0], context_bbox, chunk_size=chunk_size)
    if not polygons:
        raise ValueError("Building footprint must contain at least one polygon")

    histogram: Counter[int] = Counter()
    footprint_point_count = 0
    ground_chunks: list[np.ndarray[Any, np.dtype[np.floating[Any]]]] = []
    roof_chunks: list[np.ndarray[Any, np.dtype[np.floating[Any]]]] = []
    min_x, min_y, max_x, max_y = context_bbox
    with laspy.open(path) as reader:
        for points in reader.chunk_iterator(chunk_size):
            x = np.asarray(points.x)
            y = np.asarray(points.y)
            in_context = (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)
            if not np.any(in_context):
                continue
            x = x[in_context]
            y = y[in_context]
            z = np.asarray(points.z)[in_context]
            classifications = np.asarray(points.classification, dtype=np.uint8)[in_context]
            ground_chunks.append(z[classifications == GROUND_CLASS])
            in_footprint = np.zeros(x.shape, dtype=np.bool_)
            for rings in polygons:
                in_footprint |= points_in_polygon(x, y, rings)
            footprint_classes = classifications[in_footprint]
            footprint_z = z[in_footprint]
            values, counts = np.unique(footprint_classes, return_counts=True)
            histogram.update(
                {int(value): int(count) for value, count in zip(values, counts, strict=True)}
            )
            footprint_point_count += len(footprint_z)
            roof_chunks.append(footprint_z[footprint_classes == BUILDING_CLASS])

    ground = _non_empty_values(ground_chunks, "context contains no classified ground points")
    roof = _non_empty_values(roof_chunks, "footprint contains no classified building points")
    ground_p50 = float(np.percentile(ground, 50))
    roof_p50, roof_p95 = np.percentile(roof, [50, 95])
    footprint_area = sum(polygon_area(rings) for rings in polygons)
    unusable_count = histogram[12]
    usable_count = footprint_point_count - unusable_count
    return BuildingLidarSummary(
        footprint_area_m2=footprint_area,
        footprint_point_count=footprint_point_count,
        usable_point_count=usable_count,
        usable_point_density_m2=usable_count / footprint_area,
        class_histogram={str(key): histogram[key] for key in sorted(histogram)},
        building_point_count=histogram[BUILDING_CLASS],
        vegetation_point_count=sum(histogram[key] for key in VEGETATION_CLASSES),
        legacy_overlap_class_point_count=unusable_count,
        context_ground_point_count=len(ground),
        ground_elevation_p50_m=ground_p50,
        roof_elevation_p50_m=float(roof_p50),
        roof_elevation_p95_m=float(roof_p95),
        height_p50_m=float(roof_p50) - ground_p50,
        height_p95_m=float(roof_p95) - ground_p50,
    )


def points_in_polygon(
    x: np.ndarray[Any, np.dtype[np.floating[Any]]],
    y: np.ndarray[Any, np.dtype[np.floating[Any]]],
    rings: tuple[tuple[tuple[float, float], ...], ...],
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Return a mask for points in a polygon, excluding interior rings."""
    inside = _points_in_ring(x, y, rings[0])
    for hole in rings[1:]:
        inside &= ~_points_in_ring(x, y, hole)
    return inside


def polygon_area(rings: tuple[tuple[tuple[float, float], ...], ...]) -> float:
    """Calculate polygon area from one exterior and optional interior rings."""
    exterior_area = abs(_signed_ring_area(rings[0]))
    holes_area = sum(abs(_signed_ring_area(ring)) for ring in rings[1:])
    area = exterior_area - holes_area
    if area <= 0:
        raise ValueError("Building footprint must have positive area")
    return area


def _points_in_ring(
    x: np.ndarray[Any, np.dtype[np.floating[Any]]],
    y: np.ndarray[Any, np.dtype[np.floating[Any]]],
    ring: tuple[tuple[float, float], ...],
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    inside = np.zeros(x.shape, dtype=np.bool_)
    for start, end in zip(ring, ring[1:], strict=False):
        x1, y1 = start
        x2, y2 = end
        crosses = (y1 > y) != (y2 > y)
        intersection_x = np.zeros(x.shape, dtype=np.float64)
        np.divide((x2 - x1) * (y - y1), y2 - y1, out=intersection_x, where=crosses)
        intersection_x += x1
        inside ^= crosses & (x < intersection_x)
    return inside


def _signed_ring_area(ring: tuple[tuple[float, float], ...]) -> float:
    return 0.5 * sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(ring, ring[1:], strict=False))


def _non_empty_values(
    chunks: list[np.ndarray[Any, np.dtype[np.floating[Any]]]], error_message: str
) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
    non_empty = [chunk for chunk in chunks if len(chunk)]
    if not non_empty:
        raise ValueError(error_message)
    return np.concatenate(non_empty)
