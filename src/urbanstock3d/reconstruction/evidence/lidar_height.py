"""Rasterize classified roof points while preserving missing-data masks."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import laspy
import numpy as np

from urbanstock3d.processors.lidar import BUILDING_CLASS, points_in_polygon
from urbanstock3d.reconstruction.quality.lidar import Footprint, rasterize_coverage


@dataclass(frozen=True)
class ObservedHeightRaster:
    """A metric height grid derived only from observed LiDAR roof points."""

    elevation_m: np.ndarray[Any, np.dtype[np.floating[Any]]]
    point_count: np.ndarray[Any, np.dtype[np.integer[Any]]]
    footprint_mask: np.ndarray[Any, np.dtype[np.bool_]]
    valid_mask: np.ndarray[Any, np.dtype[np.bool_]]
    origin_x: float
    origin_y: float
    resolution_m: float
    crs: str
    height_percentile: float

    def __post_init__(self) -> None:
        shape = self.elevation_m.shape
        if any(
            array.shape != shape
            for array in (self.point_count, self.footprint_mask, self.valid_mask)
        ):
            raise ValueError("Height raster arrays must have identical shapes")
        if self.resolution_m <= 0:
            raise ValueError("Height raster resolution must be positive")
        if not 0 < self.height_percentile <= 100:
            raise ValueError("Height percentile must be in the interval (0, 100]")
        if np.any(self.valid_mask & ~self.footprint_mask):
            raise ValueError("Valid height cells must be inside the footprint")

    @property
    def valid_ratio(self) -> float:
        """Return the fraction of footprint cells containing observed height."""
        footprint_cells = int(np.count_nonzero(self.footprint_mask))
        return float(np.count_nonzero(self.valid_mask) / footprint_cells)


def build_lidar_height_rasters(
    path: Path,
    footprint: Footprint,
    *,
    resolutions_m: tuple[float, ...] = (0.5, 1.0),
    height_percentile: float = 90.0,
) -> tuple[ObservedHeightRaster, ...]:
    """Build multi-resolution roof-height grids from one local LiDAR crop."""
    if not resolutions_m:
        raise ValueError("At least one height raster resolution is required")
    if len(set(resolutions_m)) != len(resolutions_m):
        raise ValueError("Height raster resolutions must be unique")
    cloud = laspy.read(path)
    crs = cloud.header.parse_crs()
    if crs is None:
        raise ValueError("LiDAR height raster requires a declared CRS")
    classification = np.asarray(cloud.classification, dtype=np.uint8)
    roof = classification == BUILDING_CLASS
    x = np.asarray(cloud.x)[roof]
    y = np.asarray(cloud.y)[roof]
    z = np.asarray(cloud.z)[roof]
    inside = _points_in_footprint(x, y, footprint)
    x, y, z = x[inside], y[inside], z[inside]
    return tuple(
        _rasterize_observed_heights(
            x,
            y,
            z,
            footprint,
            resolution_m=resolution,
            crs=crs.to_string(),
            height_percentile=height_percentile,
        )
        for resolution in resolutions_m
    )


def save_height_raster(raster: ObservedHeightRaster, destination: Path) -> Path:
    """Persist a compact raster and every mask required to interpret it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        elevation_m=raster.elevation_m,
        point_count=raster.point_count,
        footprint_mask=raster.footprint_mask,
        valid_mask=raster.valid_mask,
        origin=np.array([raster.origin_x, raster.origin_y]),
        resolution_m=np.array(raster.resolution_m),
        crs=np.array(raster.crs),
        height_percentile=np.array(raster.height_percentile),
    )
    return destination


def _rasterize_observed_heights(
    x: np.ndarray[Any, np.dtype[np.floating[Any]]],
    y: np.ndarray[Any, np.dtype[np.floating[Any]]],
    z: np.ndarray[Any, np.dtype[np.floating[Any]]],
    footprint: Footprint,
    *,
    resolution_m: float,
    crs: str,
    height_percentile: float,
) -> ObservedHeightRaster:
    if not 0 < height_percentile <= 100:
        raise ValueError("Height percentile must be in the interval (0, 100]")
    coverage = rasterize_coverage(x, y, footprint, resolution_m=resolution_m)
    rows, cols = coverage.footprint_mask.shape
    point_cols = np.floor((x - coverage.min_x) / resolution_m).astype(int)
    point_rows = np.floor((y - coverage.min_y) / resolution_m).astype(int)
    inside_grid = (point_cols >= 0) & (point_cols < cols) & (point_rows >= 0) & (point_rows < rows)
    point_cols = point_cols[inside_grid]
    point_rows = point_rows[inside_grid]
    z = z[inside_grid]
    inside_footprint = coverage.footprint_mask[point_rows, point_cols]
    point_cols = point_cols[inside_footprint]
    point_rows = point_rows[inside_footprint]
    z = z[inside_footprint]

    cell_ids = point_rows * cols + point_cols
    order = np.argsort(cell_ids)
    cell_ids = cell_ids[order]
    z = z[order]
    elevation = np.full((rows, cols), np.nan, dtype=np.float64)
    counts = np.zeros((rows, cols), dtype=np.int32)
    unique_cells, starts, cell_counts = np.unique(
        cell_ids,
        return_index=True,
        return_counts=True,
    )
    for cell_id, start, count in zip(unique_cells, starts, cell_counts, strict=True):
        row, col = divmod(int(cell_id), cols)
        counts[row, col] = int(count)
        elevation[row, col] = float(np.percentile(z[start : start + count], height_percentile))
    valid = coverage.footprint_mask & (counts > 0)
    return ObservedHeightRaster(
        elevation_m=elevation,
        point_count=counts,
        footprint_mask=coverage.footprint_mask,
        valid_mask=valid,
        origin_x=coverage.min_x,
        origin_y=coverage.min_y,
        resolution_m=resolution_m,
        crs=crs,
        height_percentile=height_percentile,
    )


def _points_in_footprint(
    x: np.ndarray[Any, np.dtype[np.floating[Any]]],
    y: np.ndarray[Any, np.dtype[np.floating[Any]]],
    footprint: Footprint,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    inside = np.zeros(x.shape, dtype=np.bool_)
    for polygon in footprint:
        inside |= points_in_polygon(x, y, polygon)
    return inside
