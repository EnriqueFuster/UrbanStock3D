"""Align official terrain elevations with observed LiDAR height grids."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS

from urbanstock3d.reconstruction.evidence.lidar_height import ObservedHeightRaster


@dataclass(frozen=True)
class NormalizedHeightRaster:
    """Ground-normalized LiDAR height with explicit joint-validity mask."""

    height_m: np.ndarray[Any, np.dtype[np.floating[Any]]]
    terrain_elevation_m: np.ndarray[Any, np.dtype[np.floating[Any]]]
    footprint_mask: np.ndarray[Any, np.dtype[np.bool_]]
    valid_mask: np.ndarray[Any, np.dtype[np.bool_]]
    origin_x: float
    origin_y: float
    resolution_m: float
    crs: str
    terrain_source: str

    def __post_init__(self) -> None:
        shape = self.height_m.shape
        if self.terrain_elevation_m.shape != shape:
            raise ValueError("Terrain and normalized-height arrays must share a shape")
        if self.footprint_mask.shape != shape or self.valid_mask.shape != shape:
            raise ValueError("Raster masks must share the height-array shape")
        if np.any(self.valid_mask & ~self.footprint_mask):
            raise ValueError("Valid cells must lie inside the building footprint")

    @property
    def valid_ratio(self) -> float:
        """Return valid nDSM cells relative to cells inside the footprint."""
        footprint_cells = int(np.count_nonzero(self.footprint_mask))
        if footprint_cells == 0:
            return 0.0
        return float(np.count_nonzero(self.valid_mask) / footprint_cells)


def build_normalized_height_raster(
    observed: ObservedHeightRaster,
    terrain_source: str | Path,
) -> NormalizedHeightRaster:
    """Sample an official terrain COG at cell centres and subtract it from LiDAR heights."""
    rows, cols = observed.elevation_m.shape
    center_x = observed.origin_x + (np.arange(cols) + 0.5) * observed.resolution_m
    center_y = observed.origin_y + (np.arange(rows) + 0.5) * observed.resolution_m
    mesh_x, mesh_y = np.meshgrid(center_x, center_y)
    coordinates = np.column_stack((mesh_x.ravel(), mesh_y.ravel()))
    source = str(terrain_source)
    with rasterio.open(source) as dataset:
        if dataset.crs is None:
            raise ValueError("Terrain raster requires a declared CRS")
        terrain_crs = dataset.crs.to_string()
        if dataset.crs != CRS.from_string(observed.crs):
            raise ValueError(
                f"Terrain CRS {terrain_crs} does not match observed CRS {observed.crs}"
            )
        sampled = np.ma.vstack(tuple(dataset.sample(coordinates, indexes=1, masked=True)))
        terrain_values = sampled[:, 0].filled(np.nan).reshape(rows, cols).astype(np.float64)
    valid = observed.valid_mask & np.isfinite(terrain_values)
    if not np.any(valid):
        raise ValueError("Terrain raster has no valid overlap with observed height cells")
    normalized = np.full(observed.elevation_m.shape, np.nan, dtype=np.float64)
    normalized[valid] = observed.elevation_m[valid] - terrain_values[valid]
    return NormalizedHeightRaster(
        height_m=normalized,
        terrain_elevation_m=terrain_values,
        footprint_mask=observed.footprint_mask,
        valid_mask=valid,
        origin_x=observed.origin_x,
        origin_y=observed.origin_y,
        resolution_m=observed.resolution_m,
        crs=observed.crs,
        terrain_source=source,
    )


def save_normalized_height_raster(
    raster: NormalizedHeightRaster,
    destination: Path,
) -> Path:
    """Persist nDSM values, sampled terrain and the joint-validity mask."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        height_m=raster.height_m,
        terrain_elevation_m=raster.terrain_elevation_m,
        footprint_mask=raster.footprint_mask,
        valid_mask=raster.valid_mask,
        origin=np.array([raster.origin_x, raster.origin_y]),
        resolution_m=np.array(raster.resolution_m),
        crs=np.array(raster.crs),
        terrain_source=np.array(raster.terrain_source),
    )
    return destination
