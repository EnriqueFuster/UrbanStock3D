from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from urbanstock3d.reconstruction.evidence import (
    ObservedHeightRaster,
    build_normalized_height_raster,
    save_normalized_height_raster,
)


def observed_raster() -> ObservedHeightRaster:
    return ObservedHeightRaster(
        elevation_m=np.array([[100.0, 102.0], [np.nan, 112.0]]),
        point_count=np.array([[1, 1], [0, 2]], dtype=np.int32),
        footprint_mask=np.ones((2, 2), dtype=np.bool_),
        valid_mask=np.array([[True, True], [False, True]]),
        origin_x=0.0,
        origin_y=0.0,
        resolution_m=1.0,
        crs="EPSG:25830",
        height_percentile=90.0,
    )


def write_terrain(path: Path, *, crs: str = "EPSG:25830") -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs=crs,
        transform=from_origin(0.0, 2.0, 1.0, 1.0),
        nodata=-9999.0,
    ) as dataset:
        dataset.write(np.array([[90.0, 90.0], [90.0, 91.0]], dtype=np.float32), 1)


def test_builds_ndsm_by_sampling_terrain_at_observed_cell_centres(tmp_path: Path) -> None:
    terrain = tmp_path / "terrain.tif"
    write_terrain(terrain)

    normalized = build_normalized_height_raster(observed_raster(), terrain)

    assert normalized.height_m[0, 0] == pytest.approx(10.0)
    assert normalized.height_m[0, 1] == pytest.approx(11.0)
    assert np.isnan(normalized.height_m[1, 0])
    assert normalized.height_m[1, 1] == pytest.approx(22.0)
    assert normalized.valid_ratio == pytest.approx(0.75)

    destination = save_normalized_height_raster(normalized, tmp_path / "ndsm.npz")
    with np.load(destination) as saved:
        assert saved["valid_mask"].tolist() == [[True, True], [False, True]]


def test_rejects_terrain_in_a_different_crs(tmp_path: Path) -> None:
    terrain = tmp_path / "terrain.tif"
    write_terrain(terrain, crs="EPSG:4326")

    with pytest.raises(ValueError, match="does not match"):
        build_normalized_height_raster(observed_raster(), terrain)


def test_rejects_terrain_without_valid_overlap(tmp_path: Path) -> None:
    terrain = tmp_path / "terrain.tif"
    write_terrain(terrain)
    with rasterio.open(terrain, "r+") as dataset:
        dataset.write(np.full((2, 2), -9999.0, dtype=np.float32), 1)

    with pytest.raises(ValueError, match="no valid overlap"):
        build_normalized_height_raster(observed_raster(), terrain)
