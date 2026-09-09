from pathlib import Path

import laspy
import numpy as np
import pytest
from pyproj import CRS

from urbanstock3d.reconstruction.evidence import (
    build_lidar_height_rasters,
    save_height_raster,
)

FOOTPRINT = ((((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0)),),)


def test_builds_observed_height_without_interpolating_empty_cells(tmp_path: Path) -> None:
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.add_crs(CRS("EPSG:25830"))
    cloud = laspy.LasData(header)
    cloud.x = np.array([0.2, 0.4, 1.2, 3.0])
    cloud.y = np.array([0.2, 0.4, 0.2, 3.0])
    cloud.z = np.array([10.0, 12.0, 20.0, 5.0])
    cloud.classification = np.array([6, 6, 6, 2], dtype=np.uint8)
    path = tmp_path / "crop.las"
    cloud.write(path)

    (raster,) = build_lidar_height_rasters(
        path,
        FOOTPRINT,
        resolutions_m=(1.0,),
        height_percentile=50,
    )

    assert raster.crs == "EPSG:25830"
    assert raster.elevation_m.shape == (2, 2)
    assert raster.elevation_m[0, 0] == pytest.approx(11.0)
    assert raster.elevation_m[0, 1] == pytest.approx(20.0)
    assert np.isnan(raster.elevation_m[1, 0])
    assert raster.point_count.tolist() == [[2, 1], [0, 0]]
    assert raster.valid_ratio == pytest.approx(0.5)

    destination = save_height_raster(raster, tmp_path / "height.npz")
    saved = np.load(destination)
    assert saved["valid_mask"].tolist() == [[True, True], [False, False]]
    assert float(saved["resolution_m"]) == 1.0


def test_height_raster_requires_declared_crs(tmp_path: Path) -> None:
    cloud = laspy.LasData(laspy.LasHeader(point_format=6, version="1.4"))
    path = tmp_path / "missing-crs.las"
    cloud.write(path)

    with pytest.raises(ValueError, match="declared CRS"):
        build_lidar_height_rasters(path, FOOTPRINT)
