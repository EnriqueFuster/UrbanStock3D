from pathlib import Path

import laspy
import numpy as np
import pytest

from urbanstock3d.reconstruction.quality.lidar import (
    assess_lidar_quality,
    rasterize_coverage,
)

FOOTPRINT = ((((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (0.0, 0.0)),),)


def test_rasterized_coverage_and_largest_hole_are_spatial_metrics(tmp_path: Path) -> None:
    path = tmp_path / "building.las"
    header = laspy.LasHeader(point_format=6, version="1.4")
    cloud = laspy.LasData(header)
    cloud.x = np.array([0.5, 1.5, 2.5, 3.5, 6.0])
    cloud.y = np.array([0.5, 0.5, 0.5, 0.5, 6.0])
    cloud.z = np.array([10.0, 10.1, 10.2, 30.0, 2.0])
    cloud.classification = np.array([6, 6, 6, 6, 2], dtype=np.uint8)
    cloud.write(path)

    report, grid = assess_lidar_quality(path, FOOTPRINT)

    assert report.point_count == 4
    assert report.roof_point_count == 4
    assert report.density_roof_per_m2 == pytest.approx(0.25)
    assert report.coverage_100m == pytest.approx(0.25)
    assert report.coverage_050m == pytest.approx(4 / 64)
    assert report.largest_hole_ratio == pytest.approx(12 / 16)
    assert report.nn_spacing_median_m == pytest.approx(1.0)
    assert report.outlier_ratio == pytest.approx(0.25)
    assert report.quality_class == "UNCLASSIFIED"
    assert grid.occupied_mask.sum() == 4


def test_coverage_rejects_non_positive_resolution() -> None:
    with pytest.raises(ValueError, match="positive"):
        rasterize_coverage(
            np.array([1.0]),
            np.array([1.0]),
            FOOTPRINT,
            resolution_m=0,
        )
