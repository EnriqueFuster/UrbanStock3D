from pathlib import Path

import laspy
import numpy as np
import pytest

from urbanstock3d.processors.lidar import (
    inspect_lidar_header,
    points_in_polygon,
    polygon_area,
    summarize_building_lidar,
    summarize_lidar_bbox,
)


def test_inspect_lidar_header_reads_metadata_without_loading_points(tmp_path: Path) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    path = tmp_path / "sample.las"
    laspy.LasData(header).write(path)

    metadata = inspect_lidar_header(path)

    assert metadata["las_version"] == "1.2"
    assert metadata["point_format"] == 3
    assert metadata["point_count"] == 0
    assert "X" in metadata["dimensions"]


def test_summarize_lidar_bbox_counts_classes_and_density(tmp_path: Path) -> None:
    header = laspy.LasHeader(point_format=6, version="1.4")
    points = laspy.LasData(header)
    points.x = np.array([0.0, 1.0, 5.0, 10.0])
    points.y = np.array([0.0, 1.0, 5.0, 10.0])
    points.z = np.array([2.0, 8.0, 12.0, 20.0])
    points.classification = np.array([2, 6, 5, 1], dtype=np.uint8)
    path = tmp_path / "sample.las"
    points.write(path)

    summary = summarize_lidar_bbox(path, (-1.0, -1.0, 6.0, 6.0), chunk_size=2)

    assert summary.point_count == 3
    assert summary.point_density_m2 == pytest.approx(3 / 49)
    assert summary.class_histogram == {"2": 1, "5": 1, "6": 1}
    assert summary.ground_point_count == 1
    assert summary.vegetation_point_count == 1
    assert summary.building_point_count == 1
    assert summary.z_min_m == 2
    assert summary.z_p50_m == 8
    assert summary.z_p95_m == pytest.approx(11.6)
    assert summary.z_max_m == 12
    assert summary.overlap_flag_point_count == 0
    assert summary.legacy_overlap_class_point_count == 0


def test_summarize_lidar_bbox_rejects_empty_crop(tmp_path: Path) -> None:
    path = tmp_path / "empty.las"
    laspy.LasData(laspy.LasHeader(point_format=6, version="1.4")).write(path)

    with pytest.raises(ValueError, match="contains no points"):
        summarize_lidar_bbox(path, (0.0, 0.0, 1.0, 1.0))


def test_points_in_polygon_excludes_hole() -> None:
    rings = (
        ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)),
        ((4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0), (4.0, 4.0)),
    )

    mask = points_in_polygon(np.array([2.0, 5.0, 12.0]), np.array([2.0, 5.0, 5.0]), rings)

    assert mask.tolist() == [True, False, False]
    assert polygon_area(rings) == 96


def test_summarize_building_lidar_normalizes_roof_height(tmp_path: Path) -> None:
    header = laspy.LasHeader(point_format=6, version="1.4")
    points = laspy.LasData(header)
    points.x = np.array([1.0, 2.0, 3.0, 8.0, 9.0])
    points.y = np.array([1.0, 2.0, 3.0, 8.0, 9.0])
    points.z = np.array([20.0, 22.0, 24.0, 10.0, 12.0])
    points.classification = np.array([6, 6, 12, 2, 2], dtype=np.uint8)
    path = tmp_path / "building.las"
    points.write(path)
    rings = (((0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0), (0.0, 0.0)),)

    summary = summarize_building_lidar(path, rings, (0.0, 0.0, 10.0, 10.0), chunk_size=2)

    assert summary.footprint_area_m2 == 25
    assert summary.footprint_point_count == 3
    assert summary.usable_point_count == 2
    assert summary.building_point_count == 2
    assert summary.ground_elevation_p50_m == 11
    assert summary.roof_elevation_p50_m == 21
    assert summary.height_p50_m == 10
