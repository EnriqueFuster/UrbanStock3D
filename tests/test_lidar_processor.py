from pathlib import Path

import laspy

from urbanstock3d.processors.lidar import inspect_lidar_header


def test_inspect_lidar_header_reads_metadata_without_loading_points(tmp_path: Path) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    path = tmp_path / "sample.las"
    laspy.LasData(header).write(path)

    metadata = inspect_lidar_header(path)

    assert metadata["las_version"] == "1.2"
    assert metadata["point_format"] == 3
    assert metadata["point_count"] == 0
    assert "X" in metadata["dimensions"]
