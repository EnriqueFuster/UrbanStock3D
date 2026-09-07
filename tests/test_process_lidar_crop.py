import pytest

from scripts.process_lidar_crop import buffered_bbox


def test_buffered_bbox_expands_all_sides() -> None:
    assert buffered_bbox((10.0, 20.0, 30.0, 40.0), 5.0) == (5.0, 15.0, 35.0, 45.0)


def test_buffered_bbox_rejects_negative_distance() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        buffered_bbox((10.0, 20.0, 30.0, 40.0), -1.0)
