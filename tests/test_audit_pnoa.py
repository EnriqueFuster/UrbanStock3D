import pytest

from scripts.audit_pnoa import image_dimensions, projected_bbox


def test_projected_bbox_adds_metric_buffer() -> None:
    coordinates = [
        [
            [-0.3910992062, 39.4770188882],
            [-0.3908547752, 39.4770123841],
            [-0.3910992062, 39.4770188882],
        ]
    ]

    base_bbox = projected_bbox(coordinates, buffer_m=0)
    buffered_bbox = projected_bbox(coordinates, buffer_m=15)

    assert buffered_bbox[0] == pytest.approx(base_bbox[0] - 15)
    assert buffered_bbox[1] == pytest.approx(base_bbox[1] - 15)
    assert buffered_bbox[2] == pytest.approx(base_bbox[2] + 15)
    assert buffered_bbox[3] == pytest.approx(base_bbox[3] + 15)


def test_image_dimensions_preserve_requested_pixel_size() -> None:
    assert image_dimensions((0, 0, 50, 40), 0.25) == (200, 160)


@pytest.mark.parametrize("pixel_size", [0, -0.25])
def test_image_dimensions_reject_invalid_pixel_size(pixel_size: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        image_dimensions((0, 0, 50, 40), pixel_size)
