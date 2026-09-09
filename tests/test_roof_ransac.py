import numpy as np
import pytest

from urbanstock3d.reconstruction.quality.ransac import RansacParameters, detect_roof_planes

PARAMETERS = RansacParameters(
    sample_size=1_000,
    max_iterations=150,
    max_planes=4,
    distance_threshold_m=0.03,
    minimum_support_points=20,
    minimum_support_ratio=0.10,
    random_seed=7,
)


def plane_points(
    slope_x: float,
    slope_y: float,
    intercept: float,
    *,
    x_offset: float = 0.0,
) -> np.ndarray:
    x, y = np.meshgrid(np.linspace(0, 4, 10) + x_offset, np.linspace(0, 4, 10))
    z = slope_x * x + slope_y * y + intercept
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


def test_detects_one_clean_plane_with_metric_orientation() -> None:
    evidence = detect_roof_planes(plane_points(0.2, 0.0, 10.0), parameters=PARAMETERS)

    assert evidence.preliminary_plane_count == 1
    assert evidence.dominant_normal_count == 1
    assert evidence.explained_ratio == pytest.approx(1.0)
    assert evidence.planes[0].slope_deg == pytest.approx(11.31, abs=0.1)
    assert evidence.planes[0].residual_p95_m < 1e-10


def test_separates_two_differently_oriented_roof_planes() -> None:
    points = np.vstack(
        (
            plane_points(0.3, 0.0, 10.0),
            plane_points(-0.3, 0.0, 14.0, x_offset=5.0),
        )
    )

    evidence = detect_roof_planes(points, parameters=PARAMETERS)

    assert evidence.preliminary_plane_count == 2
    assert evidence.dominant_normal_count == 2
    assert evidence.explained_ratio == pytest.approx(1.0)
    assert sum(plane.support_point_count for plane in evidence.planes) == 200
    assert all(plane.support_point_count >= 80 for plane in evidence.planes)


def test_reports_insufficient_point_support_without_inventing_a_plane() -> None:
    evidence = detect_roof_planes(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]))

    assert evidence.preliminary_plane_count == 0
    assert evidence.warnings == ("too few points",)


def test_rejects_invalid_ransac_bounds() -> None:
    with pytest.raises(ValueError, match="positive"):
        RansacParameters(max_iterations=0)
