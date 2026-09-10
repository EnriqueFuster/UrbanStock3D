import numpy as np
import pytest

from urbanstock3d.reconstruction.validation import (
    RoofObservationParameters,
    select_roof_observations,
)


def test_rejects_low_returns_below_local_upper_envelope() -> None:
    points = np.array(
        [
            [0.0, 0.0, 10.0],
            [0.2, 0.0, 10.1],
            [0.0, 0.2, 9.9],
            [0.2, 0.2, 10.0],
            [0.1, 0.1, 4.0],
        ]
    )

    selected, report = select_roof_observations(points)

    assert selected.tolist() == [True, True, True, True, False]
    assert report.source_point_count == 5
    assert report.selected_point_count == 4
    assert report.rejected_point_count == 1


def test_keeps_sparse_observations_without_enough_local_evidence() -> None:
    points = np.array([[0.0, 0.0, 10.0], [10.0, 10.0, 2.0]])

    selected, report = select_roof_observations(points)

    assert selected.all()
    assert report.selected_ratio == pytest.approx(1.0)


def test_rejects_invalid_observation_parameters() -> None:
    with pytest.raises(ValueError, match="radius"):
        RoofObservationParameters(radius_m=0)
