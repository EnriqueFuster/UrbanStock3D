from dataclasses import replace

import pytest

from urbanstock3d.reconstruction.validation import (
    CityJsonQualityReport,
    LidarFitReport,
    QualityConfidence,
    ReconstructionQualityParameters,
    evaluate_reconstruction_quality,
)

GEOMETRY = CityJsonQualityReport(
    "2.2",
    8,
    6,
    {"GroundSurface": 1, "WallSurface": 4, "RoofSurface": 1},
    True,
    0,
    0.0,
    10.0,
    1.0,
    1.0,
    True,
    True,
    True,
    (),
    (),
)
FIT = LidarFitReport(1_000, 900, 900, 2, 0.1, 0.2, 0.4, 0.8, 0.95, 0.99)


def test_accepts_high_confidence_geometry_and_lidar_fit() -> None:
    report = evaluate_reconstruction_quality(GEOMETRY, FIT)

    assert report.accepted
    assert report.confidence_class is QualityConfidence.HIGH
    assert not report.failures


def test_accepts_moderate_fit_with_medium_confidence() -> None:
    report = evaluate_reconstruction_quality(
        GEOMETRY,
        replace(FIT, distance_rmse_m=0.8, distance_p95_m=1.5, within_050m_ratio=0.8),
    )

    assert report.accepted
    assert report.confidence_class is QualityConfidence.MEDIUM


def test_rejects_poor_fit_with_explicit_failures() -> None:
    report = evaluate_reconstruction_quality(
        GEOMETRY,
        replace(FIT, distance_rmse_m=4.0, distance_p95_m=8.0, within_050m_ratio=0.4),
    )

    assert not report.accepted
    assert report.confidence_class is QualityConfidence.LOW
    assert len(report.failures) == 3


def test_rejects_invalid_quality_threshold() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        ReconstructionQualityParameters(minimum_within_050m_ratio=1.1)
