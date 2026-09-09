from dataclasses import replace

import pytest

from urbanstock3d.reconstruction.evidence import HeightEvidenceMetrics
from urbanstock3d.reconstruction.models import LidarQualityReport
from urbanstock3d.reconstruction.quality.roof_hypothesis import (
    RoofFormHypothesis,
    RoofHypothesisParameters,
    infer_roof_hypothesis,
)

LIDAR = LidarQualityReport(
    available=True,
    point_count=1_000,
    roof_point_count=800,
    density_all_per_m2=6.0,
    density_roof_per_m2=5.0,
    coverage_050m=0.9,
    coverage_100m=0.95,
    largest_hole_ratio=0.02,
    nn_spacing_median_m=0.3,
    nn_spacing_p90_m=0.5,
    planar_support_ratio=0.85,
    local_residual_median_m=0.05,
    outlier_ratio=0.01,
    footprint_alignment_score=0.95,
    estimated_shift_x_m=0.0,
    estimated_shift_y_m=0.0,
    quality_class="GOOD",
    score=0.9,
)

HEIGHT = HeightEvidenceMetrics(
    resolution_m=0.5,
    footprint_cell_count=100,
    valid_cell_count=90,
    valid_ratio=0.9,
    adjacent_pair_count=150,
    gradient_p50=0.12,
    gradient_p95=0.5,
    discontinuity_threshold_m=1.0,
    discontinuity_ratio=0.03,
)


@pytest.mark.parametrize(
    ("height", "expected"),
    [
        (
            replace(HEIGHT, gradient_p50=0.02, gradient_p95=0.1),
            RoofFormHypothesis.FLAT,
        ),
        (HEIGHT, RoofFormHypothesis.PITCHED),
        (
            replace(HEIGHT, gradient_p95=1.4, discontinuity_ratio=0.2),
            RoofFormHypothesis.COMPOUND,
        ),
    ],
)
def test_infers_supported_roof_forms(
    height: HeightEvidenceMetrics,
    expected: RoofFormHypothesis,
) -> None:
    report = infer_roof_hypothesis(LIDAR, height)

    assert report.hypothesis is expected
    assert report.evidence_score > 0.8
    assert report.reasons


def test_abstains_when_spatial_support_is_insufficient() -> None:
    report = infer_roof_hypothesis(
        replace(LIDAR, density_roof_per_m2=0.5),
        replace(HEIGHT, valid_ratio=0.4),
    )

    assert report.hypothesis is RoofFormHypothesis.INSUFFICIENT_EVIDENCE
    assert len(report.reasons) == 2


def test_keeps_ambiguous_evidence_uncertain() -> None:
    report = infer_roof_hypothesis(
        replace(LIDAR, planar_support_ratio=0.4),
        replace(HEIGHT, gradient_p50=0.05, gradient_p95=0.5),
    )

    assert report.hypothesis is RoofFormHypothesis.UNCERTAIN


def test_rejects_invalid_hypothesis_thresholds() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        RoofHypothesisParameters(minimum_valid_ratio=1.1)
