from dataclasses import replace

import pytest

from urbanstock3d.reconstruction.evidence import HeightEvidenceMetrics
from urbanstock3d.reconstruction.quality.ransac import RoofPlaneEvidence
from urbanstock3d.reconstruction.quality.roof_complexity import (
    RoofComplexityClass,
    RoofComplexityParameters,
    assess_roof_complexity,
)

RECTANGLE = ((((0.0, 0.0), (10.0, 0.0), (10.0, 8.0), (0.0, 8.0), (0.0, 0.0)),),)
HEIGHT = HeightEvidenceMetrics(0.5, 320, 300, 0.9375, 500, 0.1, 0.3, 1.0, 0.02)
PLANES = RoofPlaneEvidence(1_000, 1_000, 950, 0.95, 1, 1, ())


def test_simple_supported_roof_remains_simple() -> None:
    report = assess_roof_complexity(
        RECTANGLE,
        PLANES,
        height=HEIGHT,
        building_part_count=1,
        height_range_m=1.5,
    )

    assert report.complexity_class is RoofComplexityClass.SIMPLE
    assert report.footprint_area_m2 == pytest.approx(80.0)
    assert report.footprint_vertices == 4
    assert report.score < 0.25


def test_multiple_planes_parts_steps_and_unexplained_points_raise_complexity() -> None:
    report = assess_roof_complexity(
        RECTANGLE,
        replace(
            PLANES,
            explained_point_count=400,
            explained_ratio=0.4,
            preliminary_plane_count=8,
            dominant_normal_count=4,
        ),
        height=replace(HEIGHT, discontinuity_ratio=0.3),
        building_part_count=5,
    )

    assert report.complexity_class is RoofComplexityClass.VERY_COMPLEX
    assert report.score >= 0.75
    assert "multiple preliminary planes" in report.reasons


def test_assessment_can_use_available_modalities_without_ndsm() -> None:
    report = assess_roof_complexity(RECTANGLE, PLANES)

    assert report.ndsm_discontinuity_ratio is None
    assert report.complexity_class is RoofComplexityClass.SIMPLE


def test_rejects_invalid_complexity_threshold_order() -> None:
    with pytest.raises(ValueError, match="increasing"):
        RoofComplexityParameters(simple_score_max=0.6, moderate_score_max=0.5)
