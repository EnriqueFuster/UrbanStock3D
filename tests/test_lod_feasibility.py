from dataclasses import replace

import pytest

from urbanstock3d.reconstruction.enums import LodRequest
from urbanstock3d.reconstruction.evidence import HeightEvidenceMetrics
from urbanstock3d.reconstruction.models import LidarQualityReport
from urbanstock3d.reconstruction.quality.lod_feasibility import (
    LodEvidenceAvailability,
    LodFeasibilityParameters,
    assess_lod_feasibility,
)
from urbanstock3d.reconstruction.quality.roof_complexity import (
    RoofComplexityClass,
    RoofComplexityReport,
)

AVAILABLE = LodEvidenceAvailability(True, True, True)
LIDAR = LidarQualityReport(
    True,
    1_000,
    800,
    6.0,
    5.0,
    0.9,
    0.9,
    0.05,
    0.3,
    0.5,
    0.8,
    0.05,
    0.01,
    0.95,
    0.0,
    0.0,
    "GOOD",
    0.9,
)
COMPLEXITY = RoofComplexityReport(
    100.0,
    6,
    0.75,
    2,
    4.0,
    3,
    2,
    0.75,
    0.08,
    RoofComplexityClass.MODERATE,
    0.4,
    ("multiple preliminary planes",),
)
HEIGHT = HeightEvidenceMetrics(0.5, 400, 360, 0.9, 600, 0.2, 0.7, 1.0, 0.08)


def test_strong_evidence_supports_every_lod() -> None:
    report = assess_lod_feasibility(AVAILABLE, LIDAR, COMPLEXITY, height=HEIGHT)

    assert report.lod12.supported
    assert report.lod13.supported
    assert report.lod22.supported
    assert report.highest_supported_lod is LodRequest.LOD22


def test_lod13_requires_observed_height_block_evidence() -> None:
    complexity = replace(COMPLEXITY, building_part_count=1)
    height = replace(HEIGHT, discontinuity_ratio=0.01)

    report = assess_lod_feasibility(AVAILABLE, LIDAR, complexity, height=height)

    assert report.lod12.supported
    assert not report.lod13.supported
    assert "insufficient major height-part evidence" in report.lod13.reasons


def test_sparse_geometry_blocks_lod22_but_preserves_lod12() -> None:
    sparse = replace(
        LIDAR,
        density_roof_per_m2=1.0,
        coverage_100m=0.5,
        largest_hole_ratio=0.6,
        planar_support_ratio=0.3,
    )

    report = assess_lod_feasibility(AVAILABLE, sparse, COMPLEXITY, height=HEIGHT)

    assert report.lod12.supported
    assert not report.lod22.supported
    assert any("LiDAR density" in reason for reason in report.lod22.reasons)


def test_missing_ground_prevents_all_lods() -> None:
    availability = replace(AVAILABLE, ground_elevation=False)

    report = assess_lod_feasibility(availability, LIDAR, COMPLEXITY, height=HEIGHT)

    assert not report.lod12.supported
    assert not report.lod13.supported
    assert not report.lod22.supported
    assert report.highest_supported_lod is None


def test_rejects_invalid_lod_thresholds() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        LodFeasibilityParameters(lod22_minimum_score=1.2)
