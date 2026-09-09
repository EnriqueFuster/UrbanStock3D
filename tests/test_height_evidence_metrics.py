import numpy as np
import pytest

from urbanstock3d.reconstruction.evidence import (
    NormalizedHeightRaster,
    compare_height_resolutions,
    evaluate_height_evidence,
)


def normalized(
    heights: np.ndarray,
    resolution_m: float,
    *,
    valid_mask: np.ndarray | None = None,
) -> NormalizedHeightRaster:
    valid = np.ones(heights.shape, dtype=np.bool_) if valid_mask is None else valid_mask
    return NormalizedHeightRaster(
        height_m=heights.astype(np.float64),
        terrain_elevation_m=np.full(heights.shape, 90.0),
        footprint_mask=np.ones(heights.shape, dtype=np.bool_),
        valid_mask=valid,
        origin_x=0.0,
        origin_y=0.0,
        resolution_m=resolution_m,
        crs="EPSG:25830",
        terrain_source="terrain.tif",
    )


def test_evaluates_coverage_gradients_and_height_discontinuities() -> None:
    raster = normalized(np.array([[10.0, 10.5], [12.0, 14.0]]), 1.0)

    metrics = evaluate_height_evidence(raster, discontinuity_threshold_m=1.0)

    assert metrics.valid_ratio == 1.0
    assert metrics.adjacent_pair_count == 4
    assert metrics.gradient_p50 == pytest.approx(2.0)
    assert metrics.discontinuity_ratio == pytest.approx(0.75)


def test_compares_coarse_heights_with_median_of_fine_cells() -> None:
    coarse_heights = np.array([[10.0, 12.0], [14.0, 16.0]])
    fine_heights = np.repeat(np.repeat(coarse_heights + 0.2, 2, axis=0), 2, axis=1)

    agreement = compare_height_resolutions(
        normalized(fine_heights, 0.5),
        normalized(coarse_heights, 1.0),
    )

    assert agreement.compared_cell_count == 4
    assert agreement.overlap_ratio == 1.0
    assert agreement.mae_m == pytest.approx(0.2)
    assert agreement.rmse_m == pytest.approx(0.2)
    assert agreement.absolute_error_p95_m == pytest.approx(0.2)


def test_height_metrics_require_an_adjacent_valid_pair() -> None:
    raster = normalized(
        np.array([[10.0, np.nan], [np.nan, np.nan]]),
        1.0,
        valid_mask=np.array([[True, False], [False, False]]),
    )

    with pytest.raises(ValueError, match="adjacent"):
        evaluate_height_evidence(raster)
