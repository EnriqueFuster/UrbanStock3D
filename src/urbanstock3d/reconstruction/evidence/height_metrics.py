"""Quantify whether normalized height evidence can support roof reconstruction."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from urbanstock3d.reconstruction.evidence.terrain import NormalizedHeightRaster


@dataclass(frozen=True)
class HeightEvidenceMetrics:
    """Coverage and local vertical-change statistics for one nDSM grid."""

    resolution_m: float
    footprint_cell_count: int
    valid_cell_count: int
    valid_ratio: float
    adjacent_pair_count: int
    gradient_p50: float
    gradient_p95: float
    discontinuity_threshold_m: float
    discontinuity_ratio: float

    def to_dict(self) -> dict[str, int | float]:
        """Return JSON-ready metric values."""
        return asdict(self)


@dataclass(frozen=True)
class HeightResolutionAgreement:
    """Agreement after aggregating a finer nDSM onto a coarser aligned grid."""

    fine_resolution_m: float
    coarse_resolution_m: float
    compared_cell_count: int
    overlap_ratio: float
    mae_m: float
    rmse_m: float
    absolute_error_p95_m: float

    def to_dict(self) -> dict[str, int | float]:
        """Return JSON-ready agreement values."""
        return asdict(self)


def evaluate_height_evidence(
    raster: NormalizedHeightRaster,
    *,
    discontinuity_threshold_m: float = 1.0,
) -> HeightEvidenceMetrics:
    """Measure coverage and absolute height changes between valid adjacent cells."""
    if discontinuity_threshold_m <= 0:
        raise ValueError("Discontinuity threshold must be positive")
    horizontal = _valid_differences(
        raster.height_m[:, 1:],
        raster.height_m[:, :-1],
        raster.valid_mask[:, 1:] & raster.valid_mask[:, :-1],
    )
    vertical = _valid_differences(
        raster.height_m[1:, :],
        raster.height_m[:-1, :],
        raster.valid_mask[1:, :] & raster.valid_mask[:-1, :],
    )
    differences = np.concatenate((horizontal, vertical))
    if differences.size == 0:
        raise ValueError("Height raster requires at least one valid adjacent cell pair")
    gradients = differences / raster.resolution_m
    return HeightEvidenceMetrics(
        resolution_m=raster.resolution_m,
        footprint_cell_count=int(np.count_nonzero(raster.footprint_mask)),
        valid_cell_count=int(np.count_nonzero(raster.valid_mask)),
        valid_ratio=raster.valid_ratio,
        adjacent_pair_count=int(differences.size),
        gradient_p50=_percentile_or_nan(gradients, 50),
        gradient_p95=_percentile_or_nan(gradients, 95),
        discontinuity_threshold_m=discontinuity_threshold_m,
        discontinuity_ratio=float(np.mean(differences >= discontinuity_threshold_m)),
    )


def compare_height_resolutions(
    fine: NormalizedHeightRaster,
    coarse: NormalizedHeightRaster,
) -> HeightResolutionAgreement:
    """Compare a coarse grid with median fine-cell heights at its cell centres."""
    _validate_comparable_grids(fine, coarse)
    rows, cols = coarse.height_m.shape
    fine_row_indices, fine_col_indices = np.nonzero(fine.valid_mask)
    fine_x = fine.origin_x + (fine_col_indices + 0.5) * fine.resolution_m
    fine_y = fine.origin_y + (fine_row_indices + 0.5) * fine.resolution_m
    coarse_cols = np.floor((fine_x - coarse.origin_x) / coarse.resolution_m).astype(int)
    coarse_rows = np.floor((fine_y - coarse.origin_y) / coarse.resolution_m).astype(int)
    inside = (coarse_cols >= 0) & (coarse_cols < cols) & (coarse_rows >= 0) & (coarse_rows < rows)
    coarse_cols = coarse_cols[inside]
    coarse_rows = coarse_rows[inside]
    fine_values = fine.height_m[fine_row_indices[inside], fine_col_indices[inside]]
    cell_ids = coarse_rows * cols + coarse_cols
    order = np.argsort(cell_ids)
    cell_ids = cell_ids[order]
    fine_values = fine_values[order]
    sampled = np.full(coarse.height_m.shape, np.nan, dtype=np.float64)
    sampled_valid = np.zeros(coarse.height_m.shape, dtype=np.bool_)
    unique_cells, starts, counts = np.unique(cell_ids, return_index=True, return_counts=True)
    for cell_id, start, count in zip(unique_cells, starts, counts, strict=True):
        row, col = divmod(int(cell_id), cols)
        sampled[row, col] = float(np.median(fine_values[start : start + count]))
        sampled_valid[row, col] = True
    joint = coarse.valid_mask & sampled_valid
    errors = sampled[joint] - coarse.height_m[joint]
    coarse_valid = int(np.count_nonzero(coarse.valid_mask))
    if errors.size == 0:
        raise ValueError("Height grids have no jointly valid cells")
    absolute = np.abs(errors)
    return HeightResolutionAgreement(
        fine_resolution_m=fine.resolution_m,
        coarse_resolution_m=coarse.resolution_m,
        compared_cell_count=int(errors.size),
        overlap_ratio=float(errors.size / coarse_valid) if coarse_valid else 0.0,
        mae_m=float(np.mean(absolute)),
        rmse_m=float(np.sqrt(np.mean(np.square(errors)))),
        absolute_error_p95_m=float(np.percentile(absolute, 95)),
    )


def _valid_differences(
    first: np.ndarray[Any, np.dtype[np.floating[Any]]],
    second: np.ndarray[Any, np.dtype[np.floating[Any]]],
    valid: np.ndarray[Any, np.dtype[np.bool_]],
) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
    return np.asarray(np.abs(first[valid] - second[valid]), dtype=np.float64)


def _percentile_or_nan(values: np.ndarray[Any, np.dtype[np.floating[Any]]], q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


def _validate_comparable_grids(
    fine: NormalizedHeightRaster,
    coarse: NormalizedHeightRaster,
) -> None:
    if fine.crs != coarse.crs:
        raise ValueError("Height grids must use the same CRS")
    if fine.resolution_m >= coarse.resolution_m:
        raise ValueError("The fine grid resolution must be smaller than the coarse resolution")
