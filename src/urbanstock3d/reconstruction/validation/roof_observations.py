"""Select comparable roof observations from classified building returns."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class RoofObservationParameters:
    """Parameters for a conservative local upper-envelope filter."""

    radius_m: float = 1.5
    local_percentile: float = 80.0
    vertical_tolerance_m: float = 0.75
    minimum_neighbors: int = 4

    def __post_init__(self) -> None:
        if self.radius_m <= 0:
            raise ValueError("Roof-observation radius must be positive")
        if not 0 <= self.local_percentile <= 100:
            raise ValueError("Local percentile must be between 0 and 100")
        if self.vertical_tolerance_m < 0:
            raise ValueError("Vertical tolerance cannot be negative")
        if self.minimum_neighbors <= 0:
            raise ValueError("Minimum neighbor count must be positive")


@dataclass(frozen=True)
class RoofObservationReport:
    """Audit trail for points retained as visible roof observations."""

    method: str
    source_point_count: int
    selected_point_count: int
    rejected_point_count: int
    selected_ratio: float
    parameters: RoofObservationParameters

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def select_roof_observations(
    points: np.ndarray[Any, np.dtype[np.floating[Any]]],
    *,
    parameters: RoofObservationParameters | None = None,
) -> tuple[np.ndarray[Any, np.dtype[np.bool_]], RoofObservationReport]:
    """Retain points close to each XY neighborhood's upper height envelope.

    Sparse neighborhoods are retained because there is insufficient local evidence
    to classify their points as occluded or facade-like returns.
    """
    parameters = parameters or RoofObservationParameters()
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Roof observations must be an Nx3 array")
    if not len(points):
        raise ValueError("Roof observations cannot be empty")
    if not np.all(np.isfinite(points)):
        raise ValueError("Roof observations must contain only finite coordinates")

    neighborhoods = cKDTree(points[:, :2]).query_ball_point(points[:, :2], parameters.radius_m)
    selected = np.ones(len(points), dtype=np.bool_)
    for index, neighbors in enumerate(neighborhoods):
        if len(neighbors) < parameters.minimum_neighbors:
            continue
        local_ceiling = float(np.percentile(points[neighbors, 2], parameters.local_percentile))
        selected[index] = points[index, 2] >= local_ceiling - parameters.vertical_tolerance_m

    selected_count = int(np.count_nonzero(selected))
    report = RoofObservationReport(
        method="local_upper_envelope",
        source_point_count=len(points),
        selected_point_count=selected_count,
        rejected_point_count=len(points) - selected_count,
        selected_ratio=selected_count / len(points),
        parameters=parameters,
    )
    return selected, report
