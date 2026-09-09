from dataclasses import replace

import pytest

from urbanstock3d.reconstruction.benchmark import (
    ReconstructionBenchmarkEntry,
    compare_reconstructions,
)
from urbanstock3d.reconstruction.enums import BackendName

ROOFER = ReconstructionBenchmarkEntry(
    BackendName.ROOFER,
    "2.2",
    True,
    "MEDIUM",
    True,
    True,
    24,
    14,
    0.99,
    0.8,
    1.5,
    0.8,
    (),
)


def test_ranks_only_accepted_candidates_by_metric_fit() -> None:
    city3d = replace(ROOFER, backend=BackendName.CITY3D, point_surface_rmse_m=0.5)
    qroof = replace(
        ROOFER,
        backend=BackendName.QROOF,
        accepted=False,
        point_surface_rmse_m=0.1,
        failures=("invalid topology",),
    )

    report = compare_reconstructions((ROOFER, qroof, city3d))

    assert report.accepted_ranking == (BackendName.CITY3D, BackendName.ROOFER)
    assert len(report.entries) == 3


def test_rejects_duplicate_backend_rows() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        compare_reconstructions((ROOFER, ROOFER))


def test_rejects_empty_benchmark() -> None:
    with pytest.raises(ValueError, match="at least one"):
        compare_reconstructions(())
