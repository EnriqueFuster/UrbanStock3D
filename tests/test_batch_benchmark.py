from dataclasses import replace
from pathlib import Path

import pytest

from urbanstock3d.reconstruction.batch_benchmark import (
    BuildingBenchmarkResult,
    benchmark_selected_buildings,
    summarize_backend_results,
)
from urbanstock3d.reconstruction.benchmark import ReconstructionBenchmarkEntry
from urbanstock3d.reconstruction.enums import BackendName

ENTRY = ReconstructionBenchmarkEntry(
    BackendName.ROOFER, "2.2", True, "MEDIUM", True, True, 20, 12, 0.95, 0.4, 0.8, 0.9, ()
)


def test_summarizes_only_evaluated_backend_results() -> None:
    buildings = (
        BuildingBenchmarkResult("a", "simple", (ENTRY,), {}),
        BuildingBenchmarkResult(
            "b", "complex", (replace(ENTRY, accepted=False, point_surface_rmse_m=0.8),), {}
        ),
    )

    roofer, city3d = summarize_backend_results(buildings)

    assert roofer.evaluated_count == 2
    assert roofer.accepted_count == 1
    assert roofer.mean_rmse_m == pytest.approx(0.6)
    assert city3d.evaluated_count == 0
    assert city3d.mean_rmse_m is None


def test_records_missing_artifacts_without_stopping_batch(tmp_path: Path) -> None:
    selection = tmp_path / "selected.json"
    selection.write_text(
        '{"buildings":[{"building_id":"missing","profile":"test"}]}', encoding="utf-8"
    )

    report = benchmark_selected_buildings(selection, tmp_path / "outputs")

    assert report.buildings[0].entries == ()
    assert report.buildings[0].unavailable == {
        "roofer": "missing footprint or LiDAR crop",
        "city3d": "missing footprint or LiDAR crop",
    }
