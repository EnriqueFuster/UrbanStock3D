"""Evaluate existing reconstruction artifacts across selected buildings."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from urbanstock3d.providers.pnoa_lidar import footprint_polygons_utm
from urbanstock3d.reconstruction.benchmark import (
    ReconstructionBenchmarkEntry,
    build_benchmark_entry,
)
from urbanstock3d.reconstruction.enums import BackendName
from urbanstock3d.reconstruction.validation import (
    CityJsonQualityReport,
    ObjQualityReport,
    assess_cityjson_lidar_fit,
    assess_obj_lidar_fit,
    evaluate_reconstruction_quality,
    validate_cityjsonseq,
    validate_obj,
)


@dataclass(frozen=True)
class BuildingBenchmarkResult:
    """Available backend results and explicit omissions for one building."""

    building_id: str
    profile: str
    entries: tuple[ReconstructionBenchmarkEntry, ...]
    unavailable: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["entries"] = [entry.to_dict() for entry in self.entries]
        return payload


@dataclass(frozen=True)
class BackendBenchmarkSummary:
    """Small aggregate used to inspect benchmark coverage, not rank models."""

    backend: BackendName
    evaluated_count: int
    accepted_count: int
    mean_rmse_m: float | None
    mean_p95_m: float | None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["backend"] = self.backend.value
        return payload


@dataclass(frozen=True)
class SelectedBuildingsBenchmarkReport:
    """Cross-building report that preserves missing and failed evaluations."""

    buildings: tuple[BuildingBenchmarkResult, ...]
    summaries: tuple[BackendBenchmarkSummary, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "selected_buildings_reconstruction_benchmark",
            "buildings": [building.to_dict() for building in self.buildings],
            "summaries": [summary.to_dict() for summary in self.summaries],
        }


def benchmark_selected_buildings(
    selection_path: Path, outputs_root: Path
) -> SelectedBuildingsBenchmarkReport:
    """Evaluate every model already present without launching reconstruction."""
    selection: dict[str, Any] = json.loads(selection_path.read_text(encoding="utf-8"))
    results = tuple(
        _benchmark_building(building, outputs_root / building["building_id"])
        for building in selection["buildings"]
    )
    return SelectedBuildingsBenchmarkReport(results, summarize_backend_results(results))


def summarize_backend_results(
    buildings: tuple[BuildingBenchmarkResult, ...],
) -> tuple[BackendBenchmarkSummary, ...]:
    """Aggregate evaluated candidates while retaining sample size explicitly."""
    summaries: list[BackendBenchmarkSummary] = []
    for backend in (BackendName.ROOFER, BackendName.CITY3D):
        entries = [
            entry
            for building in buildings
            for entry in building.entries
            if entry.backend is backend
        ]
        summaries.append(
            BackendBenchmarkSummary(
                backend=backend,
                evaluated_count=len(entries),
                accepted_count=sum(entry.accepted for entry in entries),
                mean_rmse_m=(
                    sum(entry.point_surface_rmse_m for entry in entries) / len(entries)
                    if entries
                    else None
                ),
                mean_p95_m=(
                    sum(entry.point_surface_p95_m for entry in entries) / len(entries)
                    if entries
                    else None
                ),
            )
        )
    return tuple(summaries)


def _benchmark_building(building: dict[str, Any], root: Path) -> BuildingBenchmarkResult:
    entries: list[ReconstructionBenchmarkEntry] = []
    unavailable: dict[str, str] = {}
    footprint_path = root / "building.geojson"
    lidar_path = root / "lidar_context_crop.laz"
    if not footprint_path.exists() or not lidar_path.exists():
        reason = "missing footprint or LiDAR crop"
        unavailable = {
            backend.value: reason for backend in (BackendName.ROOFER, BackendName.CITY3D)
        }
        return BuildingBenchmarkResult(
            building["building_id"], building["profile"], (), unavailable
        )

    footprint_data = json.loads(footprint_path.read_text(encoding="utf-8"))
    footprint = footprint_polygons_utm(footprint_data["geometry"])
    roofer_models = sorted((root / "roofer").glob("**/*.city.jsonl"))
    city3d_models = [
        path
        for name in ("building.obj", "building_python.obj")
        if (path := root / "city3d" / name).exists()
    ]
    candidates = (
        (BackendName.ROOFER, roofer_models[0] if roofer_models else None),
        (BackendName.CITY3D, city3d_models[0] if city3d_models else None),
    )
    for backend, model in candidates:
        if model is None:
            unavailable[backend.value] = "model artifact not found"
            continue
        try:
            geometry: CityJsonQualityReport | ObjQualityReport
            if backend is BackendName.ROOFER:
                geometry = validate_cityjsonseq(model, footprint)
                fit = assess_cityjson_lidar_fit(model, lidar_path, footprint)
            else:
                geometry = validate_obj(model, footprint)
                fit = assess_obj_lidar_fit(model, lidar_path, footprint)
            quality = evaluate_reconstruction_quality(geometry, fit)
            entries.append(build_benchmark_entry(backend, geometry, fit, quality))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            unavailable[backend.value] = f"evaluation failed: {error}"
    return BuildingBenchmarkResult(
        building["building_id"], building["profile"], tuple(entries), unavailable
    )
