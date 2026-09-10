"""Comparable reconstruction metrics shared by every backend."""

from dataclasses import asdict, dataclass

from urbanstock3d.reconstruction.enums import BackendName
from urbanstock3d.reconstruction.validation import (
    CityJsonQualityReport,
    LidarFitReport,
    ObjQualityReport,
    ReconstructionQualityReport,
)


@dataclass(frozen=True)
class ReconstructionBenchmarkEntry:
    """One backend result expressed only with backend-independent metrics."""

    backend: BackendName
    lod: str
    accepted: bool
    confidence_class: str
    geometry_valid: bool
    watertight: bool
    vertex_count: int
    face_count: int
    footprint_bbox_iou: float
    point_surface_rmse_m: float
    point_surface_p95_m: float
    within_050m_ratio: float
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["backend"] = self.backend.value
        return payload


@dataclass(frozen=True)
class ReconstructionBenchmarkReport:
    """A deterministic comparison that ranks only accepted candidates."""

    entries: tuple[ReconstructionBenchmarkEntry, ...]
    accepted_ranking: tuple[BackendName, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "accepted_ranking": [backend.value for backend in self.accepted_ranking],
        }


def build_benchmark_entry(
    backend: BackendName,
    geometry: CityJsonQualityReport | ObjQualityReport,
    lidar_fit: LidarFitReport,
    quality: ReconstructionQualityReport,
) -> ReconstructionBenchmarkEntry:
    """Flatten validated model evidence into the shared benchmark schema."""
    if backend is BackendName.AUTO:
        raise ValueError("Benchmark entries require a concrete backend")
    return ReconstructionBenchmarkEntry(
        backend=backend,
        lod=geometry.lod,
        accepted=quality.accepted,
        confidence_class=quality.confidence_class.value,
        geometry_valid=geometry.geometry_valid,
        watertight=geometry.watertight,
        vertex_count=geometry.vertex_count,
        face_count=geometry.face_count,
        footprint_bbox_iou=geometry.footprint_bbox_iou,
        point_surface_rmse_m=lidar_fit.distance_rmse_m,
        point_surface_p95_m=lidar_fit.distance_p95_m,
        within_050m_ratio=lidar_fit.within_050m_ratio,
        failures=quality.failures,
    )


def compare_reconstructions(
    entries: tuple[ReconstructionBenchmarkEntry, ...],
) -> ReconstructionBenchmarkReport:
    """Order accepted candidates by RMSE and P95 while retaining all failures."""
    if not entries:
        raise ValueError("A benchmark requires at least one reconstruction entry")
    backends = [entry.backend for entry in entries]
    if len(set(backends)) != len(backends):
        raise ValueError("A benchmark cannot contain duplicate backend entries")
    accepted = sorted(
        (entry for entry in entries if entry.accepted),
        key=lambda entry: (entry.point_surface_rmse_m, entry.point_surface_p95_m),
    )
    return ReconstructionBenchmarkReport(
        entries=tuple(sorted(entries, key=lambda entry: entry.backend.value)),
        accepted_ranking=tuple(entry.backend for entry in accepted),
    )
