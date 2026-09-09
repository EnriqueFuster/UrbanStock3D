"""Data contracts shared by planning and reconstruction backends."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from urbanstock3d.reconstruction.enums import (
    BackendName,
    LodRequest,
    ReconstructionPolicy,
    ReconstructionPriority,
    ReconstructionStatus,
)


@dataclass(frozen=True)
class ReconstructionRequest:
    """Describe user intent without making assumptions about available evidence."""

    lod: LodRequest = LodRequest.AUTO
    backend: BackendName = BackendName.AUTO
    policy: ReconstructionPolicy = ReconstructionPolicy.FALLBACK
    priority: ReconstructionPriority = ReconstructionPriority.BALANCED
    use_dsm: bool = True
    use_orthophoto: bool = True
    use_learned_completion: bool = True
    retain_debug_artifacts: bool = False

    def __post_init__(self) -> None:
        if (
            self.backend is BackendName.AUTO
            and self.policy is ReconstructionPolicy.FORCE_EXPERIMENTAL
        ):
            raise ValueError("force_experimental requires an explicit backend")
        if (
            self.priority
            in {
                ReconstructionPriority.BENCHMARK,
                ReconstructionPriority.RESEARCH,
            }
            and self.policy is ReconstructionPolicy.STRICT
        ):
            raise ValueError("strict policy is incompatible with benchmark or research priority")


@dataclass(frozen=True)
class ReconstructionEvidence:
    """References to aligned evidence acquired outside the reconstruction package."""

    building_id: str
    footprint: Any
    building_parts: Any | None = None
    lidar_points: Any | None = None
    dsm: Any | None = None
    dtm: Any | None = None
    ndsm: Any | None = None
    orthophoto: Any | None = None
    source_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeometryProvenance:
    """Declare which evidence contributed to delivered geometry."""

    lidar_observed: bool
    dsm_support_used: bool = False
    image_support_used: bool = False
    learned_completion_used: bool = False
    learned_topology_used: bool = False


@dataclass(frozen=True)
class ReconstructionResult:
    """Backend-independent reconstruction outcome."""

    status: ReconstructionStatus
    requested_lod: LodRequest
    targeted_lod: LodRequest | None
    delivered_lod: LodRequest | None
    backend: BackendName | None
    model_path: Path | None
    provenance: GeometryProvenance | None
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status is ReconstructionStatus.SUCCESS:
            if self.delivered_lod is None or self.backend is None or self.model_path is None:
                raise ValueError(
                    "successful results require delivered LoD, backend, and model path"
                )
            if self.provenance is None:
                raise ValueError("successful results require geometry provenance")
