"""Reconstruction-backend adapter for the native Roofer client."""

from pathlib import Path
from typing import Protocol

from urbanstock3d.errors import RooferExecutionError
from urbanstock3d.providers.roofer import RooferRun
from urbanstock3d.reconstruction.backends.base import BackendCapabilities
from urbanstock3d.reconstruction.enums import BackendName, LodRequest, ReconstructionStatus
from urbanstock3d.reconstruction.models import (
    GeometryProvenance,
    ReconstructionEvidence,
    ReconstructionResult,
)


class RooferRunner(Protocol):
    """Narrow native-client interface required by the backend adapter."""

    def reconstruct(
        self,
        point_cloud: Path,
        footprint: Path,
        output_directory: Path,
        *,
        id_attribute: str = "building_id",
        jobs: int = 1,
        lod12: bool = False,
        lod13: bool = False,
        lod22: bool = True,
    ) -> RooferRun: ...


class RooferBackend:
    """Expose native Roofer through the backend-independent contract."""

    capabilities = BackendCapabilities(
        name=BackendName.ROOFER,
        supported_lods=frozenset({LodRequest.LOD12, LodRequest.LOD13, LodRequest.LOD22}),
        requires_lidar=True,
        requires_dsm=False,
        requires_orthophoto=False,
        requires_gpu=False,
        learned_geometry=False,
        maturity="production_baseline",
    )

    def __init__(self, client: RooferRunner, output_root: Path) -> None:
        self.client = client
        self.output_root = output_root

    def is_available(self) -> bool:
        """The native executable was resolved when the client was constructed."""
        return True

    def reconstruct(
        self,
        *,
        evidence: ReconstructionEvidence,
        lod: LodRequest,
    ) -> ReconstructionResult:
        """Run exactly one requested Roofer LoD and normalize its result contract."""
        if lod is LodRequest.AUTO:
            raise ValueError("Roofer backend requires an explicit target LoD")
        point_cloud = _required_path(evidence.lidar_points, "LiDAR point cloud")
        footprint = _required_path(evidence.footprint, "projected footprint")
        output_directory = self.output_root / evidence.building_id
        try:
            run = self.client.reconstruct(
                point_cloud,
                footprint,
                output_directory,
                lod12=lod is LodRequest.LOD12,
                lod13=lod is LodRequest.LOD13,
                lod22=lod is LodRequest.LOD22,
            )
        except (RooferExecutionError, OSError) as error:
            return ReconstructionResult(
                status=ReconstructionStatus.FAILED,
                requested_lod=lod,
                targeted_lod=lod,
                delivered_lod=None,
                backend=BackendName.ROOFER,
                model_path=None,
                provenance=None,
                reasons=(str(error),),
            )
        return ReconstructionResult(
            status=ReconstructionStatus.SUCCESS,
            requested_lod=lod,
            targeted_lod=lod,
            delivered_lod=lod,
            backend=BackendName.ROOFER,
            model_path=run.output_files[0],
            provenance=GeometryProvenance(lidar_observed=True),
        )


def _required_path(value: object, label: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{label} evidence must be a pathlib.Path")
    if not value.is_file():
        raise FileNotFoundError(value)
    return value
