"""Backend interface and static capabilities."""

from dataclasses import dataclass
from typing import Protocol

from urbanstock3d.reconstruction.enums import BackendName, LodRequest
from urbanstock3d.reconstruction.models import ReconstructionEvidence, ReconstructionResult


@dataclass(frozen=True)
class BackendCapabilities:
    """Evidence and runtime requirements declared by one backend."""

    name: BackendName
    supported_lods: frozenset[LodRequest]
    requires_lidar: bool
    requires_dsm: bool
    requires_orthophoto: bool
    requires_gpu: bool
    learned_geometry: bool
    maturity: str
    output_kind: str = "CityJSON"

    def __post_init__(self) -> None:
        if self.name is BackendName.AUTO:
            raise ValueError("AUTO is a routing choice, not a backend capability")
        if not self.supported_lods or LodRequest.AUTO in self.supported_lods:
            raise ValueError("backend capabilities require explicit supported LoDs")


class ReconstructionBackend(Protocol):
    """Minimal boundary implemented by every reconstruction adapter."""

    @property
    def capabilities(self) -> BackendCapabilities: ...

    def is_available(self) -> bool: ...

    def reconstruct(
        self,
        *,
        evidence: ReconstructionEvidence,
        lod: LodRequest,
    ) -> ReconstructionResult: ...
