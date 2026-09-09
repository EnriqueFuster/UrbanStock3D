from pathlib import Path

import pytest

from urbanstock3d.reconstruction import (
    BackendName,
    GeometryProvenance,
    LodRequest,
    ReconstructionPolicy,
    ReconstructionPriority,
    ReconstructionRequest,
    ReconstructionResult,
    ReconstructionStatus,
)
from urbanstock3d.reconstruction.backends import BackendCapabilities, BackendRegistry


def test_request_defaults_to_automatic_balanced_fallback() -> None:
    request = ReconstructionRequest()

    assert request.lod is LodRequest.AUTO
    assert request.backend is BackendName.AUTO
    assert request.policy is ReconstructionPolicy.FALLBACK
    assert request.priority is ReconstructionPriority.BALANCED


def test_force_experimental_requires_explicit_backend() -> None:
    with pytest.raises(ValueError, match="explicit backend"):
        ReconstructionRequest(policy=ReconstructionPolicy.FORCE_EXPERIMENTAL)


def test_successful_result_requires_geometry_provenance() -> None:
    with pytest.raises(ValueError, match="provenance"):
        ReconstructionResult(
            status=ReconstructionStatus.SUCCESS,
            requested_lod=LodRequest.LOD22,
            targeted_lod=LodRequest.LOD22,
            delivered_lod=LodRequest.LOD22,
            backend=BackendName.ROOFER,
            model_path=Path("building.city.jsonl"),
            provenance=None,
        )


class AvailableBackend:
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

    def is_available(self) -> bool:
        return True

    def reconstruct(self, *, evidence: object, lod: LodRequest) -> ReconstructionResult:
        return ReconstructionResult(
            status=ReconstructionStatus.SUCCESS,
            requested_lod=lod,
            targeted_lod=lod,
            delivered_lod=lod,
            backend=BackendName.ROOFER,
            model_path=Path("building.city.jsonl"),
            provenance=GeometryProvenance(lidar_observed=True),
        )


def test_registry_rejects_duplicates_and_does_not_route_auto() -> None:
    registry = BackendRegistry()
    backend = AvailableBackend()
    registry.register(backend)

    assert registry.get(BackendName.ROOFER) is backend
    assert registry.is_available(BackendName.ROOFER)
    assert not registry.is_available(BackendName.AUTO)
    with pytest.raises(ValueError, match="AUTO"):
        registry.get(BackendName.AUTO)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(backend)
