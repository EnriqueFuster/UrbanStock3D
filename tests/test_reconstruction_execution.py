from pathlib import Path

from urbanstock3d.reconstruction import (
    BackendName,
    GeometryProvenance,
    LodRequest,
    ReconstructionEvidence,
    ReconstructionPlan,
    ReconstructionResult,
    ReconstructionStatus,
    execute_reconstruction_plan,
)
from urbanstock3d.reconstruction.backends import BackendCapabilities, BackendRegistry


class SuccessfulBackend:
    capabilities = BackendCapabilities(
        BackendName.ROOFER,
        frozenset({LodRequest.LOD22}),
        True,
        False,
        False,
        False,
        False,
        "baseline",
    )

    def is_available(self) -> bool:
        return True

    def reconstruct(
        self, *, evidence: ReconstructionEvidence, lod: LodRequest
    ) -> ReconstructionResult:
        return ReconstructionResult(
            ReconstructionStatus.SUCCESS,
            lod,
            lod,
            lod,
            BackendName.ROOFER,
            Path("model.city.jsonl"),
            GeometryProvenance(lidar_observed=True),
        )


def test_executes_only_the_planned_backend_and_preserves_original_request() -> None:
    registry = BackendRegistry()
    registry.register(SuccessfulBackend())
    plan = ReconstructionPlan(
        LodRequest.AUTO,
        LodRequest.LOD22,
        BackendName.AUTO,
        BackendName.ROOFER,
        "balanced",
        LodRequest.LOD13,
        ("selected Roofer",),
    )

    result = execute_reconstruction_plan(
        plan,
        ReconstructionEvidence("building-1", Path("footprint.geojson")),
        registry,
    )

    assert result.status is ReconstructionStatus.SUCCESS
    assert result.requested_lod is LodRequest.AUTO
    assert result.targeted_lod is LodRequest.LOD22


def test_non_executable_plan_abstains_without_calling_a_backend() -> None:
    plan = ReconstructionPlan(
        LodRequest.LOD22,
        None,
        BackendName.AUTO,
        None,
        None,
        None,
        ("no supported LoD",),
    )

    result = execute_reconstruction_plan(
        plan,
        ReconstructionEvidence("building-1", Path("footprint.geojson")),
        BackendRegistry(),
    )

    assert result.status is ReconstructionStatus.ABSTAINED
    assert result.reasons == ("no supported LoD",)
