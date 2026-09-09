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
    finalize_or_execute_fallback,
)
from urbanstock3d.reconstruction.backends import BackendCapabilities, BackendRegistry
from urbanstock3d.reconstruction.validation import QualityConfidence, ReconstructionQualityReport


class SuccessfulBackend:
    capabilities = BackendCapabilities(
        BackendName.ROOFER,
        frozenset({LodRequest.LOD13, LodRequest.LOD22}),
        True,
        False,
        False,
        False,
        False,
        "baseline",
    )

    def __init__(self) -> None:
        self.calls: list[LodRequest] = []

    def is_available(self) -> bool:
        return True

    def reconstruct(
        self, *, evidence: ReconstructionEvidence, lod: LodRequest
    ) -> ReconstructionResult:
        self.calls.append(lod)
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


def quality(*, accepted: bool) -> ReconstructionQualityReport:
    return ReconstructionQualityReport(
        True,
        True,
        0.2 if accepted else 4.0,
        0.4 if accepted else 8.0,
        0.95 if accepted else 0.4,
        accepted,
        QualityConfidence.HIGH if accepted else QualityConfidence.LOW,
        () if accepted else ("poor LiDAR fit",),
        (),
    )


def test_accepted_primary_is_returned_without_fallback() -> None:
    backend = SuccessfulBackend()
    registry = BackendRegistry()
    registry.register(backend)
    plan = ReconstructionPlan(
        LodRequest.AUTO,
        LodRequest.LOD22,
        BackendName.AUTO,
        BackendName.ROOFER,
        "balanced",
        LodRequest.LOD13,
        (),
    )
    primary = backend.reconstruct(
        evidence=ReconstructionEvidence("building-1", Path("footprint.geojson")),
        lod=LodRequest.LOD22,
    )

    final = finalize_or_execute_fallback(
        plan,
        ReconstructionEvidence("building-1", Path("footprint.geojson")),
        registry,
        primary,
        quality(accepted=True),
    )

    assert final is primary
    assert backend.calls == [LodRequest.LOD22]


def test_rejected_primary_executes_exactly_one_lower_lod_fallback() -> None:
    backend = SuccessfulBackend()
    registry = BackendRegistry()
    registry.register(backend)
    plan = ReconstructionPlan(
        LodRequest.AUTO,
        LodRequest.LOD22,
        BackendName.AUTO,
        BackendName.ROOFER,
        "balanced",
        LodRequest.LOD13,
        (),
    )
    evidence = ReconstructionEvidence("building-1", Path("footprint.geojson"))
    primary = backend.reconstruct(evidence=evidence, lod=LodRequest.LOD22)

    final = finalize_or_execute_fallback(plan, evidence, registry, primary, quality(accepted=False))

    assert final.status is ReconstructionStatus.SUCCESS
    assert final.targeted_lod is LodRequest.LOD13
    assert backend.calls == [LodRequest.LOD22, LodRequest.LOD13]


def test_rejected_primary_without_planned_fallback_abstains() -> None:
    plan = ReconstructionPlan(
        LodRequest.LOD22,
        LodRequest.LOD22,
        BackendName.ROOFER,
        BackendName.ROOFER,
        "balanced",
        None,
        (),
    )
    primary = ReconstructionResult(
        ReconstructionStatus.SUCCESS,
        LodRequest.LOD22,
        LodRequest.LOD22,
        LodRequest.LOD22,
        BackendName.ROOFER,
        Path("model.city.jsonl"),
        GeometryProvenance(lidar_observed=True),
    )

    final = finalize_or_execute_fallback(
        plan,
        ReconstructionEvidence("building-1", Path("footprint.geojson")),
        BackendRegistry(),
        primary,
        quality(accepted=False),
    )

    assert final.status is ReconstructionStatus.ABSTAINED
    assert "poor LiDAR fit" in final.reasons
