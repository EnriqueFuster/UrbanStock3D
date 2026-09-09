"""Execute one reconstruction plan through its selected backend."""

from dataclasses import replace

from urbanstock3d.reconstruction.backends.registry import BackendRegistry
from urbanstock3d.reconstruction.enums import ReconstructionStatus
from urbanstock3d.reconstruction.models import (
    ReconstructionEvidence,
    ReconstructionPlan,
    ReconstructionResult,
)
from urbanstock3d.reconstruction.validation.quality import ReconstructionQualityReport


def execute_reconstruction_plan(
    plan: ReconstructionPlan,
    evidence: ReconstructionEvidence,
    registry: BackendRegistry,
) -> ReconstructionResult:
    """Execute exactly the primary backend selected by the planner."""
    if not plan.executable:
        return ReconstructionResult(
            status=ReconstructionStatus.ABSTAINED,
            requested_lod=plan.requested_lod,
            targeted_lod=plan.target_lod,
            delivered_lod=None,
            backend=plan.selected_backend,
            model_path=None,
            provenance=None,
            reasons=plan.reasons,
            warnings=plan.warnings,
        )
    assert plan.selected_backend is not None
    assert plan.target_lod is not None
    backend = registry.get(plan.selected_backend)
    result = backend.reconstruct(evidence=evidence, lod=plan.target_lod)
    return replace(
        result,
        requested_lod=plan.requested_lod,
        targeted_lod=plan.target_lod,
        warnings=(*plan.warnings, *result.warnings),
    )


def finalize_or_execute_fallback(
    plan: ReconstructionPlan,
    evidence: ReconstructionEvidence,
    registry: BackendRegistry,
    primary_result: ReconstructionResult,
    primary_quality: ReconstructionQualityReport,
) -> ReconstructionResult:
    """Return an accepted primary result or execute its single planned LoD fallback."""
    if primary_quality.accepted:
        return primary_result
    if plan.fallback_lod is None or plan.selected_backend is None:
        return ReconstructionResult(
            status=ReconstructionStatus.ABSTAINED,
            requested_lod=plan.requested_lod,
            targeted_lod=None,
            delivered_lod=None,
            backend=None,
            model_path=None,
            provenance=None,
            reasons=("primary reconstruction failed quality control", *primary_quality.failures),
            warnings=primary_quality.warnings,
        )
    backend = registry.get(plan.selected_backend)
    fallback = backend.reconstruct(evidence=evidence, lod=plan.fallback_lod)
    return replace(
        fallback,
        requested_lod=plan.requested_lod,
        targeted_lod=plan.fallback_lod,
        warnings=("primary reconstruction rejected; executed one LoD fallback", *fallback.warnings),
    )
