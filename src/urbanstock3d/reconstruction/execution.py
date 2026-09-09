"""Execute one reconstruction plan through its selected backend."""

from dataclasses import replace

from urbanstock3d.reconstruction.backends.registry import BackendRegistry
from urbanstock3d.reconstruction.enums import ReconstructionStatus
from urbanstock3d.reconstruction.models import (
    ReconstructionEvidence,
    ReconstructionPlan,
    ReconstructionResult,
)


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
