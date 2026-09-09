"""Small, deterministic pre-reconstruction planner."""

from urbanstock3d.reconstruction.backends.registry import BackendRegistry
from urbanstock3d.reconstruction.enums import (
    BackendName,
    LodRequest,
    ReconstructionPolicy,
)
from urbanstock3d.reconstruction.models import (
    LidarQualityReport,
    ReconstructionPlan,
    ReconstructionRequest,
)
from urbanstock3d.reconstruction.quality.lod_feasibility import LodFeasibilityReport
from urbanstock3d.reconstruction.quality.roof_complexity import (
    RoofComplexityClass,
    RoofComplexityReport,
)

LOD_ORDER = (LodRequest.LOD12, LodRequest.LOD13, LodRequest.LOD22)
AUTO_BACKEND_ORDER = (BackendName.ROOFER, BackendName.CITY3D)


def plan_reconstruction(
    request: ReconstructionRequest,
    lidar: LidarQualityReport,
    complexity: RoofComplexityReport,
    feasibility: LodFeasibilityReport,
    registry: BackendRegistry,
) -> ReconstructionPlan:
    """Choose one feasible LoD and one available compatible backend."""
    target, lod_reasons, forced = _select_lod(request, feasibility)
    if target is None:
        return ReconstructionPlan(
            request.lod,
            None,
            request.backend,
            None,
            None,
            None,
            lod_reasons,
        )
    backend = _select_backend(request.backend, target, lidar, registry)
    if backend is None:
        return ReconstructionPlan(
            request.lod,
            target,
            request.backend,
            None,
            None,
            _fallback_lod(target, feasibility, request.policy),
            (*lod_reasons, "no available backend supports the target LoD"),
            forced=forced,
        )
    profile = _roofer_profile(lidar, complexity) if backend is BackendName.ROOFER else None
    return ReconstructionPlan(
        requested_lod=request.lod,
        target_lod=target,
        requested_backend=request.backend,
        selected_backend=backend,
        selected_profile=profile,
        fallback_lod=_fallback_lod(target, feasibility, request.policy),
        reasons=(*lod_reasons, f"selected available {backend.value} backend"),
        warnings=("execution forced beyond measured feasibility",) if forced else (),
        forced=forced,
    )


def _select_lod(
    request: ReconstructionRequest,
    feasibility: LodFeasibilityReport,
) -> tuple[LodRequest | None, tuple[str, ...], bool]:
    supported = {
        LodRequest.LOD12: feasibility.lod12.supported,
        LodRequest.LOD13: feasibility.lod13.supported,
        LodRequest.LOD22: feasibility.lod22.supported,
    }
    if request.lod is LodRequest.AUTO:
        target = feasibility.highest_supported_lod
        reasons = (
            (f"automatic target is highest supported LoD {target.value}",)
            if target
            else ("no LoD is supported by available evidence",)
        )
        return target, reasons, False
    if supported[request.lod]:
        return request.lod, (f"requested LoD {request.lod.value} is supported",), False
    if request.policy is ReconstructionPolicy.FORCE_EXPERIMENTAL:
        return request.lod, (f"forced unsupported LoD {request.lod.value}",), True
    if request.policy is ReconstructionPolicy.FALLBACK:
        requested_index = LOD_ORDER.index(request.lod)
        lower = [lod for lod in LOD_ORDER[:requested_index] if supported[lod]]
        if lower:
            target = lower[-1]
            return target, (f"downgraded unsupported request to LoD {target.value}",), False
    return None, (f"requested LoD {request.lod.value} is not supported",), False


def _select_backend(
    requested: BackendName,
    lod: LodRequest,
    lidar: LidarQualityReport,
    registry: BackendRegistry,
) -> BackendName | None:
    candidates = AUTO_BACKEND_ORDER if requested is BackendName.AUTO else (requested,)
    for name in candidates:
        if not registry.is_available(name):
            continue
        capabilities = registry.get(name).capabilities
        if lod not in capabilities.supported_lods:
            continue
        if capabilities.requires_lidar and not lidar.available:
            continue
        if capabilities.requires_dsm or capabilities.requires_orthophoto:
            continue
        return name
    return None


def _fallback_lod(
    target: LodRequest,
    feasibility: LodFeasibilityReport,
    policy: ReconstructionPolicy,
) -> LodRequest | None:
    if policy is not ReconstructionPolicy.FALLBACK:
        return None
    supported = {
        LodRequest.LOD12: feasibility.lod12.supported,
        LodRequest.LOD13: feasibility.lod13.supported,
        LodRequest.LOD22: feasibility.lod22.supported,
    }
    lower = [lod for lod in LOD_ORDER[: LOD_ORDER.index(target)] if supported[lod]]
    return lower[-1] if lower else None


def _roofer_profile(
    lidar: LidarQualityReport,
    complexity: RoofComplexityReport,
) -> str:
    if lidar.quality_class in {"EXCELLENT", "GOOD"} and complexity.complexity_class in {
        RoofComplexityClass.SIMPLE,
        RoofComplexityClass.MODERATE,
    }:
        return "balanced"
    return "conservative"
