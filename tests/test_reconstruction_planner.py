from dataclasses import replace

from urbanstock3d.reconstruction import (
    BackendName,
    LidarQualityReport,
    LodRequest,
    ReconstructionEvidence,
    ReconstructionPolicy,
    ReconstructionRequest,
    ReconstructionResult,
)
from urbanstock3d.reconstruction.backends import BackendCapabilities, BackendRegistry
from urbanstock3d.reconstruction.planner import plan_reconstruction
from urbanstock3d.reconstruction.quality.lod_feasibility import (
    LodFeasibility,
    LodFeasibilityReport,
)
from urbanstock3d.reconstruction.quality.roof_complexity import (
    RoofComplexityClass,
    RoofComplexityReport,
)


class Backend:
    def __init__(self, name: BackendName, *, available: bool = True) -> None:
        self.capabilities = BackendCapabilities(
            name,
            frozenset({LodRequest.LOD12, LodRequest.LOD13, LodRequest.LOD22}),
            True,
            False,
            False,
            False,
            False,
            "baseline",
        )
        self.available = available

    def is_available(self) -> bool:
        return self.available

    def reconstruct(
        self, *, evidence: ReconstructionEvidence, lod: LodRequest
    ) -> ReconstructionResult:
        raise NotImplementedError


LIDAR = LidarQualityReport(
    True,
    100,
    80,
    5.0,
    4.0,
    0.9,
    0.9,
    0.05,
    0.3,
    0.5,
    0.8,
    0.05,
    0.01,
    0.95,
    0.0,
    0.0,
    "GOOD",
    0.9,
)
COMPLEXITY = RoofComplexityReport(
    100.0, 4, 0.78, 1, 2.0, 2, 2, 0.8, 0.03, RoofComplexityClass.MODERATE, 0.3, ()
)
SUPPORTED = LodFeasibilityReport(
    LodFeasibility(LodRequest.LOD12, True, 1.0, ()),
    LodFeasibility(LodRequest.LOD13, True, 0.9, ()),
    LodFeasibility(LodRequest.LOD22, True, 0.8, ()),
)


def registry_with_roofer(*, available: bool = True) -> BackendRegistry:
    registry = BackendRegistry()
    registry.register(Backend(BackendName.ROOFER, available=available))
    return registry


def test_auto_selects_highest_lod_and_one_available_backend() -> None:
    plan = plan_reconstruction(
        ReconstructionRequest(), LIDAR, COMPLEXITY, SUPPORTED, registry_with_roofer()
    )

    assert plan.executable
    assert plan.target_lod is LodRequest.LOD22
    assert plan.selected_backend is BackendName.ROOFER
    assert plan.selected_profile == "balanced"
    assert plan.fallback_lod is LodRequest.LOD13


def test_fallback_policy_downgrades_an_unsupported_request() -> None:
    feasibility = replace(
        SUPPORTED,
        lod13=replace(SUPPORTED.lod13, supported=False),
        lod22=replace(SUPPORTED.lod22, supported=False),
    )
    request = ReconstructionRequest(lod=LodRequest.LOD22)

    plan = plan_reconstruction(request, LIDAR, COMPLEXITY, feasibility, registry_with_roofer())

    assert plan.target_lod is LodRequest.LOD12
    assert plan.selected_backend is BackendName.ROOFER


def test_strict_policy_abstains_from_unsupported_lod() -> None:
    feasibility = replace(SUPPORTED, lod22=replace(SUPPORTED.lod22, supported=False))
    request = ReconstructionRequest(lod=LodRequest.LOD22, policy=ReconstructionPolicy.STRICT)

    plan = plan_reconstruction(request, LIDAR, COMPLEXITY, feasibility, registry_with_roofer())

    assert not plan.executable
    assert plan.target_lod is None
    assert plan.selected_backend is None


def test_unavailable_explicit_backend_produces_no_executable_plan() -> None:
    request = ReconstructionRequest(lod=LodRequest.LOD22, backend=BackendName.ROOFER)

    plan = plan_reconstruction(
        request, LIDAR, COMPLEXITY, SUPPORTED, registry_with_roofer(available=False)
    )

    assert not plan.executable
    assert plan.target_lod is LodRequest.LOD22
    assert plan.selected_backend is None
