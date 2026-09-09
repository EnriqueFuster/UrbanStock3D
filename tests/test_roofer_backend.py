from pathlib import Path

from urbanstock3d.errors import RooferExecutionError
from urbanstock3d.providers.roofer import RooferRun
from urbanstock3d.reconstruction import (
    BackendName,
    LodRequest,
    ReconstructionEvidence,
    ReconstructionStatus,
)
from urbanstock3d.reconstruction.backends import RooferBackend


class SuccessfulRunner:
    def __init__(self) -> None:
        self.flags: dict[str, bool] = {}

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
    ) -> RooferRun:
        self.flags = {"lod12": lod12, "lod13": lod13, "lod22": lod22}
        model = output_directory / "building.city.jsonl"
        model.parent.mkdir(parents=True)
        model.write_text("{}\n", encoding="utf-8")
        return RooferRun("1.0", (), (model,), "", "")


class FailingRunner(SuccessfulRunner):
    def reconstruct(self, *args: object, **kwargs: object) -> RooferRun:
        raise RooferExecutionError("native process failed")


def evidence(tmp_path: Path) -> ReconstructionEvidence:
    lidar = tmp_path / "crop.laz"
    footprint = tmp_path / "footprint.geojson"
    lidar.touch()
    footprint.write_text("{}", encoding="utf-8")
    return ReconstructionEvidence("building-1", footprint, lidar_points=lidar)


def test_roofer_backend_translates_explicit_lod_and_provenance(tmp_path: Path) -> None:
    runner = SuccessfulRunner()
    backend = RooferBackend(runner, tmp_path / "outputs")

    result = backend.reconstruct(evidence=evidence(tmp_path), lod=LodRequest.LOD13)

    assert result.status is ReconstructionStatus.SUCCESS
    assert result.backend is BackendName.ROOFER
    assert result.delivered_lod is LodRequest.LOD13
    assert result.provenance is not None and result.provenance.lidar_observed
    assert runner.flags == {"lod12": False, "lod13": True, "lod22": False}


def test_roofer_backend_normalizes_native_failure(tmp_path: Path) -> None:
    backend = RooferBackend(FailingRunner(), tmp_path / "outputs")

    result = backend.reconstruct(evidence=evidence(tmp_path), lod=LodRequest.LOD22)

    assert result.status is ReconstructionStatus.FAILED
    assert result.model_path is None
    assert result.reasons == ("native process failed",)


def test_roofer_backend_rejects_auto_lod(tmp_path: Path) -> None:
    backend = RooferBackend(SuccessfulRunner(), tmp_path / "outputs")

    try:
        backend.reconstruct(evidence=evidence(tmp_path), lod=LodRequest.AUTO)
    except ValueError as error:
        assert "explicit" in str(error)
    else:
        raise AssertionError("AUTO LoD should have been rejected")
