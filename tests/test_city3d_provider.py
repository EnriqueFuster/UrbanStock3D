import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from urbanstock3d.errors import City3DExecutionError
from urbanstock3d.providers.city3d import City3DClient


def test_city3d_client_builds_wrapper_command(tmp_path: Path) -> None:
    executable = tmp_path / "urbanstock-city3d.exe"
    executable.touch()
    point_cloud = tmp_path / "crop.laz"
    point_cloud.touch()
    footprint = tmp_path / "footprint.geojson"
    footprint.touch()
    output = tmp_path / "result" / "building.obj"

    def completed(command: tuple[str, ...], **_: object) -> subprocess.CompletedProcess[str]:
        output.write_text("v 0 0 0\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "complete\n", "")

    runtime_directory = tmp_path / "runtime"
    runtime_directory.mkdir()
    with patch("urbanstock3d.providers.city3d.subprocess.run", side_effect=completed) as run:
        result = City3DClient(str(executable), runtime_directory=runtime_directory).reconstruct(
            point_cloud, footprint, output
        )

    command = run.call_args.args[0]
    assert command == (
        str(executable.resolve()),
        str(point_cloud.resolve()),
        str(footprint.resolve()),
        str(output.resolve()),
    )
    assert result.output_file == output
    assert run.call_args.kwargs["env"]["PATH"].startswith(str(runtime_directory.resolve()))


def test_city3d_client_reports_process_failure(tmp_path: Path) -> None:
    executable = tmp_path / "urbanstock-city3d.exe"
    executable.touch()
    point_cloud = tmp_path / "crop.laz"
    point_cloud.touch()
    footprint = tmp_path / "footprint.geojson"
    footprint.touch()
    process = Mock(returncode=2, stdout="", stderr="solver failed")

    with (
        patch("urbanstock3d.providers.city3d.subprocess.run", return_value=process),
        pytest.raises(City3DExecutionError, match="solver failed"),
    ):
        City3DClient(str(executable)).reconstruct(point_cloud, footprint, tmp_path / "result.obj")


def test_city3d_client_requires_output_artifact(tmp_path: Path) -> None:
    executable = tmp_path / "urbanstock-city3d.exe"
    executable.touch()
    point_cloud = tmp_path / "crop.laz"
    point_cloud.touch()
    footprint = tmp_path / "footprint.geojson"
    footprint.touch()
    process = Mock(returncode=0, stdout="complete", stderr="")

    with (
        patch("urbanstock3d.providers.city3d.subprocess.run", return_value=process),
        pytest.raises(City3DExecutionError, match="without producing"),
    ):
        City3DClient(str(executable)).reconstruct(point_cloud, footprint, tmp_path / "missing.obj")
