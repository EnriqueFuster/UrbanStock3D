import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from urbanstock3d.errors import RooferExecutionError
from urbanstock3d.providers.roofer import RooferClient


def test_roofer_client_builds_native_command(tmp_path: Path) -> None:
    executable = tmp_path / "roofer.exe"
    executable.touch()
    point_cloud = tmp_path / "crop.laz"
    point_cloud.touch()
    footprint = tmp_path / "footprint.geojson"
    footprint.touch()
    output_directory = tmp_path / "result"

    def completed(command: tuple[str, ...], **_: object) -> subprocess.CompletedProcess[str]:
        if "--version" in command:
            return subprocess.CompletedProcess(command, 0, "roofer 1.0.0\n", "")
        output_directory.joinpath("tile.city.jsonl").write_text("{}\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "complete\n", "")

    with patch("urbanstock3d.providers.roofer.subprocess.run", side_effect=completed) as run:
        result = RooferClient(str(executable)).reconstruct(
            point_cloud,
            footprint,
            output_directory,
            lod12=True,
        )

    command = run.call_args_list[1].args[0]
    assert result.version == "roofer 1.0.0"
    assert result.output_files == (output_directory / "tile.city.jsonl",)
    assert "--lod12" in command
    assert "--lod22" in command
    assert str(point_cloud.resolve()) in command


def test_roofer_client_reports_process_failure(tmp_path: Path) -> None:
    executable = tmp_path / "roofer.exe"
    executable.touch()
    process = Mock(returncode=2, stdout="", stderr="invalid input")

    with (
        patch("urbanstock3d.providers.roofer.subprocess.run", return_value=process),
        pytest.raises(RooferExecutionError, match="invalid input"),
    ):
        RooferClient(str(executable)).version()
