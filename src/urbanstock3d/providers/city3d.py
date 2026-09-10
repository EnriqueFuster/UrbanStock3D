"""Native City3D wrapper integration."""

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from urbanstock3d.errors import City3DExecutionError


@dataclass(frozen=True)
class City3DRun:
    """Artifacts and diagnostics produced by one City3D invocation."""

    command: tuple[str, ...]
    output_file: Path
    stdout: str
    stderr: str


class City3DClient:
    """Run the UrbanStock3D City3D wrapper without a command shell."""

    def __init__(
        self,
        executable: str = "urbanstock-city3d",
        *,
        timeout_seconds: float = 600.0,
        runtime_directory: Path | None = None,
    ):
        if timeout_seconds <= 0:
            raise ValueError("City3D timeout must be positive")
        self.executable = _resolve_executable(executable)
        self.timeout_seconds = timeout_seconds
        if runtime_directory is not None and not runtime_directory.is_dir():
            raise City3DExecutionError(
                f"City3D runtime directory does not exist: {runtime_directory}"
            )
        self.runtime_directory = runtime_directory

    def reconstruct(
        self,
        point_cloud: Path,
        footprint: Path,
        output_file: Path,
        *,
        ground_elevation_m: float,
    ) -> City3DRun:
        """Reconstruct one OBJ model through the stable wrapper contract."""
        for source in (point_cloud, footprint):
            if not source.is_file():
                raise FileNotFoundError(source)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        command = (
            str(self.executable),
            str(point_cloud.resolve()),
            str(footprint.resolve()),
            str(output_file.resolve()),
            str(ground_elevation_m),
        )
        try:
            environment = os.environ.copy()
            if self.runtime_directory is not None:
                environment["PATH"] = (
                    f"{self.runtime_directory.resolve()}{os.pathsep}{environment['PATH']}"
                )
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=output_file.parent,
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise City3DExecutionError(f"Unable to execute City3D: {error}") from error
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "no process output"
            raise City3DExecutionError(f"City3D exited with code {result.returncode}: {detail}")
        if not output_file.is_file():
            raise City3DExecutionError("City3D completed without producing its OBJ output")
        return City3DRun(command, output_file, result.stdout, result.stderr)


def _resolve_executable(executable: str) -> Path:
    candidate = Path(executable).expanduser()
    if candidate.parent != Path("."):
        if candidate.is_file():
            return candidate.resolve()
        raise City3DExecutionError(f"Configured City3D executable does not exist: {candidate}")
    discovered = shutil.which(executable)
    if discovered is None:
        raise City3DExecutionError("City3D wrapper was not found; provide its executable path")
    return Path(discovered).resolve()
