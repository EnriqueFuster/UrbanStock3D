"""Native Roofer executable integration."""

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from urbanstock3d.errors import RooferExecutionError


@dataclass(frozen=True)
class RooferRun:
    """Artifacts and diagnostics produced by one Roofer invocation."""

    version: str
    command: tuple[str, ...]
    output_files: tuple[Path, ...]
    stdout: str
    stderr: str


class RooferClient:
    """Run a configured native Roofer executable without a command shell."""

    def __init__(self, executable: str = "roofer", *, timeout_seconds: float = 300.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Roofer timeout must be positive")
        self.executable = _resolve_executable(executable)
        self.timeout_seconds = timeout_seconds

    def version(self) -> str:
        """Return the executable version string."""
        with TemporaryDirectory(prefix="urbanstock3d-roofer-version-") as temporary_dir:
            result = self._run(
                (str(self.executable), "--version"),
                working_directory=Path(temporary_dir),
            )
        version = result.stdout.strip() or result.stderr.strip()
        if not version:
            raise RooferExecutionError("Roofer returned an empty version")
        return version

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
        """Reconstruct buildings and return the generated CityJSONSeq artifacts."""
        if jobs <= 0:
            raise ValueError("Roofer jobs must be positive")
        if not any((lod12, lod13, lod22)):
            raise ValueError("At least one Roofer level of detail must be enabled")
        for source in (point_cloud, footprint):
            if not source.is_file():
                raise FileNotFoundError(source)

        output_directory.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.executable),
            "--id-attribute",
            id_attribute,
            "--jobs",
            str(jobs),
            "--lod12" if lod12 else "--no-lod12",
            "--lod13" if lod13 else "--no-lod13",
            "--lod22" if lod22 else "--no-lod22",
            str(point_cloud.resolve()),
            str(footprint.resolve()),
            str(output_directory.resolve()),
        ]
        version = self.version()
        result = self._run(tuple(command), working_directory=output_directory)
        output_files = tuple(sorted(output_directory.glob("*.city.jsonl")))
        if not output_files:
            log_path = output_directory / "roofer.log.json"
            diagnostic = result.stderr.strip() or result.stdout.strip()
            detail = f" Inspect {log_path}." if log_path.is_file() else ""
            if diagnostic:
                detail += f" Process output: {diagnostic}"
            raise RooferExecutionError(
                f"Roofer completed without producing CityJSONSeq output.{detail}"
            )
        return RooferRun(
            version=version,
            command=tuple(command),
            output_files=output_files,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def _run(
        self,
        command: tuple[str, ...],
        *,
        working_directory: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=working_directory,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RooferExecutionError(f"Unable to execute Roofer: {error}") from error
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "no process output"
            raise RooferExecutionError(f"Roofer exited with code {result.returncode}: {detail}")
        return result


def _resolve_executable(executable: str) -> Path:
    candidate = Path(executable).expanduser()
    if candidate.parent != Path("."):
        if candidate.is_file():
            return candidate.resolve()
        raise RooferExecutionError(f"Configured Roofer executable does not exist: {candidate}")
    discovered = shutil.which(executable)
    if discovered is None:
        raise RooferExecutionError(
            "Roofer executable was not found; configure URBANSTOCK_ROOFER_EXECUTABLE"
        )
    return Path(discovered).resolve()
