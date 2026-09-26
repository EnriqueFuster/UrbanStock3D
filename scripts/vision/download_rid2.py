"""Download the pinned RID2 archive with resume and integrity verification."""

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx

from urbanstock3d.vision.roof_objects.datasets import load_rid2_source, verify_rid2_archive

REPORT_INTERVAL_BYTES = 256 * 1024 * 1024


@contextmanager
def exclusive_download(lock_path: Path) -> Iterator[None]:
    """Prevent concurrent writers from corrupting the resumable partial file."""
    try:
        lock = lock_path.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError(f"Another RID2 download owns {lock_path}") from error
    try:
        with lock:
            lock.write("RID2 download in progress\n")
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-config", type=Path, default=Path("config/vision/datasets/rid2.yaml")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/vision/rid2/raw"))
    arguments = parser.parse_args()

    source = load_rid2_source(arguments.source_config)
    if not source.download_allowed:
        raise SystemExit("The configured source is not open access")

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    archive = arguments.output_dir / source.archive_name
    with exclusive_download(arguments.output_dir / ".download.lock"):
        if archive.exists():
            verify_rid2_archive(archive, source)
            print(f"Already downloaded and verified: {archive}")
            return

        partial = archive.with_suffix(f"{archive.suffix}.part")
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        timeout = httpx.Timeout(connect=30.0, read=None, write=30.0, pool=30.0)

        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            with client.stream("GET", str(source.archive_url), headers=headers) as response:
                response.raise_for_status()
                resumed = offset > 0 and response.status_code == httpx.codes.PARTIAL_CONTENT
                mode = "ab" if resumed else "wb"
                downloaded = offset if resumed else 0
                next_report = downloaded + REPORT_INTERVAL_BYTES
                print(f"Downloading RID2 to {partial} from byte {downloaded:,}")
                with partial.open(mode) as stream:
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        stream.write(chunk)
                        downloaded += len(chunk)
                        if downloaded >= next_report:
                            print(f"Downloaded {downloaded / (1024**3):.2f} GiB")
                            next_report += REPORT_INTERVAL_BYTES

        partial.replace(archive)
        verify_rid2_archive(archive, source)
    print(f"Downloaded and verified RID2 {source.source_version}: {archive}")
    if not source.commercial_reuse_confirmed:
        print("Warning: keep dataset and derived weights private until reuse terms are confirmed")


if __name__ == "__main__":
    main()
