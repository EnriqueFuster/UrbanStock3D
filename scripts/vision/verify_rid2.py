"""Verify a manually acquired RID2 archive against pinned source metadata."""

import argparse
from pathlib import Path

from urbanstock3d.vision.roof_objects.datasets import load_rid2_source, verify_rid2_archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--source-config", type=Path, default=Path("config/vision/datasets/rid2.yaml")
    )
    arguments = parser.parse_args()
    source = load_rid2_source(arguments.source_config)
    if not source.download_allowed:
        raise SystemExit("RID2 licence is unresolved; archive use is blocked pending review")
    verify_rid2_archive(arguments.archive, source)
    print(f"Verified RID2 {source.source_version}: {arguments.archive}")


if __name__ == "__main__":
    main()
