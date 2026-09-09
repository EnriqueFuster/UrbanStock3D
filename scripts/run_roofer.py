"""Run native Roofer for one prepared building dataset."""

import argparse
from pathlib import Path

from urbanstock3d.config import Settings
from urbanstock3d.providers.roofer import RooferClient


def parse_arguments() -> argparse.Namespace:
    """Read reconstruction paths and LoD options."""
    parser = argparse.ArgumentParser(description="Run native Roofer reconstruction.")
    parser.add_argument("point_cloud", type=Path)
    parser.add_argument("footprint", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--executable")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--lod12", action="store_true")
    parser.add_argument("--lod13", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Execute Roofer and report its version and artifacts."""
    args = parse_arguments()
    settings = Settings()
    client = RooferClient(args.executable or settings.roofer_executable)
    run = client.reconstruct(
        args.point_cloud,
        args.footprint,
        args.output_directory,
        jobs=args.jobs,
        lod12=args.lod12,
        lod13=args.lod13,
    )
    print(f"Roofer version: {run.version}")
    for output in run.output_files:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
