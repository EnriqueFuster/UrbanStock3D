"""Run City3D through the UrbanStock3D native wrapper."""

import argparse
from pathlib import Path

from urbanstock3d.providers.city3d import City3DClient
from urbanstock3d.reconstruction.backends.city3d import assess_city3d_inputs


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a native City3D reconstruction.")
    parser.add_argument("point_cloud", type=Path)
    parser.add_argument("footprint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--executable", default="urbanstock-city3d")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    assess_city3d_inputs(args.point_cloud, args.footprint)
    run = City3DClient(args.executable).reconstruct(
        args.point_cloud,
        args.footprint,
        args.output,
    )
    print(f"Wrote City3D OBJ to {run.output_file}")


if __name__ == "__main__":
    main()
