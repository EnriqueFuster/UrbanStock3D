"""Check whether existing LiDAR and footprint artifacts are ready for City3D."""

import argparse
import json
from pathlib import Path

from urbanstock3d.reconstruction.backends.city3d import assess_city3d_inputs


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate City3D reconstruction inputs.")
    parser.add_argument("point_cloud", type=Path)
    parser.add_argument("footprint", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    assessment = assess_city3d_inputs(args.point_cloud, args.footprint)
    payload = json.dumps(assessment.to_dict(), indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(f"Wrote City3D input assessment to {args.output}")


if __name__ == "__main__":
    main()
