"""Evaluate normalized LiDAR height evidence and cross-resolution agreement."""

import argparse
import json
from pathlib import Path

from urbanstock3d.reconstruction.evidence import (
    compare_height_resolutions,
    evaluate_height_evidence,
    load_normalized_height_raster,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fine_ndsm", type=Path)
    parser.add_argument("coarse_ndsm", type=Path)
    parser.add_argument("--discontinuity-m", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_report(
    fine_path: Path,
    coarse_path: Path,
    output: Path,
    *,
    discontinuity_m: float = 1.0,
) -> Path:
    fine = load_normalized_height_raster(fine_path)
    coarse = load_normalized_height_raster(coarse_path)
    payload = {
        "schema_version": 1,
        "kind": "normalized_height_evidence_quality",
        "sources": {"fine": fine_path.as_posix(), "coarse": coarse_path.as_posix()},
        "fine": evaluate_height_evidence(fine, discontinuity_threshold_m=discontinuity_m).to_dict(),
        "coarse": evaluate_height_evidence(
            coarse, discontinuity_threshold_m=discontinuity_m
        ).to_dict(),
        "agreement": compare_height_resolutions(fine, coarse).to_dict(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output


def main() -> None:
    arguments = parse_arguments()
    report = build_report(
        arguments.fine_ndsm,
        arguments.coarse_ndsm,
        arguments.output,
        discontinuity_m=arguments.discontinuity_m,
    )
    print(f"Wrote height-evidence report to {report}")


if __name__ == "__main__":
    main()
