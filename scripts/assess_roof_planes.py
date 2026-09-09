"""Extract bounded preliminary roof-plane evidence from a local LiDAR crop."""

import argparse
import json
from pathlib import Path
from typing import Any

from urbanstock3d.providers.pnoa_lidar import footprint_polygons_utm
from urbanstock3d.reconstruction.quality.ransac import RansacParameters, assess_roof_planes


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lidar_crop", type=Path)
    parser.add_argument("building_geojson", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distance-threshold-m", type=float, default=0.20)
    parser.add_argument("--max-planes", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=200)
    return parser.parse_args()


def build_report(
    lidar_crop: Path,
    building_geojson: Path,
    output: Path,
    *,
    parameters: RansacParameters,
) -> Path:
    building: dict[str, Any] = json.loads(building_geojson.read_text(encoding="utf-8"))
    footprint = footprint_polygons_utm(building["geometry"])
    evidence = assess_roof_planes(lidar_crop, footprint, parameters=parameters)
    payload = {
        "schema_version": 1,
        "kind": "bounded_preliminary_roof_planes",
        "building_id": building["id"],
        "source": {
            "lidar_crop": lidar_crop.as_posix(),
            "footprint": building_geojson.as_posix(),
        },
        "parameters": {
            "sample_size": parameters.sample_size,
            "max_iterations": parameters.max_iterations,
            "max_planes": parameters.max_planes,
            "distance_threshold_m": parameters.distance_threshold_m,
            "minimum_support_points": parameters.minimum_support_points,
            "minimum_support_ratio": parameters.minimum_support_ratio,
            "normal_cluster_angle_deg": parameters.normal_cluster_angle_deg,
            "random_seed": parameters.random_seed,
        },
        "evidence": evidence.to_dict(),
        "interpretation": "pre-routing evidence only; planes are not reconstructed topology",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    arguments = parse_arguments()
    parameters = RansacParameters(
        distance_threshold_m=arguments.distance_threshold_m,
        max_planes=arguments.max_planes,
        max_iterations=arguments.max_iterations,
    )
    output = build_report(
        arguments.lidar_crop,
        arguments.building_geojson,
        arguments.output,
        parameters=parameters,
    )
    print(f"Wrote preliminary roof-plane evidence to {output}")


if __name__ == "__main__":
    main()
