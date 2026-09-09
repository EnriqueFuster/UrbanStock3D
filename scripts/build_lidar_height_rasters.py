"""Build observed multi-resolution roof-height rasters from a LiDAR crop."""

import argparse
import json
from pathlib import Path
from typing import Any

from urbanstock3d.providers.pnoa_lidar import footprint_polygons_utm
from urbanstock3d.reconstruction.evidence import (
    build_lidar_height_rasters,
    save_height_raster,
)


def parse_arguments() -> argparse.Namespace:
    """Read local evidence paths and rasterization parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lidar_crop", type=Path)
    parser.add_argument("building_geojson", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height-percentile", type=float, default=90.0)
    return parser.parse_args()


def build_files(
    lidar_crop: Path,
    building_geojson: Path,
    output_dir: Path,
    *,
    height_percentile: float = 90.0,
) -> Path:
    """Build raster artifacts and a JSON provenance manifest."""
    building: dict[str, Any] = json.loads(building_geojson.read_text(encoding="utf-8"))
    footprint = footprint_polygons_utm(building["geometry"])
    rasters = build_lidar_height_rasters(
        lidar_crop,
        footprint,
        height_percentile=height_percentile,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for raster in rasters:
        resolution_cm = round(raster.resolution_m * 100)
        path = save_height_raster(raster, output_dir / f"lidar_height_{resolution_cm:03d}cm.npz")
        artifacts.append(
            {
                "path": path.as_posix(),
                "resolution_m": raster.resolution_m,
                "shape": list(raster.elevation_m.shape),
                "valid_ratio": raster.valid_ratio,
                "observed_cell_count": int(raster.valid_mask.sum()),
            }
        )
    manifest = {
        "schema_version": 1,
        "kind": "lidar_observed_height",
        "building_id": building["id"],
        "source_lidar": lidar_crop.as_posix(),
        "source_footprint": building_geojson.as_posix(),
        "crs": rasters[0].crs,
        "height_units": "metres",
        "height_reference": "orthometric absolute elevation",
        "aggregation": {"statistic": "percentile", "value": height_percentile},
        "interpolation": "none",
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "lidar_height_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    """Build observed height raster artifacts."""
    arguments = parse_arguments()
    manifest = build_files(
        arguments.lidar_crop,
        arguments.building_geojson,
        arguments.output_dir,
        height_percentile=arguments.height_percentile,
    )
    print(f"Wrote LiDAR height manifest to {manifest}")


if __name__ == "__main__":
    main()
