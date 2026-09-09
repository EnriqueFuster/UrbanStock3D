"""Subtract an official terrain COG from an observed LiDAR height raster."""

import argparse
import json
from pathlib import Path

import numpy as np

from urbanstock3d.reconstruction.evidence import (
    build_normalized_height_raster,
    load_height_raster,
    save_normalized_height_raster,
)


def parse_arguments() -> argparse.Namespace:
    """Read observed-height, terrain and output locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observed_height", type=Path)
    parser.add_argument("terrain_source", help="Local GeoTIFF/COG path or official COG URL.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def build_files(
    observed_height: Path,
    terrain_source: str,
    output: Path,
    manifest: Path,
) -> Path:
    """Build and describe one ground-normalized height raster."""
    observed = load_height_raster(observed_height)
    normalized = build_normalized_height_raster(observed, terrain_source)
    save_normalized_height_raster(normalized, output)
    valid_heights = normalized.height_m[normalized.valid_mask]
    payload = {
        "schema_version": 1,
        "kind": "lidar_observed_ndsm",
        "source_observed_height": observed_height.as_posix(),
        "source_terrain": terrain_source,
        "output": output.as_posix(),
        "crs": normalized.crs,
        "resolution_m": normalized.resolution_m,
        "valid_ratio": normalized.valid_ratio,
        "valid_cell_count": int(normalized.valid_mask.sum()),
        "height_m": {
            "p05": float(np.percentile(valid_heights, 5)),
            "p50": float(np.percentile(valid_heights, 50)),
            "p95": float(np.percentile(valid_heights, 95)),
        },
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    """Build a ground-normalized height raster from local or remote terrain."""
    arguments = parse_arguments()
    manifest = build_files(
        arguments.observed_height,
        arguments.terrain_source,
        arguments.output,
        arguments.manifest,
    )
    print(f"Wrote normalized height manifest to {manifest}")


if __name__ == "__main__":
    main()
