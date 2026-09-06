"""Inspect and process local LiDAR point-cloud artifacts."""

from pathlib import Path
from typing import Any

import laspy


def inspect_lidar_header(path: Path) -> dict[str, Any]:
    """Read reproducibility and quality metadata without loading all points."""
    with laspy.open(path) as reader:
        header = reader.header
        crs = header.parse_crs()
        return {
            "las_version": str(header.version),
            "point_format": header.point_format.id,
            "point_count": header.point_count,
            "scales": header.scales.tolist(),
            "offsets": header.offsets.tolist(),
            "minimums": header.mins.tolist(),
            "maximums": header.maxs.tolist(),
            "dimensions": list(header.point_format.dimension_names),
            "crs": crs.to_string() if crs is not None else None,
        }
