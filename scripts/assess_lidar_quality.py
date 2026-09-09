"""Assess pre-reconstruction LiDAR quality for one resolved building."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from urbanstock3d.providers.pnoa_lidar import footprint_polygons_utm
from urbanstock3d.reconstruction.quality.lidar import CoverageGrid, assess_lidar_quality


def parse_arguments() -> argparse.Namespace:
    """Read local input and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lidar_crop", type=Path)
    parser.add_argument("building_geojson", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, help="Optional 1 m coverage diagnostic PNG.")
    return parser.parse_args()


def assess_files(
    lidar_crop: Path,
    building_geojson: Path,
    output: Path,
    *,
    plot: Path | None = None,
) -> Path:
    """Assess local artifacts and persist a JSON quality report."""
    building: dict[str, Any] = json.loads(building_geojson.read_text(encoding="utf-8"))
    footprint = footprint_polygons_utm(building["geometry"])
    report, coverage_grid = assess_lidar_quality(lidar_crop, footprint)
    payload = {
        "schema_version": 1,
        "building_id": building["id"],
        "source": {"lidar_crop": lidar_crop.as_posix(), "footprint": building_geojson.as_posix()},
        "quality": asdict(report),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if plot is not None:
        _write_coverage_plot(coverage_grid, plot)
    return output


def _write_coverage_plot(grid: CoverageGrid, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    values = grid.footprint_mask.astype(float)
    values[grid.occupied_mask & grid.footprint_mask] = 2.0
    figure, axis = plt.subplots(figsize=(8, 7))
    axis.imshow(values, origin="lower", cmap="Blues", vmin=0, vmax=2, interpolation="nearest")
    axis.set(title="LiDAR roof coverage (1 m grid)", xlabel="Grid column", ylabel="Grid row")
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main() -> None:
    """Run LiDAR quality assessment from the command line."""
    arguments = parse_arguments()
    output = assess_files(
        arguments.lidar_crop,
        arguments.building_geojson,
        arguments.output,
        plot=arguments.plot,
    )
    print(f"Wrote LiDAR quality report to {output}")


if __name__ == "__main__":
    main()
