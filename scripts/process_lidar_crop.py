"""Download a CNIG LAZ temporarily and summarize a building context crop."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import httpx

from urbanstock3d.config import Settings
from urbanstock3d.processors.lidar import (
    inspect_lidar_header,
    summarize_building_lidar,
    summarize_lidar_bbox,
)
from urbanstock3d.providers.pnoa_lidar import (
    download_cnig_asset,
    footprint_bbox_utm,
    footprint_rings_utm,
    parse_cnig_asset_page,
)


def parse_arguments() -> argparse.Namespace:
    """Read the source asset, building artifact and crop options."""
    parser = argparse.ArgumentParser(description="Summarize a local PNOA-LiDAR crop.")
    parser.add_argument("detail_url")
    parser.add_argument("building_geojson", type=Path)
    parser.add_argument("--buffer-m", type=float, default=25.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def buffered_bbox(
    bbox: tuple[float, float, float, float], buffer_m: float
) -> tuple[float, float, float, float]:
    """Expand a projected bbox by a metric buffer."""
    if buffer_m < 0:
        raise ValueError("Buffer must be non-negative")
    min_x, min_y, max_x, max_y = bbox
    return min_x - buffer_m, min_y - buffer_m, max_x + buffer_m, max_y + buffer_m


def process_remote_crop(
    detail_url: str,
    building_geojson: Path,
    *,
    buffer_m: float,
    output: Path,
) -> Path:
    """Download, crop and summarize one building context without retaining raw data."""
    building: dict[str, Any] = json.loads(building_geojson.read_text(encoding="utf-8"))
    geometry = building["geometry"]
    if geometry["type"] != "Polygon":
        raise ValueError("LiDAR crop currently supports Polygon buildings only")
    building_bbox = footprint_bbox_utm(geometry["coordinates"])
    building_rings = footprint_rings_utm(geometry["coordinates"])
    crop_bbox = buffered_bbox(building_bbox, buffer_m)

    settings = Settings()
    timeout = httpx.Timeout(
        settings.http_read_timeout_seconds,
        connect=settings.http_connect_timeout_seconds,
    )
    with httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "UrbanStock3D/0.1 data-audit"},
    ) as client:
        detail_response = client.get(detail_url)
        detail_response.raise_for_status()
        asset = parse_cnig_asset_page(detail_response.content, detail_url)

        with TemporaryDirectory(prefix="urbanstock3d-lidar-") as temporary_dir:
            raw_path = Path(temporary_dir) / asset.filename
            download = download_cnig_asset(client, asset, raw_path)
            header = inspect_lidar_header(raw_path)
            crop = summarize_lidar_bbox(raw_path, crop_bbox)
            building_summary = summarize_building_lidar(
                raw_path,
                building_rings,
                crop_bbox,
            )

    report = {
        "provider": "IGN-CNIG",
        "product": "PNOA-LiDAR third coverage (2022-2025)",
        "status": "complete",
        "building_id": building["id"],
        "asset": asdict(asset),
        "download": asdict(download),
        "header": header,
        "crop": {
            "kind": "building_bbox_with_buffer",
            "buffer_m": buffer_m,
            **crop.to_dict(),
            "quality_flags": [
                flag
                for flag, applies in {
                    "elevation_outliers_suspected": (
                        crop.z_p01_m - crop.z_min_m > 20 or crop.z_max_m - crop.z_p99_m > 20
                    ),
                    "legacy_overlap_class_present": (crop.legacy_overlap_class_point_count > 0),
                }.items()
                if applies
            ],
        },
        "building": {
            "selection": "classified building points inside cadastral footprint",
            "ground_reference": "median class-2 elevation in context crop",
            **building_summary.to_dict(),
            "quality_flags": [
                flag
                for flag, applies in {
                    "legacy_overlap_class_excluded": (
                        building_summary.legacy_overlap_class_point_count > 0
                    ),
                    "low_building_point_count": building_summary.building_point_count < 100,
                    "low_ground_point_count": (building_summary.context_ground_point_count < 100),
                    "usable_density_below_published_density": (
                        building_summary.usable_point_density_m2 < asset.density_points_m2
                    ),
                }.items()
                if applies
            ],
        },
        "raw_source_retained": False,
        "crop_source_retained": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    """Run the temporary LiDAR crop processor."""
    args = parse_arguments()
    report_path = process_remote_crop(
        args.detail_url,
        args.building_geojson,
        buffer_m=args.buffer_m,
        output=args.output,
    )
    print(f"Wrote LiDAR crop report to {report_path}")


if __name__ == "__main__":
    main()
