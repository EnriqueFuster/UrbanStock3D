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
    summarize_lidar_bbox,
    summarize_multipolygon_building_lidar,
    write_lidar_bbox_crop_from_sources,
)
from urbanstock3d.providers.pnoa_lidar import (
    CnigLidarAsset,
    discover_cnig_lidar_assets,
    download_cnig_asset,
    footprint_geometry_bbox_utm,
    footprint_polygons_utm,
    parse_cnig_asset_page,
)


def parse_arguments() -> argparse.Namespace:
    """Read the source asset, building artifact and crop options."""
    parser = argparse.ArgumentParser(description="Summarize a local PNOA-LiDAR crop.")
    parser.add_argument("building_geojson", type=Path)
    parser.add_argument(
        "--detail-url",
        type=http_url,
        help="Optional CNIG detail URL override; normally discovered automatically.",
    )
    parser.add_argument("--buffer-m", type=float, default=25.0)
    parser.add_argument(
        "--save-crop",
        type=Path,
        help="Optionally retain the small context crop as LAS/LAZ for exploration.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def http_url(value: str) -> str:
    """Reject placeholders and malformed CNIG detail URLs at the CLI boundary."""
    url = httpx.URL(value)
    if url.scheme not in {"http", "https"} or not url.host:
        raise argparse.ArgumentTypeError(
            "detail_url must be a complete http(s) URL, not a placeholder"
        )
    return value


def buffered_bbox(
    bbox: tuple[float, float, float, float], buffer_m: float
) -> tuple[float, float, float, float]:
    """Expand a projected bbox by a metric buffer."""
    if buffer_m < 0:
        raise ValueError("Buffer must be non-negative")
    min_x, min_y, max_x, max_y = bbox
    return min_x - buffer_m, min_y - buffer_m, max_x + buffer_m, max_y + buffer_m


def process_remote_crop(
    building_geojson: Path,
    *,
    buffer_m: float,
    output: Path,
    save_crop: Path | None = None,
    detail_url: str | None = None,
) -> Path:
    """Download, crop and summarize one building context without retaining raw data."""
    building: dict[str, Any] = json.loads(building_geojson.read_text(encoding="utf-8"))
    geometry = building["geometry"]
    building_bbox = footprint_geometry_bbox_utm(geometry)
    building_polygons = footprint_polygons_utm(geometry)
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
        assets = (
            discover_cnig_lidar_assets(client, geometry, buffer_m=buffer_m)
            if detail_url is None
            else (_load_asset_override(client, detail_url),)
        )

        with TemporaryDirectory(prefix="urbanstock3d-lidar-") as temporary_dir:
            temporary_root = Path(temporary_dir)
            raw_paths: list[Path] = []
            downloads: list[dict[str, Any]] = []
            source_headers: list[dict[str, Any]] = []
            for asset in assets:
                raw_path = temporary_root / asset.filename
                download = download_cnig_asset(client, asset, raw_path)
                raw_paths.append(raw_path)
                downloads.append({"sequential_id": asset.sequential_id, **asdict(download)})
                source_headers.append(
                    {"sequential_id": asset.sequential_id, **inspect_lidar_header(raw_path)}
                )

            combined_crop_path = save_crop or temporary_root / "lidar_context_crop.laz"
            saved_crop_point_count = write_lidar_bbox_crop_from_sources(
                tuple(raw_paths), combined_crop_path, crop_bbox
            )
            header = inspect_lidar_header(combined_crop_path)
            crop = summarize_lidar_bbox(combined_crop_path, crop_bbox)
            building_summary = summarize_multipolygon_building_lidar(
                combined_crop_path,
                building_polygons,
                crop_bbox,
            )

    report = {
        "provider": "IGN-CNIG",
        "product": "PNOA-LiDAR third coverage (2022-2025)",
        "status": "complete",
        "building_id": building["id"],
        "assets": [asdict(asset) for asset in assets],
        "downloads": downloads,
        "source_headers": source_headers,
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
                        building_summary.usable_point_density_m2
                        < min(asset.density_points_m2 for asset in assets)
                    ),
                }.items()
                if applies
            ],
        },
        "raw_source_retained": False,
        "crop_source_retained": save_crop is not None,
        "crop_source": (
            {
                "path": save_crop.as_posix(),
                "point_count": saved_crop_point_count,
                "kind": "building_bbox_with_buffer",
            }
            if save_crop is not None
            else None
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def _load_asset_override(client: httpx.Client, detail_url: str) -> CnigLidarAsset:
    """Load an explicitly selected asset for diagnostics and reproducibility."""
    response = client.get(detail_url)
    response.raise_for_status()
    return parse_cnig_asset_page(response.content, detail_url)


def main() -> None:
    """Run the temporary LiDAR crop processor."""
    args = parse_arguments()
    report_path = process_remote_crop(
        args.building_geojson,
        buffer_m=args.buffer_m,
        output=args.output,
        save_crop=args.save_crop,
        detail_url=args.detail_url,
    )
    print(f"Wrote LiDAR crop report to {report_path}")


if __name__ == "__main__":
    main()
