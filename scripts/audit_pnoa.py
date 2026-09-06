"""Audit PNOA WMS coverage for an exported cadastral building."""

import argparse
import json
import math
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import httpx
from PIL import Image
from pyproj import Transformer

from urbanstock3d.config import Settings

WGS84 = "EPSG:4326"
PNOA_CRS = "EPSG:25830"
MAX_IMAGE_DIMENSION = 4096


def parse_arguments() -> argparse.Namespace:
    """Read the building artifact and audit parameters."""
    parser = argparse.ArgumentParser(description="Audit PNOA coverage for one building.")
    parser.add_argument("building_geojson", type=Path)
    parser.add_argument("--buffer-m", type=float, default=15.0)
    parser.add_argument("--pixel-size-m", type=float, default=0.25)
    parser.add_argument("--save-preview", action="store_true")
    return parser.parse_args()


def projected_bbox(
    coordinates: list[list[list[float]]],
    *,
    buffer_m: float,
) -> tuple[float, float, float, float]:
    """Transform GeoJSON polygon coordinates to a buffered PNOA bbox."""
    if buffer_m < 0:
        raise ValueError("Buffer must be non-negative")
    positions = [position for ring in coordinates for position in ring]
    if not positions:
        raise ValueError("Building geometry has no coordinates")

    transformer = Transformer.from_crs(WGS84, PNOA_CRS, always_xy=True)
    projected = [
        cast(tuple[float, float], transformer.transform(position[0], position[1]))
        for position in positions
    ]
    xs, ys = zip(*projected, strict=True)
    return (
        min(xs) - buffer_m,
        min(ys) - buffer_m,
        max(xs) + buffer_m,
        max(ys) + buffer_m,
    )


def image_dimensions(
    bbox: tuple[float, float, float, float],
    pixel_size_m: float,
) -> tuple[int, int]:
    """Calculate WMS image dimensions for a target ground pixel size."""
    if pixel_size_m <= 0:
        raise ValueError("Pixel size must be positive")
    min_x, min_y, max_x, max_y = bbox
    width = math.ceil((max_x - min_x) / pixel_size_m)
    height = math.ceil((max_y - min_y) / pixel_size_m)
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise ValueError("Requested PNOA image exceeds the audit size limit")
    return width, height


def audit_pnoa(
    building_geojson: Path,
    *,
    buffer_m: float,
    pixel_size_m: float,
    save_preview: bool = False,
) -> Path:
    """Request, validate and record a small PNOA WMS image."""
    geojson: dict[str, Any] = json.loads(building_geojson.read_text(encoding="utf-8"))
    geometry = geojson["geometry"]
    if geometry["type"] != "Polygon":
        raise ValueError("PNOA audit currently supports Polygon buildings only")

    bbox = projected_bbox(geometry["coordinates"], buffer_m=buffer_m)
    width, height = image_dimensions(bbox, pixel_size_m)
    settings = Settings()
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": settings.pnoa_wms_layer,
        "STYLES": "",
        "CRS": PNOA_CRS,
        "BBOX": ",".join(str(value) for value in bbox),
        "WIDTH": str(width),
        "HEIGHT": str(height),
        "FORMAT": "image/jpeg",
    }
    timeout = httpx.Timeout(
        settings.http_read_timeout_seconds,
        connect=settings.http_connect_timeout_seconds,
    )
    with httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "UrbanStock3D/0.1 data-audit"},
    ) as client:
        response = client.get(str(settings.pnoa_wms_url), params=params)
        response.raise_for_status()

    with Image.open(BytesIO(response.content)) as image:
        image.load()
        actual_size = image.size
        image_format = image.format
        image_mode = image.mode
        extrema = image.getextrema()

    if actual_size != (width, height):
        raise ValueError(f"PNOA returned image size {actual_size}, expected {(width, height)}")

    preview_path = building_geojson.with_name("pnoa_preview.jpg")
    if save_preview:
        preview_path.write_bytes(response.content)

    report = {
        "provider": "IGN-CNIG",
        "product": "PNOA maximum currentness orthophoto",
        "endpoint": str(settings.pnoa_wms_url),
        "layer": settings.pnoa_wms_layer,
        "status": "available",
        "request": {
            "crs": PNOA_CRS,
            "bbox": bbox,
            "buffer_m": buffer_m,
            "pixel_size_m": pixel_size_m,
            "width": width,
            "height": height,
        },
        "response": {
            "http_status": response.status_code,
            "content_type": response.headers.get("content-type"),
            "bytes": len(response.content),
            "image_format": image_format,
            "image_mode": image_mode,
            "channel_extrema": extrema,
        },
        "preview_path": preview_path.name if save_preview else None,
    }
    report_path = building_geojson.with_name("pnoa_audit.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    """Run the PNOA audit from the command line."""
    args = parse_arguments()
    report_path = audit_pnoa(
        args.building_geojson,
        buffer_m=args.buffer_m,
        pixel_size_m=args.pixel_size_m,
        save_preview=args.save_preview,
    )
    print(f"Wrote PNOA audit to {report_path}")


if __name__ == "__main__":
    main()
