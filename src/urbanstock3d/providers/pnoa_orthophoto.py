"""Acquire georeferenced PNOA orthophoto crops for resolved buildings."""

import json
import math
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import httpx
import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.transform import from_bounds

from urbanstock3d.config import Settings

WGS84 = "EPSG:4326"
PNOA_CRS = "EPSG:25830"
MAX_IMAGE_DIMENSION = 4096


def projected_building_bbox(
    geometry: dict[str, Any], *, buffer_m: float
) -> tuple[float, float, float, float]:
    """Return a metric bounding box for a GeoJSON Polygon or MultiPolygon."""
    if buffer_m < 0:
        raise ValueError("Buffer must be non-negative")
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if geometry_type == "Polygon":
        polygons = [coordinates]
    elif geometry_type == "MultiPolygon":
        polygons = coordinates
    else:
        raise ValueError("Building geometry must be Polygon or MultiPolygon")

    positions = [position for polygon in polygons for ring in polygon for position in ring]
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
    bbox: tuple[float, float, float, float], pixel_size_m: float
) -> tuple[int, int]:
    """Calculate safe WMS dimensions for a requested ground pixel size."""
    if pixel_size_m <= 0:
        raise ValueError("Pixel size must be positive")
    min_x, min_y, max_x, max_y = bbox
    width = math.ceil((max_x - min_x) / pixel_size_m)
    height = math.ceil((max_y - min_y) / pixel_size_m)
    if width <= 0 or height <= 0:
        raise ValueError("Building bounding box must have positive area")
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise ValueError("Requested PNOA image exceeds the size limit")
    return width, height


def acquire_pnoa_crop(
    building_geojson: Path,
    output_path: Path,
    *,
    buffer_m: float = 10.0,
    pixel_size_m: float = 0.25,
    client: httpx.Client | None = None,
) -> Path:
    """Download a PNOA WMS crop and persist it as a georeferenced RGB GeoTIFF."""
    feature: dict[str, Any] = json.loads(building_geojson.read_text(encoding="utf-8"))
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("Building GeoJSON must contain a geometry")

    bbox = projected_building_bbox(geometry, buffer_m=buffer_m)
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
    client_context = (
        nullcontext(client)
        if client is not None
        else httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "UrbanStock3D/0.1 orthophoto-acquisition"},
        )
    )
    with client_context as active_client:
        response = active_client.get(str(settings.pnoa_wms_url), params=params)
        response.raise_for_status()

    with Image.open(BytesIO(response.content)) as source_image:
        image = np.asarray(source_image.convert("RGB"))
    if image.shape[:2] != (height, width):
        raise ValueError(
            f"PNOA returned image size {(image.shape[1], image.shape[0])}, "
            f"expected {(width, height)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_bounds(*bbox, width=width, height=height)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=3,
        dtype="uint8",
        crs=PNOA_CRS,
        transform=transform,
        compress="jpeg",
    ) as dataset:
        dataset.write(np.moveaxis(image, 2, 0))
        dataset.colorinterp = (
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
        )

    metadata_path = output_path.with_suffix(".json")
    metadata = {
        "provider": "IGN-CNIG",
        "product": "PNOA maximum currentness orthophoto",
        "source_url": str(response.request.url),
        "building_id": feature.get("id"),
        "raster": {
            "path": output_path.name,
            "crs": PNOA_CRS,
            "bbox": bbox,
            "width": width,
            "height": height,
            "pixel_size_m": pixel_size_m,
            "bands": ["red", "green", "blue"],
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path
