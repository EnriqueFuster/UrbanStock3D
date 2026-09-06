"""Locate PNOA-LiDAR kilometre grid cells for a building footprint."""

import math
import re
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from time import perf_counter
from urllib.parse import urlencode

import httpx
from pyproj import Transformer

WGS84 = "EPSG:4326"
PENINSULA_UTM = "EPSG:25830"
CNIG_SEARCH_URL = "https://centrodedescargas.cnig.es/CentroDescargas/buscador.do"
GRID_SIZE_M = 1000


@dataclass(frozen=True)
class LidarGridCell:
    """One 1 x 1 km PNOA-LiDAR distribution cell."""

    easting_km: int
    northing_km: int
    utm_zone: int = 30

    @property
    def identifier(self) -> str:
        """Return the coordinate-based cell identifier used for discovery."""
        north_edge_km = self.northing_km + 1
        return f"{self.easting_km}-{north_edge_km}-H{self.utm_zone}"


@dataclass(frozen=True)
class CnigLidarAsset:
    """Metadata published by CNIG for one downloadable LAZ asset."""

    detail_url: str
    sequential_id: str
    filename: str
    flight_year: int
    density_points_m2: float
    size_mb: float
    grid_cell: str
    processing_level: str


@dataclass(frozen=True)
class DownloadResult:
    """Measurements from downloading one source asset."""

    bytes_downloaded: int
    elapsed_seconds: float


def footprint_bbox_utm(
    coordinates: list[list[list[float]]],
) -> tuple[float, float, float, float]:
    """Transform polygon coordinates from WGS84 to ETRS89 / UTM zone 30N."""
    positions = [position for ring in coordinates for position in ring]
    if not positions:
        raise ValueError("Building geometry has no coordinates")

    transformer = Transformer.from_crs(WGS84, PENINSULA_UTM, always_xy=True)
    projected = [transformer.transform(position[0], position[1]) for position in positions]
    xs, ys = zip(*projected, strict=True)
    return min(xs), min(ys), max(xs), max(ys)


def required_grid_cells(
    bbox: tuple[float, float, float, float],
    *,
    buffer_m: float = 0,
) -> tuple[LidarGridCell, ...]:
    """Return every kilometre cell intersecting a buffered projected bbox."""
    if buffer_m < 0:
        raise ValueError("Buffer must be non-negative")
    min_x, min_y, max_x, max_y = bbox
    if min_x > max_x or min_y > max_y:
        raise ValueError("Bounding box limits are invalid")

    first_x = math.floor((min_x - buffer_m) / GRID_SIZE_M)
    first_y = math.floor((min_y - buffer_m) / GRID_SIZE_M)
    last_x = math.floor(math.nextafter(max_x + buffer_m, -math.inf) / GRID_SIZE_M)
    last_y = math.floor(math.nextafter(max_y + buffer_m, -math.inf) / GRID_SIZE_M)
    return tuple(
        LidarGridCell(easting_km=x, northing_km=y)
        for x in range(first_x, last_x + 1)
        for y in range(first_y, last_y + 1)
    )


def cnig_search_url(bbox_wgs84: tuple[float, float, float, float]) -> str:
    """Build an official CNIG catalogue search URL for a WGS84 bbox."""
    bbox = ",".join(f"{value:.8f}" for value in bbox_wgs84)
    return f"{CNIG_SEARCH_URL}?{urlencode({'BBOX': bbox, 'CRS': 'EPSG:4258'})}"


def parse_cnig_asset_page(content: bytes, detail_url: str) -> CnigLidarAsset:
    """Parse public metadata from a CNIG asset detail page."""
    text = content.decode("utf-8", errors="replace")
    plain_text = re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", text)))
    sequential_id = _required_match(r"secuencial\s*=\s*['\"](\d+)['\"]", text)
    filename = _required_match(
        r"(PNOA_(\d{4})_[A-Z]+_(\d+)-(\d+)_H(\d+)_NPC(\d+)\.LAZ)",
        plain_text,
    )
    filename_match = re.search(
        r"PNOA_(\d{4})_[A-Z]+_(\d+)-(\d+)_H(\d+)_(NPC\d+)\.LAZ",
        filename,
        flags=re.IGNORECASE,
    )
    if filename_match is None:
        raise ValueError("CNIG LiDAR filename does not follow the expected convention")

    density = _required_match(r"([\d.,]+)\s*ptos/m2", plain_text)
    size = _required_match(r"Tama[^:]*:\s*([\d.,]+)\s*\(Mb\)", plain_text)
    return CnigLidarAsset(
        detail_url=detail_url,
        sequential_id=sequential_id,
        filename=filename,
        flight_year=int(filename_match.group(1)),
        density_points_m2=float(density.replace(",", ".")),
        size_mb=float(size.replace(",", ".")),
        grid_cell=(
            f"{filename_match.group(2)}-{filename_match.group(3)}-H{filename_match.group(4)}"
        ),
        processing_level=filename_match.group(5).upper(),
    )


def download_cnig_asset(
    client: httpx.Client,
    asset: CnigLidarAsset,
    destination: Path,
) -> DownloadResult:
    """Download one CNIG asset through its published individual-download form."""
    form = {
        "secuencial": asset.sequential_id,
        "urlCart": "",
        "secDescDirLA": asset.sequential_id,
        "codSerie": "LIDA3",
        "id_productor": "",
    }
    endpoint = "https://centrodedescargas.cnig.es/CentroDescargas/descargaDir"
    started_at = perf_counter()
    bytes_downloaded = 0
    with client.stream("POST", endpoint, data=form) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for chunk in response.iter_bytes():
                output.write(chunk)
                bytes_downloaded += len(chunk)

    with destination.open("rb") as downloaded:
        if downloaded.read(4) != b"LASF":
            raise ValueError("CNIG response does not contain a LAS/LAZ file")
    return DownloadResult(
        bytes_downloaded=bytes_downloaded,
        elapsed_seconds=perf_counter() - started_at,
    )


def _required_match(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Required CNIG metadata is missing: {pattern}")
    return match.group(1)
