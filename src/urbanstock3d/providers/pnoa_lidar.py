"""Locate PNOA-LiDAR kilometre grid cells for a building footprint."""

import math
import re
from collections.abc import Iterator, Sequence
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
CNIG_CATALOG_BASE_URL = "https://centrodedescargas.cnig.es/CentroDescargas/"
CNIG_LIDAR_SERIES = "LIDA3"
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
    rings = footprint_rings_utm(coordinates)
    positions = [position for ring in rings for position in ring]
    if not positions:
        raise ValueError("Building geometry has no coordinates")
    xs, ys = zip(*positions, strict=True)
    return min(xs), min(ys), max(xs), max(ys)


def footprint_rings_utm(
    coordinates: list[list[list[float]]],
) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Transform GeoJSON polygon rings to ETRS89 / UTM zone 30N."""
    if not coordinates or any(not ring for ring in coordinates):
        raise ValueError("Building geometry has no coordinates")
    transformer = Transformer.from_crs(WGS84, PENINSULA_UTM, always_xy=True)
    return tuple(
        tuple(transformer.transform(position[0], position[1]) for position in ring)
        for ring in coordinates
    )


def footprint_polygons_utm(
    geometry: dict[str, object],
) -> tuple[tuple[tuple[tuple[float, float], ...], ...], ...]:
    """Transform GeoJSON Polygon or MultiPolygon coordinates to UTM polygons."""
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list):
        raise ValueError("Building geometry has no coordinates")
    if geometry_type == "Polygon":
        polygon_coordinates = [coordinates]
    elif geometry_type == "MultiPolygon":
        polygon_coordinates = coordinates
    else:
        raise ValueError("Building geometry must be Polygon or MultiPolygon")
    return tuple(footprint_rings_utm(polygon) for polygon in polygon_coordinates)


def footprint_geometry_bbox_utm(
    geometry: dict[str, object],
) -> tuple[float, float, float, float]:
    """Return a projected bbox for a GeoJSON Polygon or MultiPolygon."""
    polygons = footprint_polygons_utm(geometry)
    positions = [position for polygon in polygons for ring in polygon for position in ring]
    xs, ys = zip(*positions, strict=True)
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


def discover_cnig_lidar_asset(
    client: httpx.Client,
    coordinates: Sequence[object],
) -> CnigLidarAsset:
    """Resolve the current third-coverage LAZ at a footprint's mean position."""
    positions = list(_iter_geojson_positions(coordinates))
    if not positions:
        raise ValueError("Building geometry has no coordinates")
    longitude = sum(position[0] for position in positions) / len(positions)
    latitude = sum(position[1] for position in positions) / len(positions)
    return _discover_cnig_lidar_asset_at_point(client, longitude, latitude)


def discover_cnig_lidar_assets(
    client: httpx.Client,
    geometry: dict[str, object],
    *,
    buffer_m: float = 0,
) -> tuple[CnigLidarAsset, ...]:
    """Resolve every third-coverage LAZ intersecting a buffered footprint bbox."""
    cells = required_grid_cells(footprint_geometry_bbox_utm(geometry), buffer_m=buffer_m)
    to_wgs84 = Transformer.from_crs(PENINSULA_UTM, WGS84, always_xy=True)
    assets: list[CnigLidarAsset] = []
    for cell in cells:
        center_x = (cell.easting_km + 0.5) * GRID_SIZE_M
        center_y = (cell.northing_km + 0.5) * GRID_SIZE_M
        longitude, latitude = to_wgs84.transform(center_x, center_y)
        asset = _discover_cnig_lidar_asset_at_point(client, longitude, latitude)
        if asset.grid_cell != cell.identifier:
            raise ValueError(
                f"CNIG returned grid cell {asset.grid_cell}, expected {cell.identifier}"
            )
        assets.append(asset)
    return tuple(dict.fromkeys(assets))


def _discover_cnig_lidar_asset_at_point(
    client: httpx.Client,
    longitude: float,
    latitude: float,
) -> CnigLidarAsset:
    """Resolve one current third-coverage LAZ at a WGS84 point."""
    point = (
        '{"type":"FeatureCollection","features":[{"type":"Feature",'
        f'"geometry":{{"type":"Point","coordinates":[{longitude},{latitude}]}}}}]}}'
    )
    search_data = {
        "lon": str(longitude),
        "lat": str(latitude),
        "series": CNIG_LIDAR_SERIES,
        "codSerie": CNIG_LIDAR_SERIES,
        "coordenadas": point,
        "codAgr": "MOMDT",
    }
    catalogue_url = f"{CNIG_CATALOG_BASE_URL}buscadorCatalogo.do?codFamilia=LIDAR"
    client.get(catalogue_url).raise_for_status()
    client.post(
        f"{CNIG_CATALOG_BASE_URL}resultados-busqueda-visor",
        data=search_data,
    ).raise_for_status()
    listing_response = client.get(
        f"{CNIG_CATALOG_BASE_URL}archivosTotalesSerieVisor",
        params={
            "numPagina": "1",
            "codAgr": "MOMDT",
            "codSerie": CNIG_LIDAR_SERIES,
            "coordenadas": point,
        },
    )
    listing_response.raise_for_status()
    filename, sequential_id = parse_cnig_lidar_listing(listing_response.text)
    detail_url = f"{CNIG_CATALOG_BASE_URL}detalleArchivo?sec={sequential_id}"
    detail_response = client.get(detail_url)
    detail_response.raise_for_status()
    asset = parse_cnig_asset_page(detail_response.content, detail_url)
    normalized_listing_name = re.sub(r"[^a-z0-9]", "", filename.casefold())
    normalized_detail_name = re.sub(r"[^a-z0-9]", "", asset.filename.casefold())
    if normalized_detail_name != normalized_listing_name:
        raise ValueError("CNIG listing and detail page identify different LiDAR assets")
    return asset


def _iter_geojson_positions(coordinates: Sequence[object]) -> Iterator[tuple[float, float]]:
    for item in coordinates:
        if (
            isinstance(item, list)
            and len(item) >= 2
            and isinstance(item[0], int | float)
            and isinstance(item[1], int | float)
        ):
            yield float(item[0]), float(item[1])
        elif isinstance(item, list):
            yield from _iter_geojson_positions(item)


def parse_cnig_lidar_listing(content: str) -> tuple[str, str]:
    """Extract the unique third-coverage LAZ and detail identifier from a listing."""
    rows = re.findall(r"<tr\b[^>]*>(.*?)</tr>", content, flags=re.IGNORECASE | re.DOTALL)
    assets: list[tuple[str, str]] = []
    for row in rows:
        filename_match = re.search(
            r"(PNOA[-_]\d{4}[-_][A-Z]+[-_]\d+-\d+[-_]H\d+[-_]NPC\d+\.laz)",
            row,
            flags=re.IGNORECASE,
        )
        detail_match = re.search(r"detalleArchivo\?sec=(\d+)", row, flags=re.IGNORECASE)
        if filename_match is not None and detail_match is not None:
            assets.append((filename_match.group(1), detail_match.group(1)))
    unique_assets = list(dict.fromkeys(assets))
    if len(unique_assets) != 1:
        raise ValueError(
            f"Expected exactly one third-coverage CNIG LiDAR asset, found {len(unique_assets)}"
        )
    return unique_assets[0]


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
