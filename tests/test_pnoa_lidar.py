import httpx
import pytest
from pyproj import Transformer

import urbanstock3d.providers.pnoa_lidar as pnoa_lidar
from urbanstock3d.providers.pnoa_lidar import (
    CnigLidarAsset,
    LidarGridCell,
    cnig_search_url,
    discover_cnig_lidar_asset,
    discover_cnig_lidar_assets,
    parse_cnig_asset_page,
    parse_cnig_lidar_listing,
    required_grid_cells,
)


def test_grid_cell_has_coordinate_identifier() -> None:
    assert LidarGridCell(724, 4372).identifier == "724-4373-H30"


def test_required_grid_cells_includes_buffer_crossing_tile_boundary() -> None:
    cells = required_grid_cells((724980, 4372100, 724990, 4372200), buffer_m=25)

    assert cells == (
        LidarGridCell(724, 4372),
        LidarGridCell(725, 4372),
    )


def test_required_grid_cells_rejects_negative_buffer() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        required_grid_cells((724000, 4372000, 725000, 4373000), buffer_m=-1)


def test_cnig_search_url_uses_official_bbox_search() -> None:
    url = cnig_search_url((-0.4, 39.4, -0.3, 39.5))

    assert url.startswith("https://centrodedescargas.cnig.es/")
    assert "BBOX=" in url
    assert "CRS=EPSG%3A4258" in url


def test_parse_cnig_asset_page_maps_official_metadata() -> None:
    page = b"""
    <script>var secuencial = '12598284';</script>
    <p>Fichero: PNOA_2023_VAL_724-4373_H30_NPC02.laz</p>
    <p>Fecha: 2023</p><p>Densidad: 5 ptos/m2</p>
    <p>Tama&ntilde;o: 34.96 (Mb)</p><p>Formato: LAZ</p>
    """

    asset = parse_cnig_asset_page(page, "https://example.test/detail?sec=12598284")

    assert asset.sequential_id == "12598284"
    assert asset.grid_cell == "724-4373-H30"
    assert asset.processing_level == "NPC02"
    assert asset.flight_year == 2023
    assert asset.density_points_m2 == 5
    assert asset.size_mb == 34.96


def test_parse_cnig_lidar_listing_extracts_unique_asset() -> None:
    listing = """
    <table><tr><td>PNOA-2023-VAL-726-4374-H30-NPC02.laz</td>
    <td><a href="./detalleArchivo?sec=12585804">Details</a></td></tr></table>
    """

    assert parse_cnig_lidar_listing(listing) == (
        "PNOA-2023-VAL-726-4374-H30-NPC02.laz",
        "12585804",
    )


def test_parse_cnig_lidar_listing_rejects_missing_asset() -> None:
    with pytest.raises(ValueError, match="found 0"):
        parse_cnig_lidar_listing("<table></table>")


def test_discover_cnig_lidar_asset_uses_third_coverage_catalogue() -> None:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if request.url.path.endswith("archivosTotalesSerieVisor"):
            return httpx.Response(
                200,
                text=(
                    "<tr><td>PNOA-2023-VAL-726-4374-H30-NPC02.laz</td>"
                    '<td><a href="./detalleArchivo?sec=12585804">Details</a></td></tr>'
                ),
            )
        if request.url.path.endswith("detalleArchivo"):
            return httpx.Response(
                200,
                text="""
                <script>var secuencial = '12585804';</script>
                <p>PNOA_2023_VAL_726-4374_H30_NPC02.laz</p>
                <p>Densidad: 5 ptos/m2</p><p>Tama&ntilde;o: 20.5 (Mb)</p>
                """,
            )
        return httpx.Response(200, text="ok")

    coordinates = [[[-0.371, 39.479], [-0.370, 39.479], [-0.371, 39.479]]]
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        asset = discover_cnig_lidar_asset(client, coordinates)

    assert asset.sequential_id == "12585804"
    assert asset.grid_cell == "726-4374-H30"
    assert requested_paths[-1].endswith("detalleArchivo")


def test_discover_cnig_lidar_assets_resolves_each_intersecting_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    to_wgs84 = Transformer.from_crs("EPSG:25830", "EPSG:4326", always_xy=True)
    left = to_wgs84.transform(724980, 4372100)
    right = to_wgs84.transform(724990, 4372200)
    returned_cells = iter(("724-4373-H30", "725-4373-H30"))

    def fake_discovery(client: httpx.Client, longitude: float, latitude: float) -> CnigLidarAsset:
        del client, longitude, latitude
        cell = next(returned_cells)
        sequential_id = "1" if cell.startswith("724") else "2"
        return CnigLidarAsset(
            detail_url=f"https://example.test/detail?sec={sequential_id}",
            sequential_id=sequential_id,
            filename=f"PNOA_2023_VAL_{cell.replace('-H30', '_H30')}_NPC02.laz",
            flight_year=2023,
            density_points_m2=5.0,
            size_mb=20.0,
            grid_cell=cell,
            processing_level="NPC02",
        )

    monkeypatch.setattr(pnoa_lidar, "_discover_cnig_lidar_asset_at_point", fake_discovery)
    geometry: dict[str, object] = {
        "type": "Polygon",
        "coordinates": [[[left[0], left[1]], [right[0], right[1]], [left[0], left[1]]]],
    }
    with httpx.Client() as client:
        assets = discover_cnig_lidar_assets(client, geometry, buffer_m=25)

    assert [asset.grid_cell for asset in assets] == ["724-4373-H30", "725-4373-H30"]
