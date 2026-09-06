import pytest

from urbanstock3d.providers.pnoa_lidar import (
    LidarGridCell,
    cnig_search_url,
    parse_cnig_asset_page,
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
