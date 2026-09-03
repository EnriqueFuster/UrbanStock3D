from xml.etree import ElementTree

from scripts.inspect_catastro import (
    extract_cadastral_root,
    find_texts,
    local_name,
    wfs_parameters,
)


def test_local_name_removes_xml_namespace() -> None:
    assert local_name("{https://example.test/schema}Building") == "Building"
    assert local_name("Building") == "Building"


def test_extract_cadastral_root_combines_response_fragments() -> None:
    root = ElementTree.fromstring(
        "<consulta_coordenadas><pc><pc1>4531917</pc1><pc2>YJ2743B</pc2></pc></consulta_coordenadas>"
    )

    assert extract_cadastral_root(root) == "4531917YJ2743B"


def test_find_texts_returns_all_matching_values() -> None:
    root = ElementTree.fromstring(
        "<FeatureCollection><documentLink>first</documentLink>"
        "<documentLink>second</documentLink></FeatureCollection>"
    )

    assert find_texts(root, "documentLink") == ["first", "second"]


def test_wfs_parameters_keep_query_specific_values_explicit() -> None:
    params = wfs_parameters(
        "4531917YJ2743B",
        "GetBuildingPartByParcel",
        "EPSG:25830",
    )

    assert params["StoredQuerie_id"] == "GetBuildingPartByParcel"
    assert params["REFCAT"] == "4531917YJ2743B"
    assert params["SRSNAME"] == "EPSG:25830"
