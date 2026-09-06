from io import BytesIO

import pytest
from PIL import Image

from scripts.audit_facade import facade_url, inspect_image


def test_facade_url_reads_exported_building_metadata() -> None:
    geojson = {"properties": {"facade_document": {"url": "https://example.test/facade.jpg"}}}

    assert facade_url(geojson) == "https://example.test/facade.jpg"


def test_facade_url_rejects_missing_document() -> None:
    with pytest.raises(ValueError, match="no advertised facade"):
        facade_url({"properties": {}})


def test_inspect_image_decodes_image_payload() -> None:
    payload = BytesIO()
    Image.new("RGB", (12, 8), color=(10, 20, 30)).save(payload, format="JPEG")

    evidence = inspect_image(payload.getvalue())

    assert evidence["width"] == 12
    assert evidence["height"] == 8
    assert evidence["format"] == "JPEG"


def test_inspect_image_rejects_non_image_payload() -> None:
    with pytest.raises(ValueError, match="not a valid image"):
        inspect_image(b"not an image")
