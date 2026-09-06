import httpx
import pytest

from urbanstock3d.config import Settings
from urbanstock3d.domain.cadastre import CoordinateQuery
from urbanstock3d.errors import ProviderResponseError, ProviderTimeoutError
from urbanstock3d.providers.catastro import CatastroProvider

COORDINATE_XML = b"""
<consulta_coordenadas xmlns="http://www.catastro.meh.es/">
  <control><cucoor>1</cucoor><cuerr>0</cuerr></control>
  <coordenadas><coord><pc><pc1>4531917</pc1><pc2>YJ2743B</pc2></pc>
  <ldt>PS PECHINA 31 VALENCIA (VALENCIA)</ldt></coord></coordenadas>
</consulta_coordenadas>
"""


def test_resolve_coordinate_uses_wgs84_and_parses_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["SRS"] == "EPSG:4326"
        assert request.url.params["CoorX"] == "-0.3911611691988909"
        return httpx.Response(200, content=COORDINATE_XML, request=request)

    settings = Settings()
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = CatastroProvider(client, settings)
        result = provider.resolve_coordinate(
            CoordinateQuery(longitude=-0.3911611691988909, latitude=39.47695517618966)
        )

    assert result.cadastral_root_id == "4531917YJ2743B"
    assert result.address == "PS PECHINA 31 VALENCIA (VALENCIA)"


def test_provider_translates_http_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = CatastroProvider(client, Settings())
        with pytest.raises(ProviderResponseError, match="HTTP 503"):
            provider.resolve_coordinate(CoordinateQuery(longitude=0, latitude=0))


def test_provider_translates_timeouts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = CatastroProvider(client, Settings())
        with pytest.raises(ProviderTimeoutError):
            provider.resolve_coordinate(CoordinateQuery(longitude=0, latitude=0))
