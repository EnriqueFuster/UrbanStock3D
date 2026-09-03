"""HTTP adapter for public Catastro services."""

import httpx

from urbanstock3d.config import Settings
from urbanstock3d.domain.cadastre import (
    CadastralBuilding,
    CadastralRootId,
    CoordinateQuery,
    CoordinateResolution,
    RefcatQuery,
)
from urbanstock3d.errors import (
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from urbanstock3d.providers.catastro_parser import (
    parse_coordinate_resolution,
    parse_single_building,
)


def create_http_client(settings: Settings) -> httpx.Client:
    """Create the shared HTTP client configuration for Catastro."""
    return httpx.Client(
        timeout=httpx.Timeout(
            settings.http_read_timeout_seconds,
            connect=settings.http_connect_timeout_seconds,
        ),
        follow_redirects=True,
        headers={"User-Agent": "UrbanStock3D/0.1"},
    )


class CatastroProvider:
    """Retrieve public cadastral identity and building information."""

    def __init__(self, client: httpx.Client, settings: Settings) -> None:
        self._client = client
        self._coordinate_url = str(settings.catastro_coordinate_url)
        self._building_wfs_url = str(settings.catastro_building_wfs_url)

    def resolve_coordinate(self, query: CoordinateQuery) -> CoordinateResolution:
        """Resolve a WGS84 coordinate to its cadastral parcel root."""
        content = self._get(
            self._coordinate_url,
            params={
                "CoorX": str(query.longitude),
                "CoorY": str(query.latitude),
                "SRS": "EPSG:4326",
            },
        )
        return parse_coordinate_resolution(content, query)

    def resolve_building(
        self,
        query: CoordinateQuery | RefcatQuery,
        *,
        srs_name: str = "EPSG:25830",
    ) -> CadastralBuilding:
        """Resolve user input and retrieve its cadastral building."""
        if isinstance(query, CoordinateQuery):
            cadastral_root_id = self.resolve_coordinate(query).cadastral_root_id
        else:
            cadastral_root_id = query.cadastral_root_id
        return self.fetch_building(cadastral_root_id, srs_name=srs_name)

    def fetch_building(
        self,
        cadastral_root_id: CadastralRootId,
        *,
        srs_name: str = "EPSG:25830",
    ) -> CadastralBuilding:
        """Fetch one building and its parts for a cadastral parcel root."""
        building_xml = self._get(
            self._building_wfs_url,
            params=_wfs_parameters(cadastral_root_id, "GetBuildingByParcel", srs_name),
        )
        parts_xml = self._get(
            self._building_wfs_url,
            params=_wfs_parameters(cadastral_root_id, "GetBuildingPartByParcel", srs_name),
        )
        return parse_single_building(building_xml, parts_xml)

    def _get(self, url: str, *, params: dict[str, str]) -> bytes:
        try:
            response = self._client.get(url, params=params)
            response.raise_for_status()
        except httpx.TimeoutException as error:
            raise ProviderTimeoutError(f"Catastro request timed out: {url}") from error
        except httpx.HTTPStatusError as error:
            raise ProviderResponseError(
                f"Catastro returned HTTP {error.response.status_code}: {url}"
            ) from error
        except httpx.RequestError as error:
            raise ProviderUnavailableError(f"Catastro request failed: {url}") from error
        return response.content


def _wfs_parameters(
    cadastral_root_id: str,
    stored_query: str,
    srs_name: str,
) -> dict[str, str]:
    return {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "StoredQuerie_id": stored_query,
        "REFCAT": cadastral_root_id,
        "SRSNAME": srs_name,
    }
