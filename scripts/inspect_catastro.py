"""Inspect real Catastro responses before defining domain models."""

import argparse
from collections import Counter
from collections.abc import Mapping
from xml.etree import ElementTree

import httpx

from urbanstock3d.config import Settings

COORDINATE_ENDPOINT = (
    "https://ovc.catastro.meh.es/"
    "OVCServWeb/OVCWcfCallejero/"
    "COVCCoordenadas.svc/rest/Consulta_RCCOOR"
)
BUILDING_ENDPOINT = "https://ovc.catastro.meh.es/INSPIRE/wfsBU.aspx"
PREVIEW_LENGTH = 120


def parse_arguments() -> argparse.Namespace:
    """Read inspection parameters from the command line."""
    parser = argparse.ArgumentParser(
        description="Inspect Catastro responses without modelling them.",
    )
    parser.add_argument("--lon", required=True, type=float)
    parser.add_argument("--lat", required=True, type=float)
    parser.add_argument("--building-srs", default="EPSG:25830")
    parser.add_argument("--check-facade", action="store_true")
    parser.add_argument("--show-raw-xml", action="store_true")
    return parser.parse_args()


def local_name(tag: str) -> str:
    """Remove an XML namespace from a tag name."""
    return tag.rsplit("}", maxsplit=1)[-1]


def find_text(root: ElementTree.Element, tag_name: str) -> str:
    """Find the first non-empty value for an XML tag."""
    for element in root.iter():
        if local_name(element.tag) == tag_name and element.text:
            return element.text.strip()

    raise ValueError(f"XML tag not found: {tag_name}")


def find_texts(root: ElementTree.Element, tag_name: str) -> list[str]:
    """Find all non-empty values for an XML tag."""
    return [
        element.text.strip()
        for element in root.iter()
        if local_name(element.tag) == tag_name and element.text
    ]


def extract_cadastral_root(root: ElementTree.Element) -> str:
    """Combine the two Catastro parcel-reference fragments."""
    return find_text(root, "pc1") + find_text(root, "pc2")


def request_xml(
    client: httpx.Client,
    endpoint: str,
    params: Mapping[str, str],
) -> tuple[httpx.Response, ElementTree.Element]:
    """Request an XML resource and parse its root element."""
    response = client.get(endpoint, params=params)
    response.raise_for_status()
    return response, ElementTree.fromstring(response.content)


def print_response_metadata(label: str, response: httpx.Response) -> None:
    """Print metadata about an HTTP response."""
    print(f"\n--- {label} ---")
    print(f"Requested URL: {response.request.url}")
    print(f"HTTP status: {response.status_code}")
    print(f"Content-Type: {response.headers.get('content-type')}")
    print(f"Response bytes: {len(response.content)}")


def print_xml_summary(root: ElementTree.Element) -> None:
    """Print tag frequencies and representative values."""
    tag_counts = Counter(local_name(element.tag) for element in root.iter())

    print(f"Root tag: {local_name(root.tag)}")
    print("\nTag frequencies:")
    for tag, count in sorted(tag_counts.items()):
        print(f"  {tag}: {count}")

    print("\nNon-empty values:")
    for element in root.iter():
        text = element.text.strip() if element.text else ""
        if not text:
            continue

        tag = local_name(element.tag)
        if tag == "documentLink" or len(text) <= PREVIEW_LENGTH:
            preview = repr(text)
        else:
            preview = f"{text[:PREVIEW_LENGTH]!r}... [length={len(text)}]"
        print(f"  {tag} = {preview}")


def wfs_parameters(
    cadastral_root: str,
    stored_query: str,
    srs_name: str,
) -> dict[str, str]:
    """Build parameters shared by Catastro building WFS queries."""
    return {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "StoredQuerie_id": stored_query,
        "REFCAT": cadastral_root,
        "SRSNAME": srs_name,
    }


def inspect_facade(client: httpx.Client, building_root: ElementTree.Element) -> None:
    """Check the first facade document without persisting the image."""
    links = find_texts(building_root, "documentLink")
    if not links:
        print("\n--- Facade image ---")
        print("No documentLink returned by Catastro.")
        return

    response = client.get(links[0])
    response.raise_for_status()
    print_response_metadata("Facade image", response)


def main() -> None:
    """Request and inspect coordinate, building and building-part responses."""
    args = parse_arguments()
    settings = Settings()
    timeout = httpx.Timeout(
        settings.http_read_timeout_seconds,
        connect=settings.http_connect_timeout_seconds,
    )

    with httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "UrbanStock3D/0.1 data-audit"},
    ) as client:
        coordinate_response, coordinate_root = request_xml(
            client,
            COORDINATE_ENDPOINT,
            {
                "CoorX": str(args.lon),
                "CoorY": str(args.lat),
                "SRS": "EPSG:4326",
            },
        )
        cadastral_root = extract_cadastral_root(coordinate_root)

        building_response, building_root = request_xml(
            client,
            BUILDING_ENDPOINT,
            wfs_parameters(cadastral_root, "GetBuildingByParcel", args.building_srs),
        )
        parts_response, parts_root = request_xml(
            client,
            BUILDING_ENDPOINT,
            wfs_parameters(cadastral_root, "GetBuildingPartByParcel", args.building_srs),
        )

        print_response_metadata("Coordinate response", coordinate_response)
        print_xml_summary(coordinate_root)
        print(f"\nCadastral root: {cadastral_root}")

        print_response_metadata("BU.Building response", building_response)
        print_xml_summary(building_root)

        print_response_metadata("BU.BuildingPart response", parts_response)
        print_xml_summary(parts_root)

        if args.show_raw_xml:
            print("\n--- Raw BU.Building XML ---")
            print(building_response.text)
            print("\n--- Raw BU.BuildingPart XML ---")
            print(parts_response.text)

        if args.check_facade:
            inspect_facade(client, building_root)


if __name__ == "__main__":
    main()
