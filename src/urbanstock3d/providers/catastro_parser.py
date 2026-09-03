"""Parse Catastro INSPIRE Building GML into domain models."""

import re
from datetime import date, datetime
from xml.etree import ElementTree

from pydantic import AnyHttpUrl

from urbanstock3d.domain.cadastre import (
    CadastralBuilding,
    CadastralBuildingPart,
    CadastralIdentity,
    ConstructionPeriod,
    CoordinateQuery,
    CoordinateResolution,
    FacadeDocument,
    Footprint2D,
    OfficialArea,
)

NAMESPACES = {
    "base": "urn:x-inspire:specification:gmlas:BaseTypes:3.2",
    "bu-core2d": "http://inspire.jrc.ec.europa.eu/schemas/bu-core2d/2.0",
    "bu-ext2d": "http://inspire.jrc.ec.europa.eu/schemas/bu-ext2d/2.0",
    "gml": "http://www.opengis.net/gml/3.2",
}


class CatastroParseError(ValueError):
    """Raised when a Catastro response violates the expected contract."""


def parse_coordinate_resolution(
    coordinate_xml: bytes,
    query: CoordinateQuery,
) -> CoordinateResolution:
    """Parse a coordinate-service response into a cadastral resolution."""
    root = _parse_xml(coordinate_xml)
    error_count = _optional_int(root, ".//cuerr")
    if error_count is None:
        raise CatastroParseError("Coordinate response has no error count")
    if error_count:
        descriptions = [
            child.text.strip()
            for child in root.iter()
            if child.tag.endswith("des") and child.text and child.text.strip()
        ]
        detail = "; ".join(descriptions) or "unspecified Catastro error"
        raise CatastroParseError(f"Coordinate resolution failed: {detail}")

    return CoordinateResolution(
        query=query,
        cadastral_root_id=_required_text(root, ".//pc1") + _required_text(root, ".//pc2"),
        address=_optional_text(root, ".//ldt"),
    )


def parse_single_building(
    building_xml: bytes,
    building_parts_xml: bytes,
) -> CadastralBuilding:
    """Parse one building and its parts from two WFS responses."""
    building_root = _parse_xml(building_xml)
    elements = building_root.findall("gml:featureMember/bu-ext2d:Building", NAMESPACES)
    if len(elements) != 1:
        raise CatastroParseError(f"Expected one building, received {len(elements)}")

    element = elements[0]
    identity = _parse_identity(element)
    parts = parse_building_parts(building_parts_xml)

    return CadastralBuilding(
        identity=identity,
        footprint=_parse_footprint(element),
        parts=parts,
        condition=_optional_text(element, "bu-core2d:conditionOfConstruction"),
        current_use=_optional_text(element, "bu-ext2d:currentUse"),
        construction_period=_parse_construction_period(element),
        building_units=_optional_int(element, "bu-ext2d:numberOfBuildingUnits"),
        dwellings=_optional_int(element, "bu-ext2d:numberOfDwellings"),
        floors_above_ground=_optional_int(element, "bu-ext2d:numberOfFloorsAboveGround"),
        official_area=_parse_official_area(element),
        facade_document=_parse_facade_document(element),
        begin_lifespan_version=_optional_datetime(
            element,
            "bu-core2d:beginLifespanVersion",
        ),
        end_lifespan_version=_optional_datetime(
            element,
            "bu-core2d:endLifespanVersion",
        ),
    )


def parse_building_parts(building_parts_xml: bytes) -> tuple[CadastralBuildingPart, ...]:
    """Parse every BuildingPart in a WFS feature collection."""
    root = _parse_xml(building_parts_xml)
    elements = root.findall("gml:featureMember/bu-ext2d:BuildingPart", NAMESPACES)
    return tuple(
        CadastralBuildingPart(
            local_id=_required_text(element, ".//base:localId"),
            floors_above_ground=_optional_int(
                element,
                "bu-ext2d:numberOfFloorsAboveGround",
            ),
            floors_below_ground=_optional_int(
                element,
                "bu-ext2d:numberOfFloorsBelowGround",
            ),
            height_below_ground_m=_optional_float(
                element,
                "bu-ext2d:heightBelowGround",
            ),
            footprint=_parse_footprint(element),
        )
        for element in elements
    )


def _parse_xml(content: bytes) -> ElementTree.Element:
    try:
        return ElementTree.fromstring(content)
    except ElementTree.ParseError as error:
        raise CatastroParseError("Catastro returned invalid XML") from error


def _parse_identity(element: ElementTree.Element) -> CadastralIdentity:
    return CadastralIdentity(
        cadastral_root_id=_required_text(
            element,
            ".//bu-core2d:externalReference/bu-core2d:ExternalReference/bu-core2d:reference",
        ),
        local_id=_required_text(element, ".//base:localId"),
        namespace=_required_text(element, ".//base:namespace"),
    )


def _parse_footprint(element: ElementTree.Element) -> Footprint2D:
    surface = element.find(".//gml:Surface", NAMESPACES)
    if surface is None:
        raise CatastroParseError("Building geometry has no GML Surface")

    crs = _normalize_crs(surface.get("srsName"))
    rings = tuple(
        _parse_positions(pos_list)
        for pos_list in surface.findall(".//gml:LinearRing/gml:posList", NAMESPACES)
    )
    if not rings:
        raise CatastroParseError("Building geometry has no linear rings")

    return Footprint2D(
        crs=crs,
        rings=rings,
        estimated_accuracy_m=_optional_float(
            element,
            ".//bu-core2d:horizontalGeometryEstimatedAccuracy",
        ),
        reference=_optional_text(
            element,
            ".//bu-core2d:horizontalGeometryReference",
        ),
    )


def _parse_positions(element: ElementTree.Element) -> tuple[tuple[float, float], ...]:
    dimension = element.get("srsDimension", "2")
    if dimension != "2":
        raise CatastroParseError(f"Expected 2D coordinates, received {dimension}D")

    values = [float(value) for value in (element.text or "").split()]
    if len(values) % 2:
        raise CatastroParseError("GML posList contains an odd coordinate count")
    return tuple(zip(values[::2], values[1::2], strict=True))


def _normalize_crs(value: str | None) -> str:
    if value is None:
        raise CatastroParseError("GML Surface has no srsName")
    match = re.fullmatch(r"(?:urn:ogc:def:crs:)?EPSG(?:::|:)(\d+)", value)
    if match is None:
        raise CatastroParseError(f"Unsupported CRS identifier: {value}")
    return f"EPSG:{match.group(1)}"


def _parse_construction_period(element: ElementTree.Element) -> ConstructionPeriod | None:
    beginning = _optional_date(element, ".//bu-core2d:dateOfConstruction//bu-core2d:beginning")
    end = _optional_date(element, ".//bu-core2d:dateOfConstruction//bu-core2d:end")
    if beginning is None and end is None:
        return None
    if beginning is None or end is None:
        raise CatastroParseError("Construction period is incomplete")
    return ConstructionPeriod(beginning=beginning, end=end)


def _parse_official_area(element: ElementTree.Element) -> OfficialArea | None:
    area = element.find("bu-ext2d:officialArea/bu-ext2d:OfficialArea", NAMESPACES)
    if area is None:
        return None
    return OfficialArea(
        value_m2=float(_required_text(area, "bu-ext2d:value")),
        reference=_required_text(area, "bu-ext2d:officialAreaReference"),
    )


def _parse_facade_document(element: ElementTree.Element) -> FacadeDocument | None:
    document = element.find("bu-ext2d:document/bu-ext2d:Document", NAMESPACES)
    if document is None:
        return None
    return FacadeDocument(
        url=AnyHttpUrl(_required_text(document, "bu-ext2d:documentLink")),
        media_format=_required_text(document, "bu-ext2d:format"),
        source_status=_optional_text(document, "bu-ext2d:sourceStatus"),
    )


def _required_text(element: ElementTree.Element, path: str) -> str:
    value = _optional_text(element, path)
    if value is None:
        raise CatastroParseError(f"Required Catastro field is missing: {path}")
    return value


def _optional_text(element: ElementTree.Element, path: str) -> str | None:
    child = element.find(path, NAMESPACES)
    if child is None or child.text is None or not child.text.strip():
        return None
    return child.text.strip()


def _optional_int(element: ElementTree.Element, path: str) -> int | None:
    value = _optional_text(element, path)
    return int(value) if value is not None else None


def _optional_float(element: ElementTree.Element, path: str) -> float | None:
    value = _optional_text(element, path)
    return float(value) if value is not None else None


def _optional_date(element: ElementTree.Element, path: str) -> date | None:
    value = _optional_text(element, path)
    return datetime.fromisoformat(value).date() if value is not None else None


def _optional_datetime(element: ElementTree.Element, path: str) -> datetime | None:
    value = _optional_text(element, path)
    return datetime.fromisoformat(value) if value is not None else None
