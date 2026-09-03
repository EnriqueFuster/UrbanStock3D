import pytest

from urbanstock3d.providers.catastro_parser import (
    CatastroParseError,
    parse_single_building,
)

BUILDING_XML = b"""\
<gml:FeatureCollection xmlns:gml="http://www.opengis.net/gml/3.2"
 xmlns:base="urn:x-inspire:specification:gmlas:BaseTypes:3.2"
 xmlns:bu-core2d="http://inspire.jrc.ec.europa.eu/schemas/bu-core2d/2.0"
 xmlns:bu-ext2d="http://inspire.jrc.ec.europa.eu/schemas/bu-ext2d/2.0">
 <gml:featureMember><bu-ext2d:Building>
  <bu-core2d:beginLifespanVersion>2020-12-18T00:00:00</bu-core2d:beginLifespanVersion>
  <bu-core2d:conditionOfConstruction>functional</bu-core2d:conditionOfConstruction>
  <bu-core2d:dateOfConstruction><bu-core2d:DateOfEvent>
   <bu-core2d:beginning>1952-01-01T00:00:00</bu-core2d:beginning>
   <bu-core2d:end>1952-01-01T00:00:00</bu-core2d:end>
  </bu-core2d:DateOfEvent></bu-core2d:dateOfConstruction>
  <bu-core2d:externalReference><bu-core2d:ExternalReference>
   <bu-core2d:reference>4531917YJ2743B</bu-core2d:reference>
  </bu-core2d:ExternalReference></bu-core2d:externalReference>
  <bu-core2d:inspireId><base:Identifier>
   <base:localId>4531917YJ2743B</base:localId><base:namespace>ES.SDGC.BU</base:namespace>
  </base:Identifier></bu-core2d:inspireId>
  <bu-ext2d:geometry><bu-core2d:BuildingGeometry><bu-core2d:geometry>
   <gml:Surface srsName="urn:ogc:def:crs:EPSG::25830"><gml:patches>
    <gml:PolygonPatch><gml:exterior><gml:LinearRing>
     <gml:posList srsDimension="2">0 0 1 0 1 1 0 0</gml:posList>
    </gml:LinearRing></gml:exterior></gml:PolygonPatch>
   </gml:patches></gml:Surface>
  </bu-core2d:geometry><bu-core2d:horizontalGeometryEstimatedAccuracy>0.1</bu-core2d:horizontalGeometryEstimatedAccuracy><bu-core2d:horizontalGeometryReference>footPrint</bu-core2d:horizontalGeometryReference></bu-core2d:BuildingGeometry></bu-ext2d:geometry>
  <bu-ext2d:currentUse>1_residential</bu-ext2d:currentUse>
  <bu-ext2d:numberOfBuildingUnits>17</bu-ext2d:numberOfBuildingUnits>
  <bu-ext2d:numberOfDwellings>16</bu-ext2d:numberOfDwellings>
  <bu-ext2d:officialArea><bu-ext2d:OfficialArea>
   <bu-ext2d:officialAreaReference>grossFloorArea</bu-ext2d:officialAreaReference>
   <bu-ext2d:value>2698</bu-ext2d:value>
  </bu-ext2d:OfficialArea></bu-ext2d:officialArea>
 </bu-ext2d:Building></gml:featureMember>
</gml:FeatureCollection>
"""

PARTS_XML = b"""\
<gml:FeatureCollection xmlns:gml="http://www.opengis.net/gml/3.2"
 xmlns:base="urn:x-inspire:specification:gmlas:BaseTypes:3.2"
 xmlns:bu-core2d="http://inspire.jrc.ec.europa.eu/schemas/bu-core2d/2.0"
 xmlns:bu-ext2d="http://inspire.jrc.ec.europa.eu/schemas/bu-ext2d/2.0">
 <gml:featureMember><bu-ext2d:BuildingPart>
  <bu-core2d:inspireId><base:Identifier><base:localId>4531917YJ2743B_part1</base:localId></base:Identifier></bu-core2d:inspireId>
  <bu-ext2d:geometry><bu-core2d:BuildingGeometry><bu-core2d:geometry>
   <gml:Surface srsName="EPSG:25830"><gml:patches><gml:PolygonPatch>
    <gml:exterior><gml:LinearRing>
     <gml:posList>0 0 1 0 1 1 0 0</gml:posList>
    </gml:LinearRing></gml:exterior>
   </gml:PolygonPatch></gml:patches></gml:Surface>
  </bu-core2d:geometry></bu-core2d:BuildingGeometry></bu-ext2d:geometry>
  <bu-ext2d:numberOfFloorsAboveGround>10</bu-ext2d:numberOfFloorsAboveGround>
  <bu-ext2d:heightBelowGround>0</bu-ext2d:heightBelowGround>
  <bu-ext2d:numberOfFloorsBelowGround>0</bu-ext2d:numberOfFloorsBelowGround>
 </bu-ext2d:BuildingPart></gml:featureMember>
</gml:FeatureCollection>
"""


def test_parse_single_building_maps_observed_fields() -> None:
    building = parse_single_building(BUILDING_XML, PARTS_XML)

    assert building.identity.building_id == "ES.SDGC.BU.4531917YJ2743B"
    assert building.current_use == "1_residential"
    assert building.dwellings == 16
    assert building.official_area is not None
    assert building.official_area.value_m2 == 2698
    assert building.footprint.crs == "EPSG:25830"
    assert building.parts[0].floors_above_ground == 10


def test_parse_single_building_rejects_empty_collection() -> None:
    empty = BUILDING_XML.replace(
        b"<gml:featureMember>",
        b"<!--",
    ).replace(
        b"</gml:featureMember>",
        b"-->",
    )

    with pytest.raises(CatastroParseError, match="Expected one building"):
        parse_single_building(empty, PARTS_XML)


def test_parser_rejects_odd_coordinate_count() -> None:
    invalid = BUILDING_XML.replace(b"0 0 1 0 1 1 0 0", b"0 0 1")

    with pytest.raises(CatastroParseError, match="odd coordinate count"):
        parse_single_building(invalid, PARTS_XML)
