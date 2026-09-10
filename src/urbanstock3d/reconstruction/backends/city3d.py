"""Input compatibility checks for the research-grade City3D backend."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import laspy


@dataclass(frozen=True)
class City3DInputAssessment:
    """Evidence that existing project artifacts can be consumed by City3D."""

    point_cloud: Path
    footprint: Path
    point_count: int
    epsg: int
    footprint_vertex_count: int

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["point_cloud"] = str(self.point_cloud)
        payload["footprint"] = str(self.footprint)
        return payload


def assess_city3d_inputs(point_cloud: Path, footprint: Path) -> City3DInputAssessment:
    """Validate the narrow City3D input subset used by UrbanStock3D."""
    if point_cloud.suffix.lower() not in {".las", ".laz"}:
        raise ValueError("City3D point cloud must be LAS or LAZ")
    if not point_cloud.is_file():
        raise FileNotFoundError(point_cloud)
    if not footprint.is_file():
        raise FileNotFoundError(footprint)

    with laspy.open(point_cloud) as reader:
        point_count = reader.header.point_count
        crs = reader.header.parse_crs()
    if point_count == 0:
        raise ValueError("City3D point cloud cannot be empty")
    epsg = crs.to_epsg() if crs is not None else None
    if epsg != 25830:
        raise ValueError("City3D inputs must use EPSG:25830")

    collection: dict[str, Any] = json.loads(footprint.read_text(encoding="utf-8"))
    crs_name = str(collection.get("crs", {}).get("properties", {}).get("name", ""))
    if not crs_name.endswith("25830"):
        raise ValueError("City3D footprint must declare EPSG:25830")
    features = collection.get("features", [])
    if len(features) != 1:
        raise ValueError("City3D input must contain exactly one footprint feature")
    geometry = features[0].get("geometry", {})
    if geometry.get("type") != "Polygon":
        raise ValueError("City3D integration currently supports one Polygon footprint")
    rings = geometry.get("coordinates", [])
    if len(rings) != 1:
        raise ValueError("City3D integration currently rejects footprint holes")
    exterior = rings[0]
    if len(exterior) < 4 or exterior[0] != exterior[-1]:
        raise ValueError("City3D footprint exterior must be a closed polygon")

    return City3DInputAssessment(
        point_cloud=point_cloud,
        footprint=footprint,
        point_count=point_count,
        epsg=epsg,
        footprint_vertex_count=len(exterior) - 1,
    )
