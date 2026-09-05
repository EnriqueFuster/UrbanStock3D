"""Write inspectable cadastral JSON and GeoJSON artifacts."""

import json
import re
from pathlib import Path
from typing import Any, cast

from pyproj import Transformer

from urbanstock3d.domain.cadastre import CadastralBuilding, CadastralBuildingPart, Footprint2D

WGS84 = "EPSG:4326"


def write_cadastral_artifacts(
    building: CadastralBuilding,
    output_root: Path,
) -> Path:
    """Write identity and WGS84 geometry artifacts for one building."""
    building_dir = output_root / _safe_directory_name(building.identity.building_id)
    building_dir.mkdir(parents=True, exist_ok=True)

    identity = building.identity.model_dump(mode="json")
    identity["building_id"] = building.identity.building_id
    _write_json(building_dir / "identity.json", identity)
    _write_json(building_dir / "building.geojson", _building_feature(building))
    _write_json(
        building_dir / "building_parts.geojson",
        {
            "type": "FeatureCollection",
            "features": [_part_feature(part) for part in building.parts],
        },
    )
    return building_dir


def _building_feature(building: CadastralBuilding) -> dict[str, Any]:
    properties = building.model_dump(
        mode="json",
        exclude={"footprint", "parts"},
    )
    properties["building_id"] = building.identity.building_id
    return {
        "type": "Feature",
        "id": building.identity.building_id,
        "geometry": _polygon_geometry(building.footprint),
        "properties": properties,
    }


def _part_feature(part: CadastralBuildingPart) -> dict[str, Any]:
    properties = part.model_dump(mode="json", exclude={"footprint"})
    return {
        "type": "Feature",
        "id": part.local_id,
        "geometry": _polygon_geometry(part.footprint),
        "properties": properties,
    }


def _polygon_geometry(footprint: Footprint2D) -> dict[str, Any]:
    transformer = Transformer.from_crs(footprint.crs, WGS84, always_xy=True)
    coordinates: list[list[list[float]]] = []
    for ring in footprint.rings:
        transformed_ring: list[list[float]] = []
        for x, y in ring:
            longitude, latitude = cast(tuple[float, float], transformer.transform(x, y))
            transformed_ring.append([longitude, latitude])
        coordinates.append(transformed_ring)
    return {"type": "Polygon", "coordinates": coordinates}


def _safe_directory_name(building_id: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]", "_", building_id)
    if name in {"", ".", ".."}:
        raise ValueError("Building identifier cannot form a safe output directory")
    return name


def _write_json(path: Path, content: object) -> None:
    path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
