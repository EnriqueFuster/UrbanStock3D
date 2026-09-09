"""Prepare a cadastral building footprint for Roofer."""

import argparse
import json
from pathlib import Path
from typing import Any

from urbanstock3d.providers.pnoa_lidar import footprint_polygons_utm


def projected_feature_collection(building: dict[str, Any]) -> dict[str, Any]:
    """Transform one WGS84 building feature to an EPSG:25830 collection."""
    geometry = building.get("geometry", {})
    polygons = footprint_polygons_utm(geometry)
    building_id = str(building["id"])
    coordinates = [[[list(position) for position in ring] for ring in rings] for rings in polygons]
    output_geometry: dict[str, Any]
    if len(coordinates) == 1:
        output_geometry = {"type": "Polygon", "coordinates": coordinates[0]}
    else:
        output_geometry = {"type": "MultiPolygon", "coordinates": coordinates}
    return {
        "type": "FeatureCollection",
        "name": "roofer_footprint",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:EPSG::25830"},
        },
        "features": [
            {
                "type": "Feature",
                "id": building_id,
                "properties": {"building_id": building_id},
                "geometry": output_geometry,
            }
        ],
    }


def prepare_roofer_input(source: Path, destination: Path) -> Path:
    """Write a projected GeoJSON footprint ready for Roofer."""
    building: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    collection = projected_feature_collection(building)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(collection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def parse_arguments() -> argparse.Namespace:
    """Read input and output paths."""
    parser = argparse.ArgumentParser(description="Prepare an EPSG:25830 Roofer footprint.")
    parser.add_argument("building_geojson", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    """Run Roofer input preparation."""
    args = parse_arguments()
    output = prepare_roofer_input(args.building_geojson, args.output)
    print(f"Wrote Roofer footprint to {output}")


if __name__ == "__main__":
    main()
