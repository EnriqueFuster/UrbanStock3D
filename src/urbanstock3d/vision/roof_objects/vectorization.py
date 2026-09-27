"""Convert roof-object masks into footprint-clipped map geometries."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize, shapes
from shapely import make_valid  # type: ignore[import-untyped]
from shapely.geometry import MultiPolygon, mapping, shape  # type: ignore[import-untyped]
from shapely.geometry.base import BaseGeometry  # type: ignore[import-untyped]
from shapely.ops import transform, unary_union  # type: ignore[import-untyped]

from urbanstock3d.vision.models import GeoPolygon, RoofObjectInstance, RoofVisionResult


def _projected_footprint(building_geojson: Path, target_crs: str) -> BaseGeometry:
    feature: dict[str, Any] = json.loads(building_geojson.read_text(encoding="utf-8"))
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("Building GeoJSON must contain a geometry")
    footprint = make_valid(shape(geometry))
    projector = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    projected = make_valid(transform(projector.transform, footprint))
    if projected.is_empty:
        raise ValueError("Projected building footprint is empty")
    return projected


def _largest_polygon(geometry: BaseGeometry) -> tuple[BaseGeometry, bool]:
    if geometry.geom_type == "Polygon":
        return geometry, False
    if isinstance(geometry, MultiPolygon):
        return max(geometry.geoms, key=lambda polygon: polygon.area), len(geometry.geoms) > 1
    raise ValueError(f"Unsupported clipped geometry: {geometry.geom_type}")


def vectorize_and_clip_roof_objects(
    result: RoofVisionResult,
    building_geojson: Path,
    output_dir: Path,
) -> RoofVisionResult:
    """Clip every predicted mask to the cadastral footprint and attach map polygons."""
    output_dir.mkdir(parents=True, exist_ok=True)
    vectorized: list[RoofObjectInstance] = []
    geojson_features: list[dict[str, Any]] = []
    output_crs: str | None = None

    for instance in result.objects:
        mask_path = Path(instance.mask_artifact)
        with rasterio.open(mask_path) as raster:
            if raster.crs is None:
                raise ValueError(f"Mask is not georeferenced: {mask_path}")
            crs_name = raster.crs.to_string()
            if output_crs is not None and output_crs != crs_name:
                raise ValueError("All roof-object masks must use the same CRS")
            output_crs = crs_name
            mask = raster.read(1) > 0
            footprint = _projected_footprint(building_geojson, crs_name)
            footprint_mask = rasterize(
                [(mapping(footprint), 1)],
                out_shape=mask.shape,
                transform=raster.transform,
                fill=0,
                dtype="uint8",
            ).astype(bool)
            clipped_mask = mask & footprint_mask
            pixel_count = int(clipped_mask.sum())
            if pixel_count == 0:
                continue
            mask_geometries = [
                shape(geometry)
                for geometry, value in shapes(
                    clipped_mask.astype(np.uint8),
                    mask=clipped_mask,
                    transform=raster.transform,
                )
                if value == 1
            ]
            clipped_geometry = make_valid(unary_union(mask_geometries).intersection(footprint))
            polygon, dropped_components = _largest_polygon(clipped_geometry)

            clipped_mask_path = output_dir / f"{instance.object_id}.tif"
            profile = raster.profile.copy()
            profile.update(count=1, dtype="uint8", compress="deflate")
            with rasterio.open(clipped_mask_path, "w", **profile) as destination:
                destination.write(clipped_mask.astype(np.uint8), 1)

        coordinates = tuple(
            tuple((float(x), float(y)) for x, y in ring)
            for ring in [
                polygon.exterior.coords,
                *(interior.coords for interior in polygon.interiors),
            ]
        )
        flags = [flag for flag in instance.quality_flags if flag != "NOT_CLIPPED_TO_FOOTPRINT"]
        if dropped_components:
            flags.append("DISCONNECTED_COMPONENTS_REDUCED_TO_LARGEST")
        updated = instance.model_copy(
            update={
                "mask_artifact": clipped_mask_path.as_posix(),
                "geometry_map": GeoPolygon(
                    coordinates=coordinates,
                    crs=crs_name,
                ),
                "pixel_area": pixel_count,
                "area_m2": float(polygon.area),
                "quality_flags": tuple(flags),
            }
        )
        vectorized.append(updated)
        geojson_features.append(
            {
                "type": "Feature",
                "id": updated.object_id,
                "geometry": mapping(polygon),
                "properties": {
                    "class_name": updated.class_name,
                    "confidence": updated.confidence,
                    "area_m2": updated.area_m2,
                },
            }
        )

    (output_dir / "objects.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "name": "roof_objects",
                "crs": {
                    "type": "name",
                    "properties": {"name": output_crs or "EPSG:25830"},
                },
                "features": geojson_features,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result.model_copy(update={"objects": tuple(vectorized)})
