"""Convert RID2 geospatial polygons into canonical YOLO segmentation labels."""

import csv
import io
import json
import shutil
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import yaml
from pydantic import BaseModel, ConfigDict
from shapely import make_valid  # type: ignore[import-untyped]
from shapely.geometry import Polygon, box, shape  # type: ignore[import-untyped]
from shapely.geometry.base import BaseGeometry  # type: ignore[import-untyped]
from shapely.strtree import STRtree  # type: ignore[import-untyped]

from urbanstock3d.vision.common.taxonomy import CanonicalTaxonomy
from urbanstock3d.vision.roof_objects.datasets import Rid2SourceConfig


class Rid2ConversionManifest(BaseModel):
    """Auditable summary of one canonical conversion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_version: str
    source_record_url: str
    archive_md5: str
    source_crs: str
    taxonomy_version: str
    licence: str | None
    train_images: int
    validation_images: int
    instance_counts: dict[str, int]
    ignored_instance_counts: dict[str, int]


def _read_json(archive: ZipFile, member: str) -> dict[str, object]:
    payload = json.loads(archive.read(member))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in {member}")
    return payload


def _split_names(archive: ZipFile, member: str) -> set[str]:
    rows = csv.DictReader(io.StringIO(archive.read(member).decode("utf-8-sig")))
    return {row["image_names"] for row in rows}


def _polygons(geometry: BaseGeometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    polygons: list[Polygon] = []
    for part in getattr(geometry, "geoms", ()):
        polygons.extend(_polygons(part))
    return polygons


def _yolo_ring(polygon: Polygon, bounds: tuple[float, float, float, float]) -> str | None:
    min_x, min_y, max_x, max_y = bounds
    width = max_x - min_x
    height = max_y - min_y
    coordinates = list(polygon.exterior.coords)[:-1]
    if width <= 0 or height <= 0 or len(coordinates) < 3:
        return None
    values: list[str] = []
    for x, y in coordinates:
        values.extend((f"{(x - min_x) / width:.6f}", f"{(max_y - y) / height:.6f}"))
    return " ".join(values)


def convert_rid2_archive(
    archive_path: Path,
    output_root: Path,
    source: Rid2SourceConfig,
    taxonomy: CanonicalTaxonomy,
    *,
    limit: int | None = None,
) -> Rid2ConversionManifest:
    """Convert reviewed RID2 classes and preserve the official train/test split."""
    label_mapping = {
        mapping.source_label: mapping.canonical for mapping in source.source_classes.values()
    }
    canonical_ids = taxonomy.roof_objects
    instance_counts: Counter[str] = Counter()
    ignored_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()

    with ZipFile(archive_path) as archive:
        train_names = _split_names(archive, "training_split_512.csv")
        validation_names = _split_names(archive, "test_split_512.csv")
        image_collection = _read_json(archive, "geometries/gdf_images_with_labels_512.json")
        object_collection = _read_json(archive, "geometries/gdf_all_superstructures.json")
        image_features = image_collection["features"]
        object_features = object_collection["features"]
        if not isinstance(image_features, list) or not isinstance(object_features, list):
            raise ValueError("RID2 GeoJSON collections must contain feature lists")

        object_geometries = [make_valid(shape(feature["geometry"])) for feature in object_features]
        tree = STRtree(object_geometries)
        selected_images = image_features[:limit] if limit is not None else image_features

        for image_feature in selected_images:
            properties = image_feature["properties"]
            image_name = f"{properties['id']}.png"
            split = "train" if image_name in train_names else "val"
            if image_name not in train_names and image_name not in validation_names:
                raise ValueError(f"Image is absent from official splits: {image_name}")

            tile = shape(image_feature["geometry"])
            tile_bounds = tile.bounds
            label_lines: list[str] = []
            for index in tree.query(tile, predicate="intersects"):
                feature = object_features[int(index)]
                source_label = str(feature["properties"]["label"])
                canonical = label_mapping.get(source_label)
                if canonical is None:
                    ignored_counts[source_label] += 1
                    continue
                clipped = object_geometries[int(index)].intersection(box(*tile_bounds))
                for polygon in _polygons(clipped):
                    ring = _yolo_ring(polygon, tile_bounds)
                    if ring is not None:
                        label_lines.append(f"{canonical_ids[canonical]} {ring}")
                        instance_counts[canonical] += 1

            image_output = output_root / "images" / split / image_name
            label_output = output_root / "labels" / split / f"{properties['id']}.txt"
            image_output.parent.mkdir(parents=True, exist_ok=True)
            label_output.parent.mkdir(parents=True, exist_ok=True)
            with (
                archive.open(f"images/{image_name}") as source_image,
                image_output.open("wb") as out,
            ):
                shutil.copyfileobj(source_image, out)
            label_output.write_text(
                "\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8"
            )
            split_counts[split] += 1

    manifest = Rid2ConversionManifest(
        source_version=source.source_version,
        source_record_url=str(source.record_url),
        archive_md5=source.archive_md5,
        source_crs="EPSG:28992",
        taxonomy_version=taxonomy.version,
        licence=source.licence,
        train_images=split_counts["train"],
        validation_images=split_counts["val"],
        instance_counts=dict(sorted(instance_counts.items())),
        ignored_instance_counts=dict(sorted(ignored_counts.items())),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    dataset_config = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "names": {identifier: name for name, identifier in taxonomy.roof_objects.items()},
    }
    (output_root / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_config, sort_keys=False), encoding="utf-8"
    )
    return manifest
