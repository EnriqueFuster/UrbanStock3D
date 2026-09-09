"""Read Roofer CityJSONSeq and export its polygonal geometry."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CityJsonSurface:
    """One semantic CityJSON surface represented by vertex-index rings."""

    semantic_type: str
    rings: tuple[tuple[int, ...], ...]


def read_cityjsonseq(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the metadata and single feature records produced by Roofer."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    metadata = next((record for record in records if record.get("type") == "CityJSON"), None)
    feature = next((record for record in records if record.get("type") == "CityJSONFeature"), None)
    if metadata is None or feature is None:
        raise ValueError("CityJSONSeq must contain metadata and feature records")
    return metadata, feature


def transformed_vertices(
    metadata: dict[str, Any], feature: dict[str, Any]
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Decode quantized CityJSON vertices into projected coordinates."""
    transform = metadata["transform"]
    vertices = np.asarray(feature["vertices"], dtype=np.float64)
    return vertices * np.asarray(transform["scale"]) + np.asarray(transform["translate"])


def lod_surfaces(feature: dict[str, Any], lod: str = "2.2") -> tuple[CityJsonSurface, ...]:
    """Extract semantic surfaces for a requested LoD from all building parts."""
    result: list[CityJsonSurface] = []
    for city_object in feature["CityObjects"].values():
        for geometry in city_object.get("geometry", []):
            if str(geometry.get("lod")) != lod or geometry.get("type") != "Solid":
                continue
            semantic_types = geometry["semantics"]["surfaces"]
            for shell, semantic_values in zip(
                geometry["boundaries"], geometry["semantics"]["values"], strict=True
            ):
                for boundary, semantic_index in zip(shell, semantic_values, strict=True):
                    result.append(
                        CityJsonSurface(
                            semantic_type=semantic_types[semantic_index]["type"],
                            rings=tuple(tuple(ring) for ring in boundary),
                        )
                    )
    if not result:
        raise ValueError(f"CityJSON feature contains no LoD {lod} solid surfaces")
    return tuple(result)


def write_obj(
    metadata: dict[str, Any],
    feature: dict[str, Any],
    destination: Path,
    *,
    lod: str = "2.2",
) -> tuple[Path, Path]:
    """Export CityJSON surfaces to an OBJ and companion material file."""
    vertices = transformed_vertices(metadata, feature)
    surfaces = lod_surfaces(feature, lod)
    if any(len(surface.rings) != 1 for surface in surfaces):
        raise ValueError("OBJ export does not yet support surface interior rings")

    destination.parent.mkdir(parents=True, exist_ok=True)
    material_path = destination.with_suffix(".mtl")
    lines = [f"mtllib {material_path.name}", f"o {feature.get('id', 'building')}"]
    lines.extend(f"v {x:.3f} {y:.3f} {z:.3f}" for x, y, z in vertices)
    current_material = None
    for surface in surfaces:
        if surface.semantic_type != current_material:
            current_material = surface.semantic_type
            lines.extend((f"g {current_material}", f"usemtl {current_material}"))
        lines.append("f " + " ".join(str(index + 1) for index in surface.rings[0]))
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    material_path.write_text(
        "newmtl RoofSurface\nKd 0.839 0.153 0.157\n\n"
        "newmtl WallSurface\nKd 0.498 0.498 0.498\n\n"
        "newmtl GroundSurface\nKd 0.549 0.337 0.294\n",
        encoding="utf-8",
    )
    return destination, material_path
