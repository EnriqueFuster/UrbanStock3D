from pathlib import Path

from urbanstock3d.formats.cityjson import lod_surfaces, transformed_vertices, write_obj


def sample_records() -> tuple[dict[str, object], dict[str, object]]:
    metadata: dict[str, object] = {
        "transform": {"scale": [0.1, 0.1, 0.1], "translate": [100.0, 200.0, 0.0]}
    }
    feature: dict[str, object] = {
        "id": "building-1",
        "vertices": [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]],
        "CityObjects": {
            "part": {
                "geometry": [
                    {
                        "type": "Solid",
                        "lod": "2.2",
                        "boundaries": [[[[0, 1, 2, 3]]]],
                        "semantics": {
                            "surfaces": [{"type": "RoofSurface"}],
                            "values": [[0]],
                        },
                    }
                ]
            }
        },
    }
    return metadata, feature


def test_extracts_and_transforms_cityjson_surfaces() -> None:
    metadata, feature = sample_records()

    vertices = transformed_vertices(metadata, feature)
    surfaces = lod_surfaces(feature)

    assert vertices[2].tolist() == [101.0, 201.0, 0.0]
    assert surfaces[0].semantic_type == "RoofSurface"
    assert surfaces[0].rings == ((0, 1, 2, 3),)


def test_writes_obj_with_semantic_material(tmp_path: Path) -> None:
    metadata, feature = sample_records()

    obj_path, material_path = write_obj(metadata, feature, tmp_path / "building.obj")

    assert "usemtl RoofSurface" in obj_path.read_text(encoding="utf-8")
    assert "f 1 2 3 4" in obj_path.read_text(encoding="utf-8")
    assert "newmtl RoofSurface" in material_path.read_text(encoding="utf-8")
