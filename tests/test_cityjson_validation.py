import json
from pathlib import Path

import laspy
import numpy as np
import pytest

from urbanstock3d.reconstruction.validation import assess_cityjson_lidar_fit, validate_cityjsonseq

FOOTPRINT = ((((0.0, 0.0), (10.0, 0.0), (10.0, 8.0), (0.0, 8.0), (0.0, 0.0)),),)


def write_cube(path: Path, *, omit_wall: bool = False) -> None:
    metadata = {
        "type": "CityJSON",
        "version": "2.0",
        "transform": {"scale": [1.0, 1.0, 1.0], "translate": [0.0, 0.0, 0.0]},
    }
    faces = [
        [[0, 3, 2, 1]],
        [[4, 5, 6, 7]],
        [[0, 1, 5, 4]],
        [[1, 2, 6, 5]],
        [[2, 3, 7, 6]],
        [[3, 0, 4, 7]],
    ]
    semantic_values = [0, 2, 1, 1, 1, 1]
    if omit_wall:
        faces.pop()
        semantic_values.pop()
    feature = {
        "type": "CityJSONFeature",
        "id": "building-1",
        "vertices": [
            [0, 0, 0],
            [10, 0, 0],
            [10, 8, 0],
            [0, 8, 0],
            [0, 0, 10],
            [10, 0, 10],
            [10, 8, 10],
            [0, 8, 10],
        ],
        "CityObjects": {
            "part": {
                "type": "BuildingPart",
                "geometry": [
                    {
                        "type": "Solid",
                        "lod": "2.2",
                        "boundaries": [faces],
                        "semantics": {
                            "surfaces": [
                                {"type": "GroundSurface"},
                                {"type": "WallSurface"},
                                {"type": "RoofSurface"},
                            ],
                            "values": [semantic_values],
                        },
                    }
                ],
            }
        },
    }
    path.write_text(json.dumps(metadata) + "\n" + json.dumps(feature) + "\n", encoding="utf-8")


def write_roof_with_hole(path: Path) -> None:
    metadata = {
        "type": "CityJSON",
        "version": "2.0",
        "transform": {"scale": [1.0, 1.0, 1.0], "translate": [0.0, 0.0, 0.0]},
    }
    feature = {
        "type": "CityJSONFeature",
        "id": "roof-with-hole",
        "vertices": [
            [0, 0, 10],
            [10, 0, 10],
            [10, 10, 10],
            [0, 10, 10],
            [4, 4, 10],
            [4, 6, 10],
            [6, 6, 10],
            [6, 4, 10],
        ],
        "CityObjects": {
            "part": {
                "type": "BuildingPart",
                "geometry": [
                    {
                        "type": "Solid",
                        "lod": "2.2",
                        "boundaries": [[[[0, 1, 2, 3], [4, 5, 6, 7]]]],
                        "semantics": {"surfaces": [{"type": "RoofSurface"}], "values": [[0]]},
                    }
                ],
            }
        },
    }
    path.write_text(json.dumps(metadata) + "\n" + json.dumps(feature) + "\n", encoding="utf-8")


def test_accepts_watertight_semantic_solid_aligned_with_footprint(tmp_path: Path) -> None:
    model = tmp_path / "cube.city.jsonl"
    write_cube(model)

    report = validate_cityjsonseq(model, FOOTPRINT)

    assert report.accepted
    assert report.geometry_valid
    assert report.watertight
    assert report.face_count == 6
    assert report.semantic_counts == {"GroundSurface": 1, "RoofSurface": 1, "WallSurface": 4}
    assert report.height_range_m == pytest.approx(10.0)
    assert report.footprint_bbox_iou == pytest.approx(1.0)
    assert report.footprint_area_ratio == pytest.approx(1.0)


def test_rejects_open_solid(tmp_path: Path) -> None:
    model = tmp_path / "open.city.jsonl"
    write_cube(model, omit_wall=True)

    report = validate_cityjsonseq(model, FOOTPRINT)

    assert not report.accepted
    assert not report.watertight
    assert "solid boundary is not watertight" in report.failures


def test_rejects_model_that_does_not_align_with_source_footprint(tmp_path: Path) -> None:
    model = tmp_path / "cube.city.jsonl"
    write_cube(model)
    shifted_footprint = ((((20.0, 0.0), (30.0, 0.0), (30.0, 8.0), (20.0, 8.0), (20.0, 0.0)),),)

    report = validate_cityjsonseq(model, shifted_footprint)

    assert not report.plausibility_pass
    assert report.footprint_bbox_iou == 0.0


def test_measures_independent_point_to_roof_distances(tmp_path: Path) -> None:
    model = tmp_path / "cube.city.jsonl"
    lidar = tmp_path / "roof.las"
    write_cube(model)
    header = laspy.LasHeader(point_format=6, version="1.4")
    cloud = laspy.LasData(header)
    cloud.x = np.array([2.0, 4.0, 6.0, 8.0])
    cloud.y = np.array([2.0, 4.0, 6.0, 7.0])
    cloud.z = np.array([10.0, 10.2, 9.8, 11.0])
    cloud.classification = np.full(4, 6, dtype=np.uint8)
    cloud.write(lidar)

    report = assess_cityjson_lidar_fit(model, lidar, FOOTPRINT)

    assert report.source_roof_point_count == 4
    assert report.roof_triangle_count == 2
    assert report.distance_rmse_m == pytest.approx(np.sqrt(1.08 / 4))
    assert report.distance_p95_m == pytest.approx(0.88)
    assert report.within_020m_ratio == pytest.approx(0.75)


def test_roof_fit_respects_surface_interior_rings(tmp_path: Path) -> None:
    model = tmp_path / "roof-with-hole.city.jsonl"
    lidar = tmp_path / "roof.las"
    write_roof_with_hole(model)
    header = laspy.LasHeader(point_format=6, version="1.4")
    cloud = laspy.LasData(header)
    cloud.x = np.array([5.0])
    cloud.y = np.array([5.0])
    cloud.z = np.array([10.0])
    cloud.classification = np.array([6], dtype=np.uint8)
    cloud.write(lidar)

    report = assess_cityjson_lidar_fit(model, lidar, FOOTPRINT)

    assert report.distance_median_m == pytest.approx(1.0)
    assert report.roof_triangle_count == 8
