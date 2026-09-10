from pathlib import Path

import laspy
import numpy as np
import pytest

from urbanstock3d.reconstruction.validation import assess_obj_lidar_fit, validate_obj

FOOTPRINT = ((((0.0, 0.0), (10.0, 0.0), (10.0, 8.0), (0.0, 8.0), (0.0, 0.0)),),)


def _write_cube(path: Path, *, omit_face: bool = False) -> None:
    vertices = (
        (0, 0, 0),
        (10, 0, 0),
        (10, 8, 0),
        (0, 8, 0),
        (0, 0, 10),
        (10, 0, 10),
        (10, 8, 10),
        (0, 8, 10),
    )
    faces = [(1, 4, 3, 2), (5, 6, 7, 8), (1, 2, 6, 5), (2, 3, 7, 6), (3, 4, 8, 7), (4, 1, 5, 8)]
    if omit_face:
        faces.pop()
    lines = [
        *(f"v {x} {y} {z}" for x, y, z in vertices),
        *("f " + " ".join(map(str, face)) for face in faces),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_validates_watertight_obj_against_footprint(tmp_path: Path) -> None:
    model = tmp_path / "cube.obj"
    _write_cube(model)

    report = validate_obj(model, FOOTPRINT)

    assert report.accepted
    assert report.watertight
    assert report.vertex_count == 8
    assert report.face_count == 6
    assert report.duplicate_vertex_count == 0
    assert report.boundary_edge_count == 0
    assert report.footprint_bbox_iou == pytest.approx(1.0)


def test_rejects_open_obj(tmp_path: Path) -> None:
    model = tmp_path / "open.obj"
    _write_cube(model, omit_face=True)

    report = validate_obj(model, FOOTPRINT)

    assert not report.accepted
    assert "mesh boundary is not watertight" in report.failures


def test_measures_obj_roof_fit(tmp_path: Path) -> None:
    model = tmp_path / "cube.obj"
    lidar = tmp_path / "roof.las"
    _write_cube(model)
    header = laspy.LasHeader(point_format=6, version="1.4")
    cloud = laspy.LasData(header)
    cloud.x = np.array([2.0, 4.0, 6.0])
    cloud.y = np.array([2.0, 4.0, 6.0])
    cloud.z = np.array([10.0, 10.2, 9.8])
    cloud.classification = np.full(3, 6, dtype=np.uint8)
    cloud.write(lidar)

    report = assess_obj_lidar_fit(model, lidar, FOOTPRINT)

    assert report.roof_triangle_count == 2
    assert report.distance_rmse_m == pytest.approx(np.sqrt(0.08 / 3))
