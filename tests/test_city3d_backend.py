import json
from pathlib import Path

import laspy
import numpy as np
import pytest
from pyproj import CRS

from urbanstock3d.reconstruction.backends.city3d import assess_city3d_inputs


def _write_cloud(path: Path) -> None:
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.add_crs(CRS.from_epsg(25830))
    cloud = laspy.LasData(header)
    cloud.x = np.array([724_000.0, 724_001.0])
    cloud.y = np.array([4_372_000.0, 4_372_001.0])
    cloud.z = np.array([20.0, 21.0])
    cloud.write(path)


def _write_footprint(path: Path, geometry: dict[str, object]) -> None:
    collection = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::25830"}},
        "features": [{"type": "Feature", "properties": {}, "geometry": geometry}],
    }
    path.write_text(json.dumps(collection), encoding="utf-8")


def test_assess_city3d_inputs_accepts_projected_laz_and_simple_polygon(tmp_path: Path) -> None:
    cloud_path = tmp_path / "crop.laz"
    footprint_path = tmp_path / "footprint.geojson"
    _write_cloud(cloud_path)
    _write_footprint(
        footprint_path,
        {
            "type": "Polygon",
            "coordinates": [
                [[724000, 4372000], [724010, 4372000], [724010, 4372010], [724000, 4372000]]
            ],
        },
    )

    assessment = assess_city3d_inputs(cloud_path, footprint_path)

    assert assessment.point_count == 2
    assert assessment.epsg == 25830
    assert assessment.footprint_vertex_count == 3


def test_assess_city3d_inputs_rejects_multipolygon(tmp_path: Path) -> None:
    cloud_path = tmp_path / "crop.laz"
    footprint_path = tmp_path / "footprint.geojson"
    _write_cloud(cloud_path)
    _write_footprint(footprint_path, {"type": "MultiPolygon", "coordinates": []})

    with pytest.raises(ValueError, match="one Polygon"):
        assess_city3d_inputs(cloud_path, footprint_path)


def test_assess_city3d_inputs_rejects_polygon_holes(tmp_path: Path) -> None:
    cloud_path = tmp_path / "crop.laz"
    footprint_path = tmp_path / "footprint.geojson"
    _write_cloud(cloud_path)
    ring = [[724000, 4372000], [724010, 4372000], [724010, 4372010], [724000, 4372000]]
    _write_footprint(footprint_path, {"type": "Polygon", "coordinates": [ring, ring]})

    with pytest.raises(ValueError, match="rejects footprint holes"):
        assess_city3d_inputs(cloud_path, footprint_path)
