import json
from pathlib import Path

import laspy
import numpy as np
import pytest
from pyproj import CRS

from urbanstock3d.errors import City3DExecutionError
from urbanstock3d.providers.city3d import City3DRun
from urbanstock3d.reconstruction import (
    BackendName,
    LodRequest,
    ReconstructionEvidence,
    ReconstructionStatus,
)
from urbanstock3d.reconstruction.backends.city3d import City3DBackend, assess_city3d_inputs


def _write_cloud(path: Path) -> None:
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.add_crs(CRS.from_epsg(25830))
    cloud = laspy.LasData(header)
    cloud.x = np.array([724_000.0, 724_001.0])
    cloud.y = np.array([4_372_000.0, 4_372_001.0])
    cloud.z = np.array([20.0, 21.0])
    cloud.classification = np.full(2, 2, dtype=np.uint8)
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
    assert assessment.ground_elevation_m == pytest.approx(20.5)


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


class SuccessfulRunner:
    def reconstruct(
        self,
        point_cloud: Path,
        footprint: Path,
        output_file: Path,
        *,
        ground_elevation_m: float,
    ) -> City3DRun:
        output_file.parent.mkdir(parents=True)
        output_file.write_text("v 0 0 0\n", encoding="utf-8")
        return City3DRun((), output_file, "", "")


class FailingRunner:
    def reconstruct(
        self,
        point_cloud: Path,
        footprint: Path,
        output_file: Path,
        *,
        ground_elevation_m: float,
    ) -> City3DRun:
        raise City3DExecutionError("native process failed")


def _valid_evidence(tmp_path: Path) -> ReconstructionEvidence:
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
    return ReconstructionEvidence("building-1", footprint_path, lidar_points=cloud_path)


def test_city3d_backend_returns_shared_success_contract(tmp_path: Path) -> None:
    backend = City3DBackend(SuccessfulRunner(), tmp_path / "outputs")

    result = backend.reconstruct(evidence=_valid_evidence(tmp_path), lod=LodRequest.LOD22)

    assert result.status is ReconstructionStatus.SUCCESS
    assert result.backend is BackendName.CITY3D
    assert result.delivered_lod is LodRequest.LOD22
    assert result.model_path == tmp_path / "outputs" / "building-1" / "city3d.obj"


def test_city3d_backend_normalizes_native_failure(tmp_path: Path) -> None:
    backend = City3DBackend(FailingRunner(), tmp_path / "outputs")

    result = backend.reconstruct(evidence=_valid_evidence(tmp_path), lod=LodRequest.LOD22)

    assert result.status is ReconstructionStatus.FAILED
    assert result.reasons == ("native process failed",)


def test_city3d_backend_rejects_unsupported_lod(tmp_path: Path) -> None:
    backend = City3DBackend(SuccessfulRunner(), tmp_path / "outputs")

    with pytest.raises(ValueError, match="only the detailed"):
        backend.reconstruct(evidence=_valid_evidence(tmp_path), lod=LodRequest.LOD13)
