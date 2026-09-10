"""Input compatibility checks for the research-grade City3D backend."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import laspy
import numpy as np

from urbanstock3d.errors import City3DExecutionError
from urbanstock3d.processors.lidar import GROUND_CLASS
from urbanstock3d.providers.city3d import City3DRun
from urbanstock3d.reconstruction.backends.base import BackendCapabilities
from urbanstock3d.reconstruction.enums import BackendName, LodRequest, ReconstructionStatus
from urbanstock3d.reconstruction.models import (
    GeometryProvenance,
    ReconstructionEvidence,
    ReconstructionResult,
)


class City3DRunner(Protocol):
    """Narrow native-client interface required by the backend adapter."""

    def reconstruct(
        self,
        point_cloud: Path,
        footprint: Path,
        output_file: Path,
        *,
        ground_elevation_m: float,
    ) -> City3DRun: ...


@dataclass(frozen=True)
class City3DInputAssessment:
    """Evidence that existing project artifacts can be consumed by City3D."""

    point_cloud: Path
    footprint: Path
    point_count: int
    epsg: int
    footprint_vertex_count: int
    ground_elevation_m: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["point_cloud"] = str(self.point_cloud)
        payload["footprint"] = str(self.footprint)
        return payload


class City3DBackend:
    """Expose the native City3D wrapper through the shared backend contract."""

    capabilities = BackendCapabilities(
        name=BackendName.CITY3D,
        supported_lods=frozenset({LodRequest.LOD22}),
        requires_lidar=True,
        requires_dsm=False,
        requires_orthophoto=False,
        requires_gpu=False,
        learned_geometry=False,
        maturity="research_candidate",
        output_kind="OBJ",
    )

    def __init__(self, client: City3DRunner, output_root: Path) -> None:
        self.client = client
        self.output_root = output_root

    def is_available(self) -> bool:
        return True

    def reconstruct(
        self,
        *,
        evidence: ReconstructionEvidence,
        lod: LodRequest,
    ) -> ReconstructionResult:
        if lod is not LodRequest.LOD22:
            raise ValueError("City3D currently supports only the detailed LoD2.2 candidate")
        if not isinstance(evidence.lidar_points, Path) or not isinstance(evidence.footprint, Path):
            raise TypeError("City3D evidence must contain pathlib.Path inputs")
        try:
            output_file = self.output_root / evidence.building_id / "city3d.obj"
            assessment = assess_city3d_inputs(evidence.lidar_points, evidence.footprint)
            run = self.client.reconstruct(
                evidence.lidar_points,
                evidence.footprint,
                output_file,
                ground_elevation_m=assessment.ground_elevation_m,
            )
        except (City3DExecutionError, OSError, ValueError) as error:
            return ReconstructionResult(
                status=ReconstructionStatus.FAILED,
                requested_lod=lod,
                targeted_lod=lod,
                delivered_lod=None,
                backend=BackendName.CITY3D,
                model_path=None,
                provenance=None,
                reasons=(str(error),),
            )
        return ReconstructionResult(
            status=ReconstructionStatus.SUCCESS,
            requested_lod=lod,
            targeted_lod=lod,
            delivered_lod=lod,
            backend=BackendName.CITY3D,
            model_path=run.output_file,
            provenance=GeometryProvenance(lidar_observed=True),
        )


def assess_city3d_inputs(point_cloud: Path, footprint: Path) -> City3DInputAssessment:
    """Validate the narrow City3D input subset used by UrbanStock3D."""
    if point_cloud.suffix.lower() not in {".las", ".laz"}:
        raise ValueError("City3D point cloud must be LAS or LAZ")
    if not point_cloud.is_file():
        raise FileNotFoundError(point_cloud)
    if not footprint.is_file():
        raise FileNotFoundError(footprint)

    ground_chunks: list[np.ndarray[Any, np.dtype[np.floating[Any]]]] = []
    with laspy.open(point_cloud) as reader:
        point_count = reader.header.point_count
        crs = reader.header.parse_crs()
        for points in reader.chunk_iterator(1_000_000):
            classification = np.asarray(points.classification, dtype=np.uint8)
            ground_chunks.append(np.asarray(points.z)[classification == GROUND_CLASS])
    if point_count == 0:
        raise ValueError("City3D point cloud cannot be empty")
    epsg = crs.to_epsg() if crs is not None else None
    if epsg != 25830:
        raise ValueError("City3D inputs must use EPSG:25830")
    non_empty_ground = [chunk for chunk in ground_chunks if len(chunk)]
    if not non_empty_ground:
        raise ValueError("City3D point cloud contains no classified ground points")
    ground = np.concatenate(non_empty_ground)

    collection: dict[str, Any] = json.loads(footprint.read_text(encoding="utf-8"))
    crs_name = str(collection.get("crs", {}).get("properties", {}).get("name", ""))
    if not crs_name.endswith("25830"):
        raise ValueError("City3D footprint must declare EPSG:25830")
    features = collection.get("features", [])
    if len(features) != 1:
        raise ValueError("City3D input must contain exactly one footprint feature")
    geometry = features[0].get("geometry", {})
    if geometry.get("type") != "Polygon":
        raise ValueError("City3D integration currently supports one Polygon footprint")
    rings = geometry.get("coordinates", [])
    if len(rings) != 1:
        raise ValueError("City3D integration currently rejects footprint holes")
    exterior = rings[0]
    if len(exterior) < 4 or exterior[0] != exterior[-1]:
        raise ValueError("City3D footprint exterior must be a closed polygon")

    return City3DInputAssessment(
        point_cloud=point_cloud,
        footprint=footprint,
        point_count=point_count,
        epsg=epsg,
        footprint_vertex_count=len(exterior) - 1,
        ground_elevation_m=float(np.median(ground)),
    )
