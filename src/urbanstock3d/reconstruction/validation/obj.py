"""Structural and LiDAR-fit checks for City3D OBJ meshes."""

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from urbanstock3d.reconstruction.quality.lidar import Footprint
from urbanstock3d.reconstruction.validation.cityjson import (
    CityJsonQualityParameters,
    _bbox_iou,
    _footprint_bbox,
    _polygon_area_3d,
)
from urbanstock3d.reconstruction.validation.lidar_fit import (
    LidarFitParameters,
    LidarFitReport,
    LidarResiduals,
    _load_roof_observations,
    point_triangle_distance,
)


@dataclass(frozen=True)
class ObjQualityReport:
    """Backend-neutral fields required to compare a City3D OBJ mesh."""

    lod: str
    vertex_count: int
    face_count: int
    duplicate_vertex_count: int
    boundary_edge_count: int
    non_manifold_edge_count: int
    semantic_counts: dict[str, int]
    watertight: bool
    invalid_ring_count: int
    tiny_face_fraction: float
    height_range_m: float
    footprint_bbox_iou: float
    footprint_area_ratio: float | None
    geometry_valid: bool
    plausibility_pass: bool
    accepted: bool
    failures: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def read_obj(
    path: Path,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], tuple[tuple[int, ...], ...]]:
    """Read the vertex and polygon records needed by the benchmark."""
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or fields[0] == "#":
            continue
        if fields[0] == "v" and len(fields) >= 4:
            vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
        elif fields[0] == "f":
            indices = tuple(int(value.split("/", 1)[0]) for value in fields[1:])
            faces.append(
                tuple(index - 1 if index > 0 else len(vertices) + index for index in indices)
            )
    return np.asarray(vertices, dtype=np.float64).reshape((-1, 3)), tuple(faces)


def validate_obj(
    model_path: Path,
    footprint: Footprint,
    *,
    parameters: CityJsonQualityParameters | None = None,
) -> ObjQualityReport:
    """Validate City3D geometry using the same structural gates as CityJSON."""
    parameters = parameters or CityJsonQualityParameters()
    vertices, faces = read_obj(model_path)
    invalid_faces = sum(
        len(face) < 3 or any(index < 0 or index >= len(vertices) for index in face)
        for face in faces
    )
    valid_faces = [
        face for face in faces if len(face) >= 3 and all(0 <= i < len(vertices) for i in face)
    ]
    _, welded_indices = np.unique(np.round(vertices, decimals=6), axis=0, return_inverse=True)
    edge_counts: Counter[tuple[int, int]] = Counter()
    areas: list[float] = []
    for face in valid_faces:
        welded_face = tuple(int(welded_indices[index]) for index in face)
        closed = (*welded_face, welded_face[0])
        edge_counts.update((min(a, b), max(a, b)) for a, b in zip(closed, closed[1:]) if a != b)
        areas.append(_polygon_area_3d(vertices[np.asarray(face)]))
    finite = bool(len(vertices)) and bool(np.all(np.isfinite(vertices)))
    watertight = bool(edge_counts) and all(count == 2 for count in edge_counts.values())
    boundary_edge_count = sum(count == 1 for count in edge_counts.values())
    non_manifold_edge_count = sum(count > 2 for count in edge_counts.values())
    height = float(np.ptp(vertices[:, 2])) if finite else 0.0
    model_bbox = (
        (
            float(np.min(vertices[:, 0])),
            float(np.min(vertices[:, 1])),
            float(np.max(vertices[:, 0])),
            float(np.max(vertices[:, 1])),
        )
        if finite
        else (0.0, 0.0, 0.0, 0.0)
    )
    bbox_iou = _bbox_iou(_footprint_bbox(footprint), model_bbox) if finite else 0.0
    tiny_fraction = (
        float(np.mean(np.asarray(areas) < parameters.tiny_face_area_m2)) if areas else 1.0
    )
    failures: list[str] = []
    if not finite:
        failures.append("vertices are empty or non-finite")
    if invalid_faces:
        failures.append("faces contain invalid vertex references")
    if not watertight:
        failures.append("mesh boundary is not watertight")
    geometry_valid = not failures
    plausibility_failures: list[str] = []
    if bbox_iou < parameters.minimum_footprint_bbox_iou:
        plausibility_failures.append("model footprint bbox does not align with source footprint")
    if not parameters.minimum_height_m <= height <= parameters.maximum_height_m:
        plausibility_failures.append("model height is outside configured plausibility bounds")
    failures.extend(plausibility_failures)
    duplicate_vertices = len(vertices) - len(np.unique(np.round(vertices, decimals=6), axis=0))
    warnings = ["OBJ has no semantic surface labels"]
    if duplicate_vertices:
        warnings.append(f"{duplicate_vertices} duplicate vertices welded for topology checks")
    return ObjQualityReport(
        lod="2.2",
        vertex_count=len(vertices),
        face_count=len(valid_faces),
        duplicate_vertex_count=duplicate_vertices,
        boundary_edge_count=boundary_edge_count,
        non_manifold_edge_count=non_manifold_edge_count,
        semantic_counts={},
        watertight=watertight,
        invalid_ring_count=invalid_faces,
        tiny_face_fraction=tiny_fraction,
        height_range_m=height,
        footprint_bbox_iou=bbox_iou,
        footprint_area_ratio=None,
        geometry_valid=geometry_valid,
        plausibility_pass=not plausibility_failures,
        accepted=not failures,
        failures=tuple(failures),
        warnings=tuple(warnings),
    )


def assess_obj_lidar_fit(
    model_path: Path,
    lidar_path: Path,
    footprint: Footprint,
    *,
    parameters: LidarFitParameters | None = None,
) -> LidarFitReport:
    """Measure roof observations against upward-facing City3D triangles."""
    parameters = parameters or LidarFitParameters()
    residuals = compute_obj_lidar_residuals(
        model_path, lidar_path, footprint, parameters=parameters
    )
    distances = residuals.distances_m
    return LidarFitReport(
        source_roof_point_count=residuals.source_point_count,
        selected_roof_point_count=residuals.selected_point_count,
        sampled_roof_point_count=len(residuals.points),
        roof_triangle_count=residuals.roof_triangle_count,
        distance_median_m=float(np.median(distances)),
        distance_rmse_m=float(np.sqrt(np.mean(np.square(distances)))),
        distance_p95_m=float(np.percentile(distances, 95)),
        within_020m_ratio=float(np.mean(distances <= 0.2 + 1e-9)),
        within_050m_ratio=float(np.mean(distances <= 0.5 + 1e-9)),
        within_100m_ratio=float(np.mean(distances <= 1.0 + 1e-9)),
        warnings=residuals.warnings,
    )


def compute_obj_lidar_residuals(
    model_path: Path,
    lidar_path: Path,
    footprint: Footprint,
    *,
    parameters: LidarFitParameters | None = None,
) -> LidarResiduals:
    """Return spatial observations and residuals for an OBJ diagnostic plot."""
    parameters = parameters or LidarFitParameters()
    vertices, faces = read_obj(model_path)
    triangles = _roof_triangles(vertices, faces)
    points, source_count, selected_count, selection_warning = _load_roof_observations(
        lidar_path, footprint, parameters
    )
    distances = np.asarray(
        [
            min(point_triangle_distance(point, triangle) for triangle in triangles)
            for point in points
        ]
    )
    return LidarResiduals(
        points=points,
        distances_m=distances,
        source_point_count=source_count,
        selected_point_count=selected_count,
        roof_triangle_count=len(triangles),
        warnings=(
            "OBJ roof faces inferred from orientation and elevation",
            selection_warning,
        ),
    )


def _roof_triangles(
    vertices: np.ndarray[Any, np.dtype[np.float64]], faces: tuple[tuple[int, ...], ...]
) -> list[np.ndarray[Any, np.dtype[np.float64]]]:
    if not len(vertices):
        raise ValueError("OBJ contains no vertices")
    minimum_z = float(np.min(vertices[:, 2]))
    triangles: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    for face in faces:
        if len(face) < 3 or any(index < 0 or index >= len(vertices) for index in face):
            continue
        points = vertices[np.asarray(face)]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        norm = float(np.linalg.norm(normal))
        if (
            norm
            and abs(float(normal[2])) / norm >= 0.25
            and float(np.mean(points[:, 2])) > minimum_z + 1.0
        ):
            triangles.extend(
                vertices[np.asarray((face[0], face[index], face[index + 1]))]
                for index in range(1, len(face) - 1)
            )
    if not triangles:
        raise ValueError("OBJ contains no inferred roof triangles")
    return triangles
