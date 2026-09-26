"""Provider-independent outputs produced by building-vision tasks."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class VisionModel(BaseModel):
    """Shared strict and immutable vision model."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AssessmentStatus(StrEnum):
    """Distinguish model observations from unsupported absence claims."""

    DETECTED = "DETECTED"
    NOT_DETECTED = "NOT_DETECTED"
    NOT_ASSESSABLE = "NOT_ASSESSABLE"
    UNAVAILABLE = "UNAVAILABLE"
    EXPERIMENTAL = "EXPERIMENTAL"


class GeoPolygon(VisionModel):
    """GeoJSON-compatible polygon with an explicit coordinate reference system."""

    type: Literal["Polygon"] = "Polygon"
    coordinates: tuple[tuple[tuple[float, float], ...], ...]
    crs: str = Field(pattern=r"^EPSG:\d+$")

    @model_validator(mode="after")
    def validate_rings(self) -> "GeoPolygon":
        """Require closed polygon rings with enough coordinates."""
        if not self.coordinates:
            raise ValueError("A polygon requires an exterior ring")
        for ring in self.coordinates:
            if len(ring) < 4 or ring[0] != ring[-1]:
                raise ValueError("Polygon rings must be closed and contain at least four positions")
        return self


class ModelProvenance(VisionModel):
    """Model identity required for every derived vision attribute."""

    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    taxonomy_version: str = Field(min_length=1)
    source_image: str = Field(min_length=1)


class RoofObjectInstance(VisionModel):
    """One segmented rooftop object and its optional map-space geometry."""

    object_id: str = Field(min_length=1)
    class_name: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    mask_artifact: str = Field(min_length=1)
    geometry_map: GeoPolygon | None = None
    pixel_area: int = Field(gt=0)
    area_m2: float | None = Field(default=None, gt=0)
    roof_surface_id: str | None = None
    provenance: ModelProvenance
    quality_flags: tuple[str, ...] = ()


class RoofVisionResult(VisionModel):
    """API-ready roof-vision result without embedding large raster masks."""

    building_id: str = Field(min_length=1)
    status: AssessmentStatus
    objects: tuple[RoofObjectInstance, ...] = ()
    warnings: tuple[str, ...] = ()
