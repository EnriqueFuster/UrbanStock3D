"""Provider-independent cadastral building models."""

from datetime import date, datetime
from typing import Annotated

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, StringConstraints, field_validator

CadastralRootId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^[0-9A-Z]{14}$",
    ),
]
Position2D = tuple[float, float]
LinearRing = Annotated[tuple[Position2D, ...], Field(min_length=4)]


class DomainModel(BaseModel):
    """Shared validation behaviour for immutable domain values."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class CadastralIdentity(DomainModel):
    """Stable identifiers returned for one cadastral building."""

    cadastral_root_id: CadastralRootId
    local_id: str = Field(min_length=1)
    namespace: str = Field(min_length=1)

    @field_validator("cadastral_root_id", mode="before")
    @classmethod
    def normalize_cadastral_root(cls, value: object) -> object:
        """Normalize textual cadastral roots before validating their format."""
        return value.strip().upper() if isinstance(value, str) else value

    @property
    def building_id(self) -> str:
        """Return the complete INSPIRE identifier."""
        return f"{self.namespace}.{self.local_id}"


class CoordinateQuery(DomainModel):
    """A geographic point used to resolve a cadastral parcel."""

    longitude: float = Field(ge=-180, le=180)
    latitude: float = Field(ge=-90, le=90)


class CoordinateResolution(DomainModel):
    """The cadastral parcel reference found at a geographic point."""

    query: CoordinateQuery
    cadastral_root_id: CadastralRootId
    address: str | None = None


class RefcatQuery(DomainModel):
    """A 14, 18 or 20 character cadastral reference supplied by a user."""

    value: str

    @field_validator("value", mode="before")
    @classmethod
    def normalize_refcat(cls, value: object) -> object:
        """Remove surrounding whitespace and normalize letters to uppercase."""
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("value")
    @classmethod
    def validate_refcat(cls, value: str) -> str:
        """Accept only documented Catastro reference lengths and characters."""
        if len(value) not in {14, 18, 20}:
            raise ValueError("REFCAT must contain 14, 18 or 20 characters")
        if not value.isalnum() or not value.isascii():
            raise ValueError("REFCAT must contain only ASCII letters and numbers")
        return value

    @property
    def cadastral_root_id(self) -> str:
        """Return the parcel-level first 14 characters."""
        return self.value[:14]


class ConstructionPeriod(DomainModel):
    """Registered construction period, when supplied by Catastro."""

    beginning: date
    end: date


class OfficialArea(DomainModel):
    """An official cadastral area and its declared meaning."""

    value_m2: float = Field(ge=0)
    reference: str = Field(min_length=1)


class FacadeDocument(DomainModel):
    """Reference to a facade document supplied by Catastro."""

    url: AnyHttpUrl
    media_format: str = Field(min_length=1)
    source_status: str | None = None


class Footprint2D(DomainModel):
    """Building footprint rings in an explicit coordinate system."""

    crs: str = Field(pattern=r"^EPSG:\d+$")
    rings: tuple[LinearRing, ...] = Field(min_length=1)
    estimated_accuracy_m: float | None = Field(default=None, ge=0)
    reference: str | None = None


class CadastralBuildingPart(DomainModel):
    """A cadastral subdivision with its registered floor counts."""

    local_id: str = Field(min_length=1)
    floors_above_ground: int | None = Field(default=None, ge=0)
    floors_below_ground: int | None = Field(default=None, ge=0)
    height_below_ground_m: float | None = Field(default=None, ge=0)
    footprint: Footprint2D


class CadastralBuilding(DomainModel):
    """Official cadastral attributes for one resolved building."""

    identity: CadastralIdentity
    footprint: Footprint2D
    parts: tuple[CadastralBuildingPart, ...] = ()
    condition: str | None = None
    current_use: str | None = None
    construction_period: ConstructionPeriod | None = None
    building_units: int | None = Field(default=None, ge=0)
    dwellings: int | None = Field(default=None, ge=0)
    floors_above_ground: int | None = Field(default=None, ge=0)
    official_area: OfficialArea | None = None
    facade_document: FacadeDocument | None = None
    begin_lifespan_version: datetime | None = None
    end_lifespan_version: datetime | None = None
