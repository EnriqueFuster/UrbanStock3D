"""Versioned canonical taxonomies independent of source datasets."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class CanonicalTaxonomy(BaseModel):
    """Stable production class identifiers for implemented vision tasks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(min_length=1)
    roof_objects: dict[str, int]

    @model_validator(mode="after")
    def validate_class_ids(self) -> "CanonicalTaxonomy":
        """Prevent ambiguous or silently reordered class identifiers."""
        identifiers = tuple(self.roof_objects.values())
        if not self.roof_objects:
            raise ValueError("Roof-object taxonomy cannot be empty")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Taxonomy class identifiers must be unique")
        if set(identifiers) != set(range(len(identifiers))):
            raise ValueError("Taxonomy class identifiers must be contiguous from zero")
        return self

    def require_class(self, name: str) -> int:
        """Return an ID or reject predictions outside the canonical taxonomy."""
        try:
            return self.roof_objects[name]
        except KeyError as error:
            raise ValueError(f"Unknown canonical roof-object class: {name}") from error


def load_taxonomy(path: Path) -> CanonicalTaxonomy:
    """Load and validate one canonical taxonomy YAML file."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Taxonomy YAML must contain a mapping")
    return CanonicalTaxonomy.model_validate(payload)
