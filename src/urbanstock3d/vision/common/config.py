"""Validated configuration for the first roof-object vertical slice."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from urbanstock3d.vision.common.taxonomy import CanonicalTaxonomy


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(ConfigModel):
    family: str = Field(min_length=1)
    checkpoint: str = Field(min_length=1)
    pretrained: bool


class InputConfig(ConfigModel):
    image_size: int = Field(gt=0)
    tiling: bool
    overlap: float = Field(ge=0, lt=1)


class TrainingConfig(ConfigModel):
    image_size: int = Field(gt=0)
    epochs: int = Field(gt=0)
    batch_size: int | Literal["auto"]
    seed: int = Field(ge=0)
    early_stopping_patience: int = Field(gt=0)


class TrackingConfig(ConfigModel):
    experiment: str = Field(pattern=r"^urbanstock/[a-z0-9_]+$")


class ExportConfig(ConfigModel):
    formats: tuple[Literal["onnx"], ...]


class RoofObjectsConfig(ConfigModel):
    """Complete V0 configuration for the roof-object task."""

    task_name: Literal["roof_objects"]
    task_type: Literal["instance_segmentation"]
    taxonomy_version: str
    dataset_id: str | None
    model: ModelConfig
    input: InputConfig
    training: TrainingConfig
    tracking: TrackingConfig
    export: ExportConfig

    @model_validator(mode="after")
    def validate_tiling(self) -> "RoofObjectsConfig":
        if not self.input.tiling and self.input.overlap != 0:
            raise ValueError("Tile overlap must be zero when tiling is disabled")
        return self

    def validate_taxonomy(self, taxonomy: CanonicalTaxonomy) -> None:
        """Reject a task configuration bound to another taxonomy version."""
        if self.taxonomy_version != taxonomy.version:
            raise ValueError(
                f"Config taxonomy {self.taxonomy_version} does not match {taxonomy.version}"
            )


def load_roof_objects_config(path: Path) -> RoofObjectsConfig:
    """Load and validate roof-object configuration from YAML."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Roof-object YAML must contain a mapping")
    return RoofObjectsConfig.model_validate(payload)
