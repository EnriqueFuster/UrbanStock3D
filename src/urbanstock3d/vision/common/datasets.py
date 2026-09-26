"""Dataset boundary for reproducible source-specific adapters."""

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from urbanstock3d.vision.common.taxonomy import CanonicalTaxonomy


class DatasetManifest(BaseModel):
    """Minimum auditable identity of a processed dataset version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1)
    taxonomy_version: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    licence: str = Field(min_length=1)
    total_samples: int = Field(ge=0)


class DatasetAdapter(Protocol):
    """Keep download/conversion concerns outside model training."""

    name: str

    def download(self, destination: Path) -> None: ...

    def verify(self, root: Path) -> None: ...

    def convert_to_canonical(
        self, source_root: Path, output_root: Path, taxonomy: CanonicalTaxonomy
    ) -> DatasetManifest: ...
