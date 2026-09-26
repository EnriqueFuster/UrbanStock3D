"""RID2 source audit and canonical label mapping."""

import hashlib
from enum import StrEnum
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from urbanstock3d.vision.common.taxonomy import CanonicalTaxonomy


class DatasetModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetAccess(StrEnum):
    OPEN = "open"
    RESTRICTED = "restricted"


class SourceClassMapping(DatasetModel):
    """One reviewed source label and its optional product class."""

    name: str = Field(min_length=1)
    source_label: str = Field(min_length=1)
    canonical: str | None
    note: str = Field(min_length=1)


class Rid2SourceConfig(DatasetModel):
    """Pinned RID2 source identity and licence gate."""

    dataset_name: str = Field(pattern="^RID2$")
    source_version: str = Field(min_length=1)
    record_url: HttpUrl
    archive_url: HttpUrl
    archive_name: str = Field(min_length=1)
    archive_size_bytes: int | None = Field(default=None, gt=0)
    archive_md5: str = Field(pattern=r"^[0-9a-f]{32}$")
    licence: str | None
    licence_reviewed: bool
    access: DatasetAccess
    source_classes: dict[int, SourceClassMapping]

    @model_validator(mode="after")
    def validate_source_ids(self) -> "Rid2SourceConfig":
        identifiers = set(self.source_classes)
        if identifiers != set(range(len(identifiers))):
            raise ValueError("RID2 source class IDs must be contiguous from zero")
        if self.licence_reviewed and not self.licence:
            raise ValueError("A reviewed dataset must declare its licence")
        labels = [mapping.source_label for mapping in self.source_classes.values()]
        if len(labels) != len(set(labels)):
            raise ValueError("RID2 source labels must be unique")
        return self

    @property
    def download_allowed(self) -> bool:
        """Allow acquisition when the repository exposes the files as open access."""
        return self.access is DatasetAccess.OPEN

    @property
    def commercial_reuse_confirmed(self) -> bool:
        """Report whether redistribution/commercial reuse has explicit terms."""
        return self.licence_reviewed and self.licence is not None


class DatasetAudit(DatasetModel):
    """Human-readable readiness report generated before data acquisition."""

    dataset_name: str
    source_version: str
    source_class_count: int
    mapped_source_class_count: int
    ignored_source_classes: tuple[str, ...]
    missing_canonical_classes: tuple[str, ...]
    mapping_warnings: tuple[str, ...]
    licence: str | None
    licence_reviewed: bool
    download_allowed: bool
    commercial_reuse_confirmed: bool
    ready_for_conversion: bool


class ArchiveInventory(DatasetModel):
    """Compact, serializable inventory produced without extracting an archive."""

    archive_name: str
    file_count: int
    compressed_size_bytes: int
    uncompressed_size_bytes: int
    suffix_counts: dict[str, int]
    top_level_entries: tuple[str, ...]
    sample_members: tuple[str, ...]


def load_rid2_source(path: Path) -> Rid2SourceConfig:
    """Load the pinned RID2 source description."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("RID2 source YAML must contain a mapping")
    return Rid2SourceConfig.model_validate(payload)


def audit_rid2(source: Rid2SourceConfig, taxonomy: CanonicalTaxonomy) -> DatasetAudit:
    """Validate source mappings without claiming ignored labels as ground truth."""
    mapped = [mapping for mapping in source.source_classes.values() if mapping.canonical]
    mapped_names = {mapping.canonical for mapping in mapped}
    for canonical in mapped_names:
        if canonical is not None:
            taxonomy.require_class(canonical)
    ignored = tuple(
        mapping.name for mapping in source.source_classes.values() if mapping.canonical is None
    )
    missing = tuple(name for name in taxonomy.roof_objects if name not in mapped_names)
    warnings = tuple(
        f"{mapping.name} -> {mapping.canonical}: {mapping.note}"
        for mapping in mapped
        if "Merged" in mapping.note or "Includes" in mapping.note
    )
    # One source does not need to cover the complete product taxonomy. Missing
    # classes can be supplied by another reviewed dataset.
    ready = source.download_allowed
    return DatasetAudit(
        dataset_name=source.dataset_name,
        source_version=source.source_version,
        source_class_count=len(source.source_classes),
        mapped_source_class_count=len(mapped),
        ignored_source_classes=ignored,
        missing_canonical_classes=missing,
        mapping_warnings=warnings,
        licence=source.licence,
        licence_reviewed=source.licence_reviewed,
        download_allowed=source.download_allowed,
        commercial_reuse_confirmed=source.commercial_reuse_confirmed,
        ready_for_conversion=ready,
    )


def verify_rid2_archive(path: Path, source: Rid2SourceConfig) -> None:
    """Verify archive identity against exact metadata published by Zenodo."""
    if path.name != source.archive_name:
        raise ValueError(f"Expected archive name {source.archive_name}")
    if source.archive_size_bytes is not None and path.stat().st_size != source.archive_size_bytes:
        raise ValueError("RID2 archive size does not match pinned source metadata")
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != source.archive_md5:
        raise ValueError("RID2 archive checksum does not match pinned source metadata")


def inspect_zip_archive(path: Path, *, sample_size: int = 20) -> ArchiveInventory:
    """Inventory a ZIP and reject members that could escape an extraction root."""
    suffix_counts: dict[str, int] = {}
    top_level_entries: set[str] = set()
    samples: list[str] = []
    compressed_size = 0
    uncompressed_size = 0
    file_count = 0

    with ZipFile(path) as archive:
        for member in archive.infolist():
            normalized_name = member.filename.replace("\\", "/")
            member_path = PurePosixPath(normalized_name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe ZIP member path: {member.filename}")
            if not member_path.parts:
                continue
            top_level_entries.add(member_path.parts[0])
            if member.is_dir():
                continue
            file_count += 1
            compressed_size += member.compress_size
            uncompressed_size += member.file_size
            suffix = member_path.suffix.lower() or "<no suffix>"
            suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
            if len(samples) < sample_size:
                samples.append(normalized_name)

    return ArchiveInventory(
        archive_name=path.name,
        file_count=file_count,
        compressed_size_bytes=compressed_size,
        uncompressed_size_bytes=uncompressed_size,
        suffix_counts=dict(sorted(suffix_counts.items())),
        top_level_entries=tuple(sorted(top_level_entries)),
        sample_members=tuple(samples),
    )
