from pathlib import Path
from zipfile import ZipFile

import pytest

from urbanstock3d.vision.common import load_taxonomy
from urbanstock3d.vision.roof_objects.datasets import (
    audit_rid2,
    inspect_zip_archive,
    load_rid2_source,
    verify_rid2_archive,
)

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config" / "vision"


def test_audit_exposes_mapping_and_licence_blocker() -> None:
    taxonomy = load_taxonomy(CONFIG_ROOT / "taxonomy.yaml")
    source = load_rid2_source(CONFIG_ROOT / "datasets" / "rid2.yaml")

    audit = audit_rid2(source, taxonomy)

    assert audit.source_class_count == 12
    assert audit.mapped_source_class_count == 5
    assert audit.missing_canonical_classes == ("elevator_overrun",)
    assert "dormer" in audit.ignored_source_classes
    assert not audit.licence_reviewed
    assert audit.download_allowed
    assert audit.ready_for_conversion
    assert not audit.commercial_reuse_confirmed


def test_archive_verifier_rejects_unpinned_file(tmp_path: Path) -> None:
    source = load_rid2_source(CONFIG_ROOT / "datasets" / "rid2.yaml")
    archive = tmp_path / source.archive_name
    archive.write_bytes(b"not RID2")

    with pytest.raises(ValueError, match="checksum"):
        verify_rid2_archive(archive, source)


def test_zip_inventory_counts_files_without_extracting(tmp_path: Path) -> None:
    archive = tmp_path / "sample.zip"
    with ZipFile(archive, "w") as output:
        output.writestr("RID2/images/one.tif", b"image")
        output.writestr("RID2/masks/one.png", b"mask")

    inventory = inspect_zip_archive(archive)

    assert inventory.file_count == 2
    assert inventory.suffix_counts == {".png": 1, ".tif": 1}
    assert inventory.top_level_entries == ("RID2",)


def test_zip_inventory_rejects_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with ZipFile(archive, "w") as output:
        output.writestr("../outside.txt", b"unsafe")

    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        inspect_zip_archive(archive)
