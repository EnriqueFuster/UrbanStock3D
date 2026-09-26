from pathlib import Path

import pytest

from urbanstock3d.vision.common import load_taxonomy
from urbanstock3d.vision.roof_objects.datasets import (
    audit_rid2,
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
    assert not audit.download_allowed
    assert not audit.ready_for_conversion


def test_archive_verifier_rejects_unpinned_file(tmp_path: Path) -> None:
    source = load_rid2_source(CONFIG_ROOT / "datasets" / "rid2.yaml")
    archive = tmp_path / source.archive_name
    archive.write_bytes(b"not RID2")

    with pytest.raises(ValueError, match="checksum"):
        verify_rid2_archive(archive, source)
