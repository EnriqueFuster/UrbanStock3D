from pathlib import Path

import pytest

from urbanstock3d.vision.common import load_roof_objects_config, load_taxonomy

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config" / "vision"


def test_loads_matching_roof_object_config_and_taxonomy() -> None:
    taxonomy = load_taxonomy(CONFIG_ROOT / "taxonomy.yaml")
    config = load_roof_objects_config(CONFIG_ROOT / "roof_objects.yaml")

    config.validate_taxonomy(taxonomy)

    assert taxonomy.roof_objects == {
        "pv_panel": 0,
        "hvac_unit": 1,
        "chimney": 2,
        "elevator_overrun": 3,
        "skylight": 4,
    }
    assert config.task_type == "instance_segmentation"
    assert config.model.checkpoint == "models/pretrained/yolo26s-seg.pt"
    assert config.training.image_size == 512
    assert config.tracking.experiment == "urbanstock/roof_objects"


def test_rejects_unknown_canonical_class() -> None:
    taxonomy = load_taxonomy(CONFIG_ROOT / "taxonomy.yaml")

    with pytest.raises(ValueError, match="Unknown canonical"):
        taxonomy.require_class("courtyard")
