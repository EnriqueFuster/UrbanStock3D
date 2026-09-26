"""Runtime boundary implemented by future segmentation backends."""

from pathlib import Path
from typing import Protocol

from urbanstock3d.vision.models import RoofObjectInstance


class RoofObjectSegmenter(Protocol):
    """Backend-neutral roof-object inference interface."""

    model_name: str
    model_version: str

    def predict(self, image: Path) -> tuple[RoofObjectInstance, ...]: ...
