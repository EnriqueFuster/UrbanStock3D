"""Runtime boundary and Ultralytics backend for roof-object segmentation."""

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import rasterio
from PIL import Image

from urbanstock3d.vision.common import load_taxonomy
from urbanstock3d.vision.models import (
    AssessmentStatus,
    ModelProvenance,
    RoofObjectInstance,
    RoofVisionResult,
)


class RoofObjectSegmenter(Protocol):
    """Backend-neutral roof-object inference interface."""

    model_name: str
    model_version: str

    def predict(self, image: Path) -> tuple[RoofObjectInstance, ...]: ...


def _array(value: Any) -> np.ndarray:
    """Convert a NumPy or Torch-like value without importing Torch at module load."""
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class UltralyticsRoofObjectSegmenter:
    """Run a trained YOLO segmentation checkpoint on a georeferenced raster."""

    model_name = "urbanstock-roof-objects"

    def __init__(
        self,
        weights: Path,
        output_dir: Path,
        *,
        confidence: float = 0.25,
        image_size: int = 512,
        device: int | str = 0,
        model: Any | None = None,
    ) -> None:
        if not weights.exists():
            raise FileNotFoundError(f"Missing roof-object checkpoint: {weights}")
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between zero and one")
        self.model_version = weights.parents[1].name
        self.output_dir = output_dir
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        if model is None:
            from ultralytics import YOLO  # type: ignore[attr-defined]

            model = YOLO(weights)
        self._model: Any = model
        self._taxonomy = load_taxonomy(Path("config/vision/taxonomy.yaml"))

    def predict(self, image: Path) -> tuple[RoofObjectInstance, ...]:
        """Persist binary masks and return one structured object per prediction."""
        with rasterio.open(image) as raster:
            if raster.crs is None:
                raise ValueError("Roof-object inference requires a georeferenced input raster")
            rgb = np.moveaxis(raster.read((1, 2, 3)), 0, 2)
            profile = raster.profile.copy()
            pixel_area_m2 = abs(raster.transform.a * raster.transform.e)

        predictions: list[Any] = list(
            self._model.predict(
                rgb,
                imgsz=self.image_size,
                conf=self.confidence,
                device=self.device,
                verbose=False,
            )
        )
        if len(predictions) != 1:
            raise ValueError("Expected one prediction result for one input image")
        result = predictions[0]
        if result.masks is None or result.boxes is None:
            return ()

        masks = _array(result.masks.data)
        class_ids = _array(result.boxes.cls).astype(int)
        confidences = _array(result.boxes.conf)
        if not (len(masks) == len(class_ids) == len(confidences)):
            raise ValueError("Prediction masks and boxes have inconsistent lengths")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        instances: list[RoofObjectInstance] = []
        for index, (mask, class_id, score) in enumerate(
            zip(masks, class_ids, confidences, strict=True), start=1
        ):
            class_name = str(result.names[int(class_id)])
            self._taxonomy.require_class(class_name)
            binary_mask = np.asarray(mask >= 0.5, dtype=np.uint8)
            target_shape = (profile["height"], profile["width"])
            if binary_mask.shape != target_shape:
                binary_mask = np.asarray(
                    Image.fromarray(binary_mask).resize(
                        (target_shape[1], target_shape[0]), resample=Image.Resampling.NEAREST
                    )
                )
            pixel_count = int(binary_mask.sum())
            if pixel_count == 0:
                continue

            object_id = f"{class_name}-{index:03d}"
            mask_path = self.output_dir / f"{object_id}.tif"
            mask_profile = {
                **profile,
                "driver": "GTiff",
                "count": 1,
                "dtype": "uint8",
                "compress": "deflate",
            }
            with rasterio.open(mask_path, "w", **mask_profile) as destination:
                destination.write(binary_mask, 1)

            instances.append(
                RoofObjectInstance(
                    object_id=object_id,
                    class_name=class_name,
                    confidence=float(score),
                    mask_artifact=mask_path.as_posix(),
                    pixel_area=pixel_count,
                    area_m2=pixel_count * pixel_area_m2,
                    provenance=ModelProvenance(
                        model_name=self.model_name,
                        model_version=self.model_version,
                        taxonomy_version=self._taxonomy.version,
                        source_image=image.as_posix(),
                    ),
                    quality_flags=("PNOA_DOMAIN_NOT_VALIDATED", "NOT_CLIPPED_TO_FOOTPRINT"),
                )
            )
        return tuple(instances)


def infer_roof_objects(
    building_id: str, image: Path, segmenter: RoofObjectSegmenter, output_path: Path
) -> RoofVisionResult:
    """Run one backend and persist the API-ready experimental result."""
    objects = segmenter.predict(image)
    result = RoofVisionResult(
        building_id=building_id,
        status=AssessmentStatus.EXPERIMENTAL,
        objects=objects,
        warnings=("RID2 baseline has not been validated or fine-tuned on Spanish PNOA imagery.",),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return result
