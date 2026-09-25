from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import supervision as sv
from supervision.detection.tools.inference_slicer import move_detections

from ml_pipes.operator import Operator
from ml_pipes.tensor import TensorRegistry
from ml_pipes.vision import ImagePayload, TileRect


@dataclass(frozen=True)
class Detection:
    """One Supervision detection and its per-detection data."""

    xyxy: npt.NDArray[np.number]
    mask: npt.NDArray[np.bool_] | None
    confidence: np.generic | None
    class_id: np.generic | None
    tracker_id: np.generic | None
    data: Mapping[str, Any]


@Operator
class ImageToArray:
    """Convert an ``ImagePayload`` to a BGR HWC uint8 array."""

    def __call__(self, image: ImagePayload) -> npt.NDArray[np.uint8]:
        if image.layout != "HWC":
            raise ValueError(f"ImageToArray expects HWC images, got {image.layout!r}")
        if image.array.ndim != 3 or image.array.shape[2] != 3:
            raise ValueError(
                "ImageToArray expects three-channel HWC images, "
                f"got shape {image.array.shape!r}"
            )
        if image.array.dtype != np.uint8:
            raise ValueError(
                "ImageToArray expects uint8 images, "
                f"got {image.array.dtype!s}"
            )

        array = np.ascontiguousarray(image.array)
        if image.color_space == "BGR":
            return cast(npt.NDArray[np.uint8], array.copy())
        if image.color_space == "RGB":
            return cast(npt.NDArray[np.uint8], array[..., ::-1].copy())
        raise ValueError(f"ImageToArray expects BGR or RGB images, got {image.color_space!r}")


@Operator
class DetectionsFromInference:
    def __init__(self, *, compact_masks: bool = False) -> None:
        self.compact_masks = compact_masks

    def __call__(self, roboflow_result: dict[str, Any] | Any) -> sv.Detections:
        return sv.Detections.from_inference(
            roboflow_result,
            compact_masks=self.compact_masks,
        )


@Operator
class DetectionsFromUltralytics:
    def __call__(self, ultralytics_result: Any) -> sv.Detections:
        return sv.Detections.from_ultralytics(ultralytics_result)


@Operator
class DetectionsFromTensorRegistry:
    """Convert named detection tensors into Supervision's data container."""

    def __init__(
        self,
        boxes: str = "boxes",
        scores: str = "scores",
        classes: str = "classes",
        masks: str | None = None,
    ) -> None:
        self.boxes = boxes
        self.scores = scores
        self.classes = classes
        self.masks = masks

    def __call__(self, registry: TensorRegistry) -> sv.Detections:
        return sv.Detections(
            xyxy=registry[self.boxes],
            confidence=registry[self.scores],
            class_id=registry[self.classes],
            mask=None if self.masks is None else registry[self.masks],
        )


@Operator
class DetectionsFilter:
    """Apply a custom filter to a :class:`supervision.Detections` instance.

    The filter callable is deliberately kept separate from ``Map`` so that
    pipelines retain the concrete ``sv.Detections`` input/output contract when
    using inline lambdas. It may return filtered detections directly or a
    boolean mask used to select the filtered detections.
    """

    def __init__(
        self,
        filter_fn: Callable[
            [sv.Detections], sv.Detections | npt.NDArray[np.bool_] | bool
        ],
    ) -> None:
        self.filter_fn = filter_fn

    def __call__(self, detections: sv.Detections) -> sv.Detections:
        filtered = self.filter_fn(detections)
        if isinstance(filtered, sv.Detections):
            return filtered
        if isinstance(filtered, (bool, np.bool_)):
            filtered = np.full(len(detections), filtered, dtype=bool)
        if isinstance(filtered, np.ndarray) and filtered.dtype == bool:
            return cast(
                sv.Detections,
                detections[cast(npt.NDArray[np.generic], filtered)],
            )
        raise TypeError(
            "Detections.Filter callback must return supervision.Detections or "
            f"a boolean NumPy mask, got {type(filtered).__name__}."
        )


@Operator
class DetectionsNMS:
    def __init__(
        self,
        threshold: float = 0.5,
        class_agnostic: bool = False,
        overlap_metric: sv.OverlapMetric = sv.OverlapMetric.IOU,
    ) -> None:
        self.threshold = threshold
        self.class_agnostic = class_agnostic
        self.overlap_metric = overlap_metric

    def __call__(self, detections: sv.Detections) -> sv.Detections:
        return detections.with_nms(
            threshold=self.threshold,
            class_agnostic=self.class_agnostic,
            overlap_metric=self.overlap_metric,
        )


@Operator
class DetectionsNMM:
    def __init__(
        self,
        iou_threshold: float = 0.5,
        class_agnostic: bool = False,
        overlap_metric: sv.OverlapMetric = sv.OverlapMetric.IOU,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.class_agnostic = class_agnostic
        self.overlap_metric = overlap_metric

    def __call__(self, detections: sv.Detections) -> sv.Detections:
        return detections.with_nmm(
            threshold=self.iou_threshold,
            class_agnostic=self.class_agnostic,
            overlap_metric=self.overlap_metric,
        )


@Operator
class DetectionsSmoother:
    def __init__(self, length: int = 5) -> None:
        self.smoother = sv.DetectionsSmoother(length=length)

    def update(self, detections: sv.Detections) -> sv.Detections:
        return self.smoother.update_with_detections(detections)

    def reset(self) -> None:
        self.smoother.reset()

    def __call__(self, detections: sv.Detections) -> sv.Detections:
        return self.update(detections)


@Operator
class DetectionsStitch:
    def __call__(
        self,
        detections: list[sv.Detections],
        tile_rects: list[TileRect],
    ) -> sv.Detections:
        if len(detections) != len(tile_rects):
            raise ValueError(
                "DetectionsStitch requires one tile rectangle for each detection result."
            )
        if not tile_rects:
            return sv.Detections.empty()

        resolution_wh = (
            max(rect.x2 for rect in tile_rects),
            max(rect.y2 for rect in tile_rects),
        )
        repositioned = [
            move_detections(
                detections=tile_detections,
                offset=np.asarray((rect.x1, rect.y1), dtype=np.int32),
                resolution_wh=resolution_wh,
            )
            for tile_detections, rect in zip(detections, tile_rects)
        ]
        return sv.Detections.merge(repositioned)
