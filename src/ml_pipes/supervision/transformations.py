"""Detection-aware adapters for Supervision's box and crop helpers."""
from __future__ import annotations

from dataclasses import replace
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import supervision as sv

from ml_pipes.operator import Operator
from ml_pipes.vision import ImagePayload


ImageInput: TypeAlias = npt.NDArray[np.uint8] | ImagePayload
BoxRounding: TypeAlias = Literal["none", "nearest", "outward", "inward"]
InvalidBoxPolicy: TypeAlias = Literal["keep", "drop", "raise"]


def _image_array(image: ImageInput) -> tuple[npt.NDArray[np.uint8], str]:
    array = image.array if isinstance(image, ImagePayload) else image
    color_space = image.color_space if isinstance(image, ImagePayload) else "BGR"
    if (
        not isinstance(array, np.ndarray)
        or array.ndim != 3
        or array.shape[2] != 3
        or array.dtype != np.uint8
        or min(array.shape[:2]) < 1
        or color_space not in ("RGB", "BGR")
        or (isinstance(image, ImagePayload) and image.layout != "HWC")
    ):
        raise ValueError("Expected a non-empty RGB/BGR HWC uint8 image with three channels.")
    return array, color_space


def _valid_boxes(boxes: npt.NDArray[np.number]) -> npt.NDArray[np.bool_]:
    return (
        np.isfinite(boxes).all(axis=1)
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
    )


@Operator
class DetectionsPadBoxes:
    """Pad ``xyxy`` by pixels per side, without clipping or moving other geometry."""

    def __init__(self, px: int, py: int | None = None) -> None:
        self.px = px
        self.py = py

    def __call__(self, detections: sv.Detections) -> sv.Detections:
        return replace(detections, xyxy=sv.pad_boxes(detections.xyxy, px=self.px, py=self.py))


@Operator
class DetectionsScaleBoxes:
    """Scale box dimensions around their centers; omit y to use the x factor."""

    def __init__(self, factor_x: float, factor_y: float | None = None) -> None:
        factor_y = factor_x if factor_y is None else factor_y
        if not all(np.isfinite(factor) and factor > 0 for factor in (factor_x, factor_y)):
            raise ValueError("Scale factors must be finite and positive.")
        self.factor_x = float(factor_x)
        self.factor_y = float(factor_y)

    def __call__(self, detections: sv.Detections) -> sv.Detections:
        boxes = sv.scale_boxes(detections.xyxy, factor=self.factor_x)
        if self.factor_y != self.factor_x:
            vertical = sv.scale_boxes(detections.xyxy, factor=self.factor_y)
            boxes[:, [1, 3]] = vertical[:, [1, 3]]
        return replace(detections, xyxy=boxes)


@Operator
class DetectionsMoveBoxes:
    """Translate ``xyxy`` by (dx, dy); masks and other geometry stay unchanged."""

    def __init__(self, offset: tuple[int, int]) -> None:
        self.offset = offset

    def __call__(self, detections: sv.Detections) -> sv.Detections:
        return replace(detections, xyxy=sv.move_boxes(detections.xyxy, offset=self.offset))


@Operator
class DetectionsClipBoxes:
    """Clip boxes to the image, optionally round edges and handle invalid rows.

    Defaults match ``sv.clip_boxes``: no rounding and no row removal. ``drop``
    and ``raise`` reject nonfinite/nonpositive input boxes, as well as boxes
    that become empty after clipping or rounding. All per-row fields follow
    dropped boxes. Only ``xyxy`` geometry is changed; the image is passed through.
    """

    def __init__(
        self,
        *,
        rounding: BoxRounding = "none",
        invalid: InvalidBoxPolicy = "keep",
    ) -> None:
        if rounding not in ("none", "nearest", "outward", "inward"):
            raise ValueError("rounding must be 'none', 'nearest', 'outward', or 'inward'.")
        if invalid not in ("keep", "drop", "raise"):
            raise ValueError("invalid must be 'keep', 'drop', or 'raise'.")
        self.rounding = rounding
        self.invalid = invalid

    def __call__(
        self, image: ImageInput, detections: sv.Detections,
    ) -> tuple[ImageInput, sv.Detections]:
        array, _ = _image_array(image)
        height, width = array.shape[:2]
        boxes = sv.clip_boxes(detections.xyxy, resolution_wh=(width, height))
        if self.rounding == "nearest":
            boxes = boxes.round()
        elif self.rounding == "outward":
            boxes[:, :2] = np.floor(boxes[:, :2])
            boxes[:, 2:] = np.ceil(boxes[:, 2:])
        elif self.rounding == "inward":
            boxes[:, :2] = np.ceil(boxes[:, :2])
            boxes[:, 2:] = np.floor(boxes[:, 2:])

        if self.invalid != "keep":
            valid = _valid_boxes(detections.xyxy) & _valid_boxes(boxes)
            if self.invalid == "raise" and not valid.all():
                raise ValueError("Clipping produced invalid boxes (nonfinite or nonpositive area).")
            if self.invalid == "drop":
                detections, boxes = detections[valid], boxes[valid]
        return image, replace(detections, xyxy=boxes)


@Operator
class DetectionsCrop:
    """Extract aligned image regions, retaining source-coordinate detections.

    Like ``sv.crop_image``, round to nearest pixels and clamp to image bounds.
    No rows are silently removed: invalid or pixel-empty regions raise. Use
    ``Detections.ClipBoxes(rounding=..., invalid='drop')`` first to filter them.
    Default crops are independent copies; ``copy=False`` permits image views.
    """

    def __init__(self, *, copy: bool = True) -> None:
        self.copy = copy

    def __call__(
        self, image: ImageInput, detections: sv.Detections,
    ) -> tuple[sv.Detections, list[ImagePayload]]:
        array, color_space = _image_array(image)
        height, width = array.shape[:2]
        # Clamp before integer conversion to avoid overflow on far-outside boxes.
        boxes = sv.clip_boxes(detections.xyxy, resolution_wh=(width, height)).round()
        if not (_valid_boxes(detections.xyxy) & _valid_boxes(boxes)).all():
            raise ValueError(
                "Crop requires finite, positive, non-empty pixel regions; use "
                "Detections.ClipBoxes(rounding=..., invalid='drop') first."
            )
        crops = []
        for box in boxes:
            crop = sv.crop_image(array, xyxy=box)
            crops.append(ImagePayload(
                array=crop.copy() if self.copy else crop,
                color_space=color_space,
            ))
        return detections, crops


@Operator
class DetectionsMerge:
    """Concatenate detection sets using ``sv.Detections.merge`` (not box union)."""

    def __call__(self, detections: list[sv.Detections]) -> sv.Detections:
        return sv.Detections.merge(detections)
