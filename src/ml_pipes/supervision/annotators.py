from __future__ import annotations

from collections.abc import Callable, Iterator
from numbers import Integral
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import supervision as sv

from ml_pipes.operator import Operator

from .core import Detection


def _detection_records(detections: sv.Detections) -> Iterator[Detection]:
    """Yield the public per-detection view used by formatter callbacks."""
    for xyxy, mask, confidence, class_id, tracker_id, data in detections:
        yield Detection(
            xyxy=xyxy,
            mask=mask,
            confidence=confidence,
            class_id=class_id,
            tracker_id=tracker_id,
            data=data,
        )


class _DetectionColorLookup:
    """Build Supervision's per-detection palette-index array."""

    def __init__(
        self,
        custom_color_lookup: Callable[[Detection], int] | None,
    ) -> None:
        if custom_color_lookup is not None and not callable(custom_color_lookup):
            raise TypeError("custom_color_lookup must be callable.")
        self.custom_color_lookup = custom_color_lookup

    def resolve(
        self, detections: sv.Detections
    ) -> npt.NDArray[np.int64] | None:
        if self.custom_color_lookup is None or len(detections) == 0:
            return None

        lookup = np.empty(len(detections), dtype=np.int64)
        for index, detection in enumerate(_detection_records(detections)):
            palette_index = self.custom_color_lookup(detection)
            if not isinstance(palette_index, Integral):
                raise TypeError(
                    "custom_color_lookup must return an integer palette index, "
                    f"got {type(palette_index).__name__}."
                )
            lookup[index] = palette_index
        return lookup


@Operator
class BoxAnnotator:
    def __init__(
        self,
        color: sv.Color | sv.ColorPalette | str = sv.ColorPalette.DEFAULT,
        thickness: int = 2,
        color_lookup: sv.ColorLookup = sv.ColorLookup.CLASS,
        custom_color_lookup: Callable[[Detection], int] | None = None,
    ) -> None:
        self._detection_color_lookup = _DetectionColorLookup(custom_color_lookup)
        self.annotator = sv.BoxAnnotator(
            color=color,
            thickness=thickness,
            color_lookup=color_lookup,
        )

    def __call__(
        self,
        scene: npt.NDArray[np.uint8],
        detections: sv.Detections,
    ) -> tuple[npt.NDArray[np.uint8], sv.Detections]:
        custom_color_lookup = self._detection_color_lookup.resolve(detections)
        annotated = self.annotator.annotate(
            scene=scene.copy(),
            detections=detections,
            custom_color_lookup=custom_color_lookup,
        )
        return annotated, detections


@Operator
class OrientedBoxAnnotator:
    def __init__(
        self,
        color: sv.Color | sv.ColorPalette | str = sv.ColorPalette.DEFAULT,
        thickness: int = 2,
        color_lookup: sv.ColorLookup = sv.ColorLookup.CLASS,
        custom_color_lookup: Callable[[Detection], int] | None = None,
    ) -> None:
        self._detection_color_lookup = _DetectionColorLookup(custom_color_lookup)
        self.annotator = sv.OrientedBoxAnnotator(
            color=color,
            thickness=thickness,
            color_lookup=color_lookup,
        )

    def __call__(
        self,
        scene: npt.NDArray[np.uint8],
        detections: sv.Detections,
    ) -> tuple[npt.NDArray[np.uint8], sv.Detections]:
        custom_color_lookup = self._detection_color_lookup.resolve(detections)
        annotated = self.annotator.annotate(
            scene=scene.copy(),
            detections=detections,
            custom_color_lookup=custom_color_lookup,
        )
        return annotated, detections


@Operator
class MaskAnnotator:
    def __init__(
        self,
        color: sv.Color | sv.ColorPalette | str = sv.ColorPalette.DEFAULT,
        opacity: float = 0.5,
        color_lookup: sv.ColorLookup = sv.ColorLookup.CLASS,
        custom_color_lookup: Callable[[Detection], int] | None = None,
    ) -> None:
        self._detection_color_lookup = _DetectionColorLookup(custom_color_lookup)
        self.annotator = sv.MaskAnnotator(
            color=color,
            opacity=opacity,
            color_lookup=color_lookup,
        )

    def __call__(
        self,
        scene: npt.NDArray[np.uint8],
        detections: sv.Detections,
    ) -> tuple[npt.NDArray[np.uint8], sv.Detections]:
        custom_color_lookup = self._detection_color_lookup.resolve(detections)
        annotated = self.annotator.annotate(
            scene=scene.copy(),
            detections=detections,
            custom_color_lookup=custom_color_lookup,
        )
        return annotated, detections


@Operator
class PolygonAnnotator:
    def __init__(
        self,
        color: sv.Color | sv.ColorPalette | str = sv.ColorPalette.DEFAULT,
        thickness: int = 2,
        color_lookup: sv.ColorLookup = sv.ColorLookup.CLASS,
        custom_color_lookup: Callable[[Detection], int] | None = None,
    ) -> None:
        self._detection_color_lookup = _DetectionColorLookup(custom_color_lookup)
        self.annotator = sv.PolygonAnnotator(
            color=color,
            thickness=thickness,
            color_lookup=color_lookup,
        )

    def __call__(
        self,
        scene: npt.NDArray[np.uint8],
        detections: sv.Detections,
    ) -> tuple[npt.NDArray[np.uint8], sv.Detections]:
        custom_color_lookup = self._detection_color_lookup.resolve(detections)
        annotated = self.annotator.annotate(
            scene=scene.copy(),
            detections=detections,
            custom_color_lookup=custom_color_lookup,
        )
        return annotated, detections


@Operator
class PolygonZoneAnnotator:
    def __init__(
        self,
        zone: sv.PolygonZone,
        color: sv.Color = sv.Color.WHITE,
        thickness: int = 2,
        text_color: sv.Color = sv.Color.BLACK,
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,
        display_in_zone_count: bool = True,
        opacity: float = 0.0,
        label: str | None = None,
    ) -> None:
        self.label = label
        self.annotator = sv.PolygonZoneAnnotator(
            zone=zone,
            color=color,
            thickness=thickness,
            text_color=text_color,
            text_scale=text_scale,
            text_thickness=text_thickness,
            text_padding=text_padding,
            display_in_zone_count=display_in_zone_count,
            opacity=opacity,
        )

    def __call__(
        self,
        scene: npt.NDArray[np.uint8],
        detections: sv.Detections,
    ) -> tuple[npt.NDArray[np.uint8], sv.Detections]:
        annotated = self.annotator.annotate(scene=scene.copy(), label=self.label)
        return annotated, detections


@Operator
class LineZoneAnnotator:
    def __init__(
        self,
        line_zone: sv.LineZone,
        thickness: int = 2,
        color: sv.Color = sv.Color.WHITE,
        text_thickness: int = 2,
        text_color: sv.Color = sv.Color.BLACK,
        text_scale: float = 0.5,
        text_offset: float = 1.5,
        text_padding: int = 10,
        custom_in_text: str | None = None,
        custom_out_text: str | None = None,
        display_in_count: bool = True,
        display_out_count: bool = True,
        display_text_box: bool = True,
        text_orient_to_line: bool = False,
        text_centered: bool = True,
    ) -> None:
        self.line_zone = line_zone
        self.annotator = sv.LineZoneAnnotator(
            thickness=thickness,
            color=color,
            text_thickness=text_thickness,
            text_color=text_color,
            text_scale=text_scale,
            text_offset=text_offset,
            text_padding=text_padding,
            custom_in_text=custom_in_text,
            custom_out_text=custom_out_text,
            display_in_count=display_in_count,
            display_out_count=display_out_count,
            display_text_box=display_text_box,
            text_orient_to_line=text_orient_to_line,
            text_centered=text_centered,
        )

    def __call__(
        self,
        scene: npt.NDArray[np.uint8],
        detections: sv.Detections,
    ) -> tuple[npt.NDArray[np.uint8], sv.Detections]:
        annotated = self.annotator.annotate(
            frame=scene.copy(),
            line_counter=self.line_zone,
        )
        return annotated, detections


@Operator
class FPSAnnotator:
    def __init__(
        self,
        sample_size: int = 30,
        position: sv.Position = sv.Position.TOP_LEFT,
    ) -> None:
        self.position = position
        self.monitor = sv.FPSMonitor(sample_size=sample_size)

    def reset(self) -> None:
        self.monitor.reset()

    def __call__(
        self,
        scene: npt.NDArray[np.uint8],
        detections: sv.Detections,
    ) -> tuple[npt.NDArray[np.uint8], sv.Detections]:
        self.monitor.tick()
        label = f"FPS: {self.monitor.fps:.2f}"
        annotated = sv.draw_text(
            scene=scene.copy(),
            text=label,
            text_anchor=self._text_anchor(scene, label),
            text_color=sv.Color.WHITE,
            background_color=sv.Color.BLACK,
        )
        return annotated, detections

    def _text_anchor(
        self,
        scene: npt.NDArray[np.uint8],
        label: str,
    ) -> sv.Point:
        import cv2

        (text_width, text_height), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
        )
        box_width = text_width + 20
        box_height = text_height + 20
        margin = 10
        scene_height, scene_width = scene.shape[:2]
        half_width = box_width / 2
        half_height = box_height / 2

        if self.position == sv.Position.TOP_LEFT:
            x = half_width + margin
            y = half_height + margin
        elif self.position == sv.Position.TOP_CENTER:
            x = scene_width / 2
            y = half_height + margin
        elif self.position == sv.Position.TOP_RIGHT:
            x = scene_width - half_width - margin
            y = half_height + margin
        elif self.position == sv.Position.CENTER_LEFT:
            x = half_width + margin
            y = scene_height / 2
        elif self.position in {sv.Position.CENTER, sv.Position.CENTER_OF_MASS}:
            x = scene_width / 2
            y = scene_height / 2
        elif self.position == sv.Position.CENTER_RIGHT:
            x = scene_width - half_width - margin
            y = scene_height / 2
        elif self.position == sv.Position.BOTTOM_LEFT:
            x = half_width + margin
            y = scene_height - half_height - margin
        elif self.position == sv.Position.BOTTOM_CENTER:
            x = scene_width / 2
            y = scene_height - half_height - margin
        elif self.position == sv.Position.BOTTOM_RIGHT:
            x = scene_width - half_width - margin
            y = scene_height - half_height - margin
        else:
            raise ValueError(f"Unsupported FPSAnnotator position: {self.position!r}")

        x = float(np.clip(x, half_width, max(half_width, scene_width - half_width)))
        y = float(np.clip(y, half_height, max(half_height, scene_height - half_height)))
        return sv.Point(x=x, y=y)


@Operator
class ColorAnnotator:
    def __init__(
        self,
        color: sv.Color | sv.ColorPalette | str = sv.ColorPalette.DEFAULT,
        opacity: float = 0.5,
        color_lookup: sv.ColorLookup = sv.ColorLookup.CLASS,
        custom_color_lookup: Callable[[Detection], int] | None = None,
    ) -> None:
        self._detection_color_lookup = _DetectionColorLookup(custom_color_lookup)
        self.annotator = sv.ColorAnnotator(
            color=color,
            opacity=opacity,
            color_lookup=color_lookup,
        )

    def __call__(
        self,
        scene: npt.NDArray[np.uint8],
        detections: sv.Detections,
    ) -> tuple[npt.NDArray[np.uint8], sv.Detections]:
        custom_color_lookup = self._detection_color_lookup.resolve(detections)
        annotated = self.annotator.annotate(
            scene=scene.copy(),
            detections=detections,
            custom_color_lookup=custom_color_lookup,
        )
        return annotated, detections


@Operator
class HaloAnnotator:
    def __init__(
        self,
        color: sv.Color | sv.ColorPalette | str = sv.ColorPalette.DEFAULT,
        opacity: float = 0.8,
        kernel_size: int = 40,
        color_lookup: sv.ColorLookup = sv.ColorLookup.CLASS,
        custom_color_lookup: Callable[[Detection], int] | None = None,
    ) -> None:
        self._detection_color_lookup = _DetectionColorLookup(custom_color_lookup)
        self.annotator = sv.HaloAnnotator(
            color=color,
            opacity=opacity,
            kernel_size=kernel_size,
            color_lookup=color_lookup,
        )

    def __call__(
        self,
        scene: npt.NDArray[np.uint8],
        detections: sv.Detections,
    ) -> tuple[npt.NDArray[np.uint8], sv.Detections]:
        custom_color_lookup = self._detection_color_lookup.resolve(detections)
        annotated = self.annotator.annotate(
            scene=scene.copy(),
            detections=detections,
            custom_color_lookup=custom_color_lookup,
        )
        return annotated, detections


@Operator
class EllipseAnnotator:
    def __init__(
        self,
        color: sv.Color | sv.ColorPalette | str = sv.ColorPalette.DEFAULT,
        thickness: int = 2,
        start_angle: int = -45,
        end_angle: int = 235,
        color_lookup: sv.ColorLookup = sv.ColorLookup.CLASS,
        custom_color_lookup: Callable[[Detection], int] | None = None,
    ) -> None:
        self._detection_color_lookup = _DetectionColorLookup(custom_color_lookup)
        self.annotator = sv.EllipseAnnotator(
            color=color,
            thickness=thickness,
            start_angle=start_angle,
            end_angle=end_angle,
            color_lookup=color_lookup,
        )

    def __call__(
        self,
        scene: npt.NDArray[np.uint8],
        detections: sv.Detections,
    ) -> tuple[npt.NDArray[np.uint8], sv.Detections]:
        custom_color_lookup = self._detection_color_lookup.resolve(detections)
        annotated = self.annotator.annotate(
            scene=scene.copy(),
            detections=detections,
            custom_color_lookup=custom_color_lookup,
        )
        return annotated, detections


@Operator
class BoxCornerAnnotator:
    def __init__(
        self,
        color: sv.Color | sv.ColorPalette | str = sv.ColorPalette.DEFAULT,
        thickness: int = 4,
        corner_length: int = 15,
        color_lookup: sv.ColorLookup = sv.ColorLookup.CLASS,
        custom_color_lookup: Callable[[Detection], int] | None = None,
    ) -> None:
        self._detection_color_lookup = _DetectionColorLookup(custom_color_lookup)
        self.annotator = sv.BoxCornerAnnotator(
            color=color,
            thickness=thickness,
            corner_length=corner_length,
            color_lookup=color_lookup,
        )

    def __call__(
        self,
        scene: npt.NDArray[np.uint8],
        detections: sv.Detections,
    ) -> tuple[npt.NDArray[np.uint8], sv.Detections]:
        custom_color_lookup = self._detection_color_lookup.resolve(detections)
        annotated = self.annotator.annotate(
            scene=scene.copy(),
            detections=detections,
            custom_color_lookup=custom_color_lookup,
        )
        return annotated, detections


@Operator
class CircleAnnotator:
    def __init__(
        self,
        color: sv.Color | sv.ColorPalette | str = sv.ColorPalette.DEFAULT,
        thickness: int = 2,
        color_lookup: sv.ColorLookup = sv.ColorLookup.CLASS,
        custom_color_lookup: Callable[[Detection], int] | None = None,
    ) -> None:
        self._detection_color_lookup = _DetectionColorLookup(custom_color_lookup)
        self.annotator = sv.CircleAnnotator(
            color=color,
            thickness=thickness,
            color_lookup=color_lookup,
        )

    def __call__(
        self,
        scene: npt.NDArray[np.uint8],
        detections: sv.Detections,
    ) -> tuple[npt.NDArray[np.uint8], sv.Detections]:
        custom_color_lookup = self._detection_color_lookup.resolve(detections)
        annotated = self.annotator.annotate(
            scene=scene.copy(),
            detections=detections,
            custom_color_lookup=custom_color_lookup,
        )
        return annotated, detections


@Operator
class DotAnnotator:
    def __init__(
        self,
        color: sv.Color | sv.ColorPalette | str = sv.ColorPalette.DEFAULT,
        radius: int = 4,
        position: sv.Position = sv.Position.CENTER,
        color_lookup: sv.ColorLookup = sv.ColorLookup.CLASS,
        custom_color_lookup: Callable[[Detection], int] | None = None,
        outline_thickness: int = 0,
        outline_color: sv.Color | sv.ColorPalette | str = sv.Color.BLACK,
    ) -> None:
        self._detection_color_lookup = _DetectionColorLookup(custom_color_lookup)
        self.annotator = sv.DotAnnotator(
            color=color,
            radius=radius,
            position=position,
            color_lookup=color_lookup,
            outline_thickness=outline_thickness,
            outline_color=outline_color,
        )

    def __call__(
        self,
        scene: npt.NDArray[np.uint8],
        detections: sv.Detections,
    ) -> tuple[npt.NDArray[np.uint8], sv.Detections]:
        custom_color_lookup = self._detection_color_lookup.resolve(detections)
        annotated = self.annotator.annotate(
            scene=scene.copy(),
            detections=detections,
            custom_color_lookup=custom_color_lookup,
        )
        return annotated, detections


class _LabelFormatter:
    """Build standard label text from fields on each detection."""

    def __init__(
        self,
        *,
        show_class: bool = False,
        show_confidence: bool = False,
        show_tracker_id: bool = False,
        tracker_id_prefix: str = "#",
    ) -> None:
        self.show_class = show_class
        self.show_confidence = show_confidence
        self.show_tracker_id = show_tracker_id
        self.tracker_id_prefix = tracker_id_prefix

    @staticmethod
    def _optional_class_labels(detections: sv.Detections) -> list[str] | None:
        class_names = getattr(detections, "data", {}).get("class_name")
        if class_names is not None and len(class_names) == len(detections):
            return [str(name) for name in np.asarray(class_names, dtype=object).tolist()]

        class_id = detections.class_id
        if class_id is not None and len(class_id) == len(detections):
            return [str(int(class_id_value)) for class_id_value in np.asarray(class_id, dtype=np.int32).tolist()]

        return None

    @staticmethod
    def _optional_scores(detections: sv.Detections) -> npt.NDArray[np.float32] | None:
        confidence = detections.confidence
        if confidence is None or len(confidence) != len(detections):
            return None
        return cast(npt.NDArray[np.float32], np.asarray(confidence, dtype=np.float32))

    @staticmethod
    def _optional_tracker_id(detections: sv.Detections) -> npt.NDArray[np.int32] | None:
        tracker_id = getattr(detections, "tracker_id", None)
        if tracker_id is None or len(tracker_id) != len(detections):
            return None
        return cast(npt.NDArray[np.int32], np.asarray(tracker_id, dtype=np.int32))

    def _labels(self, detections: sv.Detections) -> list[str] | None:
        if not any((self.show_class, self.show_confidence, self.show_tracker_id)):
            return None

        if len(detections) == 0:
            return []

        class_labels = self._optional_class_labels(detections) if self.show_class else None
        confidence = self._optional_scores(detections) if self.show_confidence else None
        tracker_id = self._optional_tracker_id(detections) if self.show_tracker_id else None

        rendered: list[str] = []
        for index in range(len(detections)):
            parts: list[str] = []
            if self.show_tracker_id:
                if tracker_id is None:
                    parts.append(f"{self.tracker_id_prefix}MISSING_TRACKER_ID")
                else:
                    parts.append(f"{self.tracker_id_prefix}{int(tracker_id[index])}")
            if self.show_class:
                if class_labels is None:
                    parts.append("MISSING_CLASS")
                else:
                    parts.append(class_labels[index])
            if self.show_confidence:
                if confidence is None:
                    parts.append("MISSING_CONFIDENCE")
                else:
                    parts.append(f"{float(confidence[index]):.2f}")
            rendered.append(" ".join(parts))
        return rendered

    def __call__(self, detections: sv.Detections) -> list[str] | None:
        return self._labels(detections)


class _CustomLabelFormatter:
    """Build one label from each detection with a user-provided callback."""

    def __init__(self, label_formatter: Callable[[Detection], str]) -> None:
        self.label_formatter = label_formatter

    def __call__(self, detections: sv.Detections) -> list[str]:
        labels: list[str] = []
        for detection in _detection_records(detections):
            rendered = self.label_formatter(detection)
            if not isinstance(rendered, str):
                raise TypeError(
                    "label_formatter must return str, "
                    f"got {type(rendered).__name__}."
                )
            labels.append(rendered)
        return labels


def _build_label_formatter(
    *,
    show_class: bool,
    show_confidence: bool,
    show_tracker_id: bool,
    tracker_id_prefix: str,
    custom_label_formatter: Callable[[Detection], str] | None,
) -> Callable[[sv.Detections], list[str] | None]:
    if custom_label_formatter is not None:
        if any((show_class, show_confidence, show_tracker_id)):
            raise ValueError(
                "label_formatter cannot be combined with show_class, "
                "show_confidence, or show_tracker_id."
            )
        if not callable(custom_label_formatter):
            raise TypeError("label_formatter must be callable.")
        return _CustomLabelFormatter(custom_label_formatter)

    return _LabelFormatter(
        show_class=show_class,
        show_confidence=show_confidence,
        show_tracker_id=show_tracker_id,
        tracker_id_prefix=tracker_id_prefix,
    )


@Operator
class LabelAnnotator:
    """Annotate detections with default, selected, or custom label text."""

    def __init__(
        self,
        color: sv.Color | sv.ColorPalette | str = sv.ColorPalette.DEFAULT,
        color_lookup: sv.ColorLookup = sv.ColorLookup.CLASS,
        custom_color_lookup: Callable[[Detection], int] | None = None,
        text_color: sv.Color | sv.ColorPalette | str = sv.Color.WHITE,
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,
        text_position: sv.Position = sv.Position.TOP_LEFT,
        text_offset: tuple[int, int] = (0, 0),
        border_radius: int = 0,
        smart_position: bool = False,
        max_line_length: int | None = None,
        show_class: bool = False,
        show_confidence: bool = False,
        show_tracker_id: bool = False,
        tracker_id_prefix: str = "#",
        label_formatter: Callable[[Detection], str] | None = None,
    ) -> None:
        self._detection_color_lookup = _DetectionColorLookup(custom_color_lookup)
        self._label_formatter = _build_label_formatter(
            show_class=show_class,
            show_confidence=show_confidence,
            show_tracker_id=show_tracker_id,
            tracker_id_prefix=tracker_id_prefix,
            custom_label_formatter=label_formatter,
        )
        self.annotator = sv.LabelAnnotator(
            color=color,
            color_lookup=color_lookup,
            text_color=text_color,
            text_scale=text_scale,
            text_thickness=text_thickness,
            text_padding=text_padding,
            text_position=text_position,
            text_offset=text_offset,
            border_radius=border_radius,
            smart_position=smart_position,
            max_line_length=max_line_length,
        )

    def __call__(
        self,
        scene: npt.NDArray[np.uint8],
        detections: sv.Detections,
    ) -> tuple[npt.NDArray[np.uint8], sv.Detections]:
        custom_color_lookup = self._detection_color_lookup.resolve(detections)
        annotated = self.annotator.annotate(
            scene=scene.copy(),
            detections=detections,
            custom_color_lookup=custom_color_lookup,
            labels=self._label_formatter(detections),
        )
        return annotated, detections


@Operator
class RichLabelAnnotator:
    def __init__(
        self,
        color: sv.Color | sv.ColorPalette | str = sv.ColorPalette.DEFAULT,
        color_lookup: sv.ColorLookup = sv.ColorLookup.CLASS,
        custom_color_lookup: Callable[[Detection], int] | None = None,
        text_color: sv.Color | sv.ColorPalette | str = sv.Color.WHITE,
        font_path: str | None = None,
        font_size: int = 10,
        text_padding: int = 10,
        text_position: sv.Position = sv.Position.TOP_LEFT,
        text_offset: tuple[int, int] = (0, 0),
        border_radius: int = 0,
        smart_position: bool = False,
        max_line_length: int | None = None,
        show_class: bool = False,
        show_confidence: bool = False,
        show_tracker_id: bool = False,
        tracker_id_prefix: str = "#",
        label_formatter: Callable[[Detection], str] | None = None,
    ) -> None:
        self._detection_color_lookup = _DetectionColorLookup(custom_color_lookup)
        self._label_formatter = _build_label_formatter(
            show_class=show_class,
            show_confidence=show_confidence,
            show_tracker_id=show_tracker_id,
            tracker_id_prefix=tracker_id_prefix,
            custom_label_formatter=label_formatter,
        )
        self.annotator = sv.RichLabelAnnotator(
            color=color,
            color_lookup=color_lookup,
            text_color=text_color,
            font_path=font_path,
            font_size=font_size,
            text_padding=text_padding,
            text_position=text_position,
            text_offset=text_offset,
            border_radius=border_radius,
            smart_position=smart_position,
            max_line_length=max_line_length,
        )

    def __call__(
        self,
        scene: npt.NDArray[np.uint8],
        detections: sv.Detections,
    ) -> tuple[npt.NDArray[np.uint8], sv.Detections]:
        custom_color_lookup = self._detection_color_lookup.resolve(detections)
        annotated = self.annotator.annotate(
            scene=scene.copy(),
            detections=detections,
            custom_color_lookup=custom_color_lookup,
            labels=self._label_formatter(detections),
        )
        return annotated, detections


@Operator
class IconAnnotator:
    def __init__(
        self,
        icon_resolution_wh: tuple[int, int] = (64, 64),
        icon_position: sv.Position = sv.Position.TOP_CENTER,
        offset_xy: tuple[int, int] = (0, 0),
        icon_path: str | list[str] = "",
    ) -> None:
        self.icon_path = icon_path
        self.annotator = sv.IconAnnotator(
            icon_resolution_wh=icon_resolution_wh,
            icon_position=icon_position,
            offset_xy=offset_xy,
        )

    def __call__(
        self,
        scene: npt.NDArray[np.uint8],
        detections: sv.Detections,
    ) -> tuple[npt.NDArray[np.uint8], sv.Detections]:
        annotated = self.annotator.annotate(
            scene=scene.copy(),
            detections=detections,
            icon_path=self.icon_path,
        )
        return annotated, detections


@Operator
class BlurAnnotator:
    def __init__(self, kernel_size: int | None = None) -> None:
        self.annotator = sv.BlurAnnotator(kernel_size=kernel_size)

    def __call__(
        self,
        scene: npt.NDArray[np.uint8],
        detections: sv.Detections,
    ) -> tuple[npt.NDArray[np.uint8], sv.Detections]:
        annotated = self.annotator.annotate(scene=scene.copy(), detections=detections)
        return annotated, detections


@Operator
class TraceAnnotator:
    def __init__(
        self,
        color: sv.Color | sv.ColorPalette | str = sv.ColorPalette.DEFAULT,
        position: sv.Position = sv.Position.CENTER,
        trace_length: int = 30,
        thickness: int = 2,
        smooth: bool = False,
        color_lookup: sv.ColorLookup = sv.ColorLookup.CLASS,
        custom_color_lookup: Callable[[Detection], int] | None = None,
    ) -> None:
        self._detection_color_lookup = _DetectionColorLookup(custom_color_lookup)
        self.annotator = sv.TraceAnnotator(
            color=color,
            position=position,
            trace_length=trace_length,
            thickness=thickness,
            smooth=smooth,
            color_lookup=color_lookup,
        )

    def __call__(
        self,
        scene: npt.NDArray[np.uint8],
        detections: sv.Detections,
    ) -> tuple[npt.NDArray[np.uint8], sv.Detections]:
        custom_color_lookup = self._detection_color_lookup.resolve(detections)
        annotated = self.annotator.annotate(
            scene=scene.copy(),
            detections=detections,
            custom_color_lookup=custom_color_lookup,
        )
        return annotated, detections


@Operator
class HeatMapAnnotator:
    def __init__(
        self,
        position: sv.Position = sv.Position.BOTTOM_CENTER,
        opacity: float = 0.2,
        radius: int = 40,
        kernel_size: int | None = 25,
        top_hue: int = 0,
        low_hue: int = 125,
    ) -> None:
        self.annotator = sv.HeatMapAnnotator(
            position=position,
            opacity=opacity,
            radius=radius,
            kernel_size=kernel_size,
            top_hue=top_hue,
            low_hue=low_hue,
        )

    def __call__(
        self,
        scene: npt.NDArray[np.uint8],
        detections: sv.Detections,
    ) -> tuple[npt.NDArray[np.uint8], sv.Detections]:
        annotated = self.annotator.annotate(scene=scene.copy(), detections=detections)
        return annotated, detections


@Operator
class PixelateAnnotator:
    def __init__(self, pixel_size: int | None = None) -> None:
        self.annotator = sv.PixelateAnnotator(pixel_size=pixel_size)

    def __call__(
        self,
        scene: npt.NDArray[np.uint8],
        detections: sv.Detections,
    ) -> tuple[npt.NDArray[np.uint8], sv.Detections]:
        annotated = self.annotator.annotate(scene=scene.copy(), detections=detections)
        return annotated, detections


@Operator
class TriangleAnnotator:
    def __init__(
        self,
        color: sv.Color | sv.ColorPalette | str = sv.ColorPalette.DEFAULT,
        base: int = 10,
        height: int = 10,
        position: sv.Position = sv.Position.TOP_CENTER,
        color_lookup: sv.ColorLookup = sv.ColorLookup.CLASS,
        custom_color_lookup: Callable[[Detection], int] | None = None,
        outline_thickness: int = 0,
        outline_color: sv.Color | sv.ColorPalette | str = sv.Color.BLACK,
    ) -> None:
        self._detection_color_lookup = _DetectionColorLookup(custom_color_lookup)
        self.annotator = sv.TriangleAnnotator(
            color=color,
            base=base,
            height=height,
            position=position,
            color_lookup=color_lookup,
            outline_thickness=outline_thickness,
            outline_color=outline_color,
        )

    def __call__(
        self,
        scene: npt.NDArray[np.uint8],
        detections: sv.Detections,
    ) -> tuple[npt.NDArray[np.uint8], sv.Detections]:
        custom_color_lookup = self._detection_color_lookup.resolve(detections)
        annotated = self.annotator.annotate(
            scene=scene.copy(),
            detections=detections,
            custom_color_lookup=custom_color_lookup,
        )
        return annotated, detections


@Operator
class RoundBoxAnnotator:
    def __init__(
        self,
        color: sv.Color | sv.ColorPalette | str = sv.ColorPalette.DEFAULT,
        thickness: int = 2,
        color_lookup: sv.ColorLookup = sv.ColorLookup.CLASS,
        custom_color_lookup: Callable[[Detection], int] | None = None,
        roundness: float = 0.6,
    ) -> None:
        self._detection_color_lookup = _DetectionColorLookup(custom_color_lookup)
        self.annotator = sv.RoundBoxAnnotator(
            color=color,
            thickness=thickness,
            color_lookup=color_lookup,
            roundness=roundness,
        )

    def __call__(
        self,
        scene: npt.NDArray[np.uint8],
        detections: sv.Detections,
    ) -> tuple[npt.NDArray[np.uint8], sv.Detections]:
        custom_color_lookup = self._detection_color_lookup.resolve(detections)
        annotated = self.annotator.annotate(
            scene=scene.copy(),
            detections=detections,
            custom_color_lookup=custom_color_lookup,
        )
        return annotated, detections


@Operator
class PercentageBarAnnotator:
    def __init__(
        self,
        height: int = 16,
        width: int = 80,
        color: sv.Color | sv.ColorPalette | str = sv.ColorPalette.DEFAULT,
        border_color: sv.Color | str = sv.Color.BLACK,
        position: sv.Position = sv.Position.TOP_CENTER,
        color_lookup: sv.ColorLookup = sv.ColorLookup.CLASS,
        custom_color_lookup: Callable[[Detection], int] | None = None,
        border_thickness: int | None = None,
        custom_values: Any = None,
    ) -> None:
        self.custom_values = custom_values
        self._detection_color_lookup = _DetectionColorLookup(custom_color_lookup)
        self.annotator = sv.PercentageBarAnnotator(
            height=height,
            width=width,
            color=color,
            border_color=border_color,
            position=position,
            color_lookup=color_lookup,
            border_thickness=border_thickness,
        )

    def __call__(
        self,
        scene: npt.NDArray[np.uint8],
        detections: sv.Detections,
    ) -> tuple[npt.NDArray[np.uint8], sv.Detections]:
        custom_color_lookup = self._detection_color_lookup.resolve(detections)
        annotated = self.annotator.annotate(
            scene=scene.copy(),
            detections=detections,
            custom_color_lookup=custom_color_lookup,
            custom_values=self.custom_values,
        )
        return annotated, detections


@Operator
class CropAnnotator:
    def __init__(
        self,
        position: sv.Position = sv.Position.TOP_CENTER,
        scale_factor: float = 2.0,
        border_color: sv.Color | sv.ColorPalette | str = sv.ColorPalette.DEFAULT,
        border_thickness: int = 2,
        border_color_lookup: sv.ColorLookup = sv.ColorLookup.CLASS,
        custom_color_lookup: Callable[[Detection], int] | None = None,
    ) -> None:
        self._detection_color_lookup = _DetectionColorLookup(custom_color_lookup)
        self.annotator = sv.CropAnnotator(
            position=position,
            scale_factor=scale_factor,
            border_color=border_color,
            border_thickness=border_thickness,
            border_color_lookup=border_color_lookup,
        )

    def __call__(
        self,
        scene: npt.NDArray[np.uint8],
        detections: sv.Detections,
    ) -> tuple[npt.NDArray[np.uint8], sv.Detections]:
        custom_color_lookup = self._detection_color_lookup.resolve(detections)
        annotated = self.annotator.annotate(
            scene=scene.copy(),
            detections=detections,
            custom_color_lookup=custom_color_lookup,
        )
        return annotated, detections


@Operator
class BackgroundOverlayAnnotator:
    def __init__(
        self,
        color: sv.Color = sv.Color.BLACK,
        opacity: float = 0.5,
        force_box: bool = False,
    ) -> None:
        self.annotator = sv.BackgroundOverlayAnnotator(
            color=color,
            opacity=opacity,
            force_box=force_box,
        )

    def __call__(
        self,
        scene: npt.NDArray[np.uint8],
        detections: sv.Detections,
    ) -> tuple[npt.NDArray[np.uint8], sv.Detections]:
        annotated = self.annotator.annotate(scene=scene.copy(), detections=detections)
        return annotated, detections


@Operator
class ComparisonAnnotator:
    def __init__(
        self,
        color_1: sv.Color = sv.Color.RED,
        color_2: sv.Color = sv.Color.GREEN,
        color_overlap: sv.Color = sv.Color.BLUE,
        *,
        opacity: float = 0.75,
        label_1: str = "",
        label_2: str = "",
        label_overlap: str = "",
        label_scale: float = 1.0,
    ) -> None:
        self.annotator = sv.ComparisonAnnotator(
            color_1=color_1,
            color_2=color_2,
            color_overlap=color_overlap,
            opacity=opacity,
            label_1=label_1,
            label_2=label_2,
            label_overlap=label_overlap,
            label_scale=label_scale,
        )

    def __call__(
        self,
        scene: npt.NDArray[np.uint8],
        detections_1: sv.Detections,
        detections_2: sv.Detections,
    ) -> tuple[npt.NDArray[np.uint8], sv.Detections, sv.Detections]:
        annotated = self.annotator.annotate(
            scene=scene.copy(),
            detections_1=detections_1,
            detections_2=detections_2,
        )
        return annotated, detections_1, detections_2
