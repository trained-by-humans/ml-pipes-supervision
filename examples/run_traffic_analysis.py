"""Count tracked vehicle routes through Supervision's traffic-analysis video.

Run from the repo root:
    python examples/run_traffic_analysis.py
    python examples/run_traffic_analysis.py --input path/to/video.mov
    python examples/run_traffic_analysis.py --output traffic-routes.mp4

The default input is the public ``traffic_analysis.mov`` video used by
Supervision's traffic-analysis example. The matching custom YOLO weights are
also downloaded when ``--weights`` is omitted. Install the optional tools:

    python -m pip install gdown "ml-pipes-ultralytics @ git+https://github.com/requiem4machines/ml-pipes-ultralytics.git"
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import supervision as sv

try:
    from common import ASSETS_DIR
except ModuleNotFoundError:  # Supports importing the local operators in tests.
    from examples.common import ASSETS_DIR
from ml_pipes.core import Pipeline
from ml_pipes.operator import Operator
from ml_pipes.standard import Pick, Recall, Select, Store
from ml_pipes.supervision import (
    BoxAnnotator,
    Detection,
    Detections,
    ImageWindow,
    LabelAnnotator,
    TraceAnnotator,
)
from ml_pipes.supervision.trackers import ByteTrack

RouteKey: TypeAlias = tuple[int, int]
Polygons: TypeAlias = tuple[npt.NDArray[np.int64], ...]
Zones: TypeAlias = tuple[sv.PolygonZone, ...]

DEFAULT_VIDEO_NAME = "traffic_analysis.mov"
DEFAULT_WEIGHTS_NAME = "traffic_analysis.pt"
DEFAULT_OUTPUT_NAME = "traffic-route-counting-result.mp4"
TRAFFIC_ANALYSIS_VIDEO_ID = "1qadBd7lgpediafCpL_yedGjQPk-FLK-W"
TRAFFIC_ANALYSIS_WEIGHTS_ID = "1y-IfToCjRXa3ZdC1JpnKRopC7mcQW-5z"
# ``traffic_analysis.pt`` defines bus, car, truck, and van as IDs 0 through 3.
VEHICLE_CLASS_IDS = frozenset({0, 1, 2, 3})
COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])

ENTRY_ZONE_POLYGONS = (
    np.asarray([[592, 282], [900, 282], [900, 82], [592, 82]], dtype=np.int64),
    np.asarray([[950, 860], [1250, 860], [1250, 1060], [950, 1060]], dtype=np.int64),
    np.asarray([[592, 582], [592, 860], [392, 860], [392, 582]], dtype=np.int64),
    np.asarray([[1250, 282], [1250, 530], [1450, 530], [1450, 282]], dtype=np.int64),
)
EXIT_ZONE_POLYGONS = (
    np.asarray([[950, 282], [1250, 282], [1250, 82], [950, 82]], dtype=np.int64),
    np.asarray([[592, 860], [900, 860], [900, 1060], [592, 1060]], dtype=np.int64),
    np.asarray([[592, 282], [592, 550], [392, 550], [392, 282]], dtype=np.int64),
    np.asarray([[1250, 860], [1250, 560], [1450, 560], [1450, 860]], dtype=np.int64),
)


def _copy_detections_with_data(
    detections: sv.Detections,
    field: str,
    values: npt.NDArray[np.int32],
) -> sv.Detections:
    data = dict(detections.data)
    data[field] = values
    return sv.Detections(
        xyxy=detections.xyxy.copy(),
        mask=detections.mask,
        confidence=None
        if detections.confidence is None
        else detections.confidence.copy(),
        class_id=None if detections.class_id is None else detections.class_id.copy(),
        tracker_id=None
        if detections.tracker_id is None
        else detections.tracker_id.copy(),
        data=data,
        metadata=dict(detections.metadata),
    )


def _zone_ids(detections: sv.Detections, field: str) -> npt.NDArray[np.int32]:
    values = detections.data.get(field)
    if values is None:
        raise ValueError(f"Detections are missing the {field!r} zone data field.")
    values_array = np.asarray(values)
    if len(values_array) != len(detections):
        raise ValueError(
            f"The {field!r} zone data field must have one value per detection."
        )
    return np.asarray(values_array, dtype=np.int32)


@Operator
class MarkZone:
    """Attach the current matching zone ID without filtering detections.

    The first matching zone wins when zones overlap. In normal traffic layouts
    the configured zones are disjoint, so every detection has at most one ID.
    """

    def __init__(self, zones: Zones, field: str) -> None:
        if not zones:
            raise ValueError("MarkZone requires at least one zone.")
        if not field:
            raise ValueError("MarkZone field must not be empty.")
        self.zones = zones
        self.field = field

    def __call__(self, detections: sv.Detections) -> sv.Detections:
        marked = np.full(len(detections), -1, dtype=np.int32)
        for zone_id, zone in enumerate(self.zones):
            is_in_zone = zone.trigger(detections)
            marked[(marked == -1) & is_in_zone] = zone_id
        return _copy_detections_with_data(detections, self.field, marked)


@Operator
class TrackingRouteCounter:
    """Count unique tracked routes from marked source and destination zones.

    ``destination_field=None`` turns one zone field into a bidirectional set
    of gateways: the first zone visited is the origin and a later, different
    zone becomes its destination.
    """

    def __init__(
        self,
        origin_field: str,
        destination_field: str | None = None,
        output_field: str = "route_origin",
    ) -> None:
        if not origin_field:
            raise ValueError("origin_field must not be empty.")
        if destination_field is not None and not destination_field:
            raise ValueError("destination_field must not be empty when provided.")
        if not output_field:
            raise ValueError("output_field must not be empty.")
        self.origin_field = origin_field
        self.is_bidirectional = destination_field is None
        self.destination_field = destination_field or origin_field
        self.output_field = output_field
        self._origin_by_tracker_id: dict[int, int] = {}
        self._tracker_ids_by_route: dict[RouteKey, set[int]] = {}

    @property
    def route_counts(self) -> dict[RouteKey, int]:
        """Return a snapshot of unique tracked-object totals by route."""
        return {
            route: len(tracker_ids)
            for route, tracker_ids in self._tracker_ids_by_route.items()
        }

    def reset(self) -> None:
        """Clear origins and route totals before processing a new stream."""
        self._origin_by_tracker_id.clear()
        self._tracker_ids_by_route.clear()

    def __call__(self, detections: sv.Detections) -> sv.Detections:
        if detections.tracker_id is None:
            raise ValueError("TrackingRouteCounter requires detections with tracker_id values.")

        origin_marks = _zone_ids(detections, self.origin_field)
        destination_marks = _zone_ids(detections, self.destination_field)
        tracker_ids = np.asarray(detections.tracker_id, dtype=np.int64)
        route_origins = np.full(len(detections), -1, dtype=np.int32)

        for index, tracker_id_value in enumerate(tracker_ids):
            tracker_id = int(tracker_id_value)
            if tracker_id < 0:
                continue

            marked_origin = int(origin_marks[index])
            if marked_origin >= 0:
                self._origin_by_tracker_id.setdefault(tracker_id, marked_origin)

            route_origin = self._origin_by_tracker_id.get(tracker_id, -1)
            route_origins[index] = route_origin
            marked_destination = int(destination_marks[index])
            if (
                route_origin >= 0
                and marked_destination >= 0
                and (not self.is_bidirectional or marked_destination != route_origin)
            ):
                self._tracker_ids_by_route.setdefault(
                    (route_origin, marked_destination), set()
                ).add(tracker_id)

        return _copy_detections_with_data(detections, self.output_field, route_origins)


def route_origin_color_lookup(detection: Detection) -> int:
    """Choose a route-palette index after ``has_route_origin`` filters detections."""
    return int(detection.data["route_origin"])


@Operator
class RouteCountAnnotator:
    """Draw route zones and totals already maintained by a route counter."""

    def __init__(
        self,
        entry_polygons: Polygons,
        exit_polygons: Polygons,
        route_counter: TrackingRouteCounter,
        color: sv.ColorPalette = COLORS,
        thickness: int = 2,
        text_scale: float = 0.7,
        text_thickness: int = 2,
    ) -> None:
        self.entry_polygons = entry_polygons
        self.exit_polygons = exit_polygons
        self.route_counter = route_counter
        self.color = color
        self.thickness = thickness
        self.text_scale = text_scale
        self.text_thickness = text_thickness

    def __call__(
        self,
        scene: npt.NDArray[np.uint8],
        detections: sv.Detections,
    ) -> tuple[npt.NDArray[np.uint8], sv.Detections]:
        annotated = scene.copy()
        for zone_id, polygon in enumerate(self.entry_polygons):
            annotated = sv.draw_polygon(
                scene=annotated,
                polygon=polygon,
                color=self.color.by_idx(zone_id),
                thickness=self.thickness,
            )
        for zone_id, polygon in enumerate(self.exit_polygons):
            annotated = sv.draw_polygon(
                scene=annotated,
                polygon=polygon,
                color=self.color.by_idx(zone_id),
                thickness=self.thickness,
            )

        for destination_id, destination_polygon in enumerate(self.exit_polygons):
            destination_center = sv.get_polygon_center(destination_polygon)
            route_rows = sorted(
                (origin_id, count)
                for (origin_id, route_destination_id), count in self.route_counter.route_counts.items()
                if route_destination_id == destination_id
            )
            for row, (origin_id, count) in enumerate(route_rows):
                label = f"Entry {origin_id + 1}: {count}"
                text_anchor = sv.Point(
                    x=destination_center.x,
                    y=destination_center.y + (40 * row),
                )
                annotated = sv.draw_text(
                    scene=annotated,
                    text=label,
                    text_anchor=text_anchor,
                    text_color=sv.Color.BLACK,
                    background_color=self.color.by_idx(origin_id),
                    text_scale=self.text_scale,
                    text_thickness=self.text_thickness,
                )
        return annotated, detections


def build_zones(
    polygons: Polygons,
) -> Zones:
    return tuple(
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=[sv.Position.CENTER],
        )
        for polygon in polygons
    )


def keep_vehicle_detections(detections: sv.Detections) -> npt.NDArray[np.bool_]:
    if detections.class_id is None:
        raise ValueError("Vehicle filtering requires detection class IDs.")
    return np.isin(np.asarray(detections.class_id), tuple(VEHICLE_CLASS_IDS))


def has_route_origin(detections: sv.Detections) -> npt.NDArray[np.bool_]:
    return _zone_ids(detections, "route_origin") >= 0


def build_frame_pipeline(
    weights_path: Path,
    entry_polygons: Polygons,
    exit_polygons: Polygons,
) -> Pipeline[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    try:
        from ml_pipes.ultralytics import yolo
    except ImportError as error:
        raise RuntimeError(
            "Traffic analysis requires ml-pipes-ultralytics. Install it with "
            "'python -m pip install \"ml-pipes-ultralytics @ "
            "git+https://github.com/requiem4machines/ml-pipes-ultralytics.git\"'."
        ) from error

    route_counter = TrackingRouteCounter("entry_zone", "exit_zone")

    return Pipeline(
        [
            Store("source_frame"),
            yolo.Predict(model=weights_path, conf=0.3, iou=0.7),
            Select(0),
            Detections.FromUltralytics(),
            Detections.Filter(keep_vehicle_detections),
            ByteTrack(),
            MarkZone(build_zones(entry_polygons), "entry_zone"),
            MarkZone(build_zones(exit_polygons), "exit_zone"),
            route_counter,
            Detections.Filter(has_route_origin),
            Recall("source_frame", prepend=True),
            TraceAnnotator(thickness=2, color=COLORS, custom_color_lookup=route_origin_color_lookup),
            BoxAnnotator(color=COLORS, custom_color_lookup=route_origin_color_lookup),
            LabelAnnotator(show_tracker_id=True, color=COLORS, custom_color_lookup=route_origin_color_lookup),
            RouteCountAnnotator(entry_polygons, exit_polygons, route_counter),
            ImageWindow("Traffic Route Counting", at=0),
            Pick(0),
        ],
        auto_validate=True,
    )


def resolve_asset_path(
    path: Path | None,
    default_name: str,
    google_drive_id: str,
    asset_description: str,
    option_name: str,
) -> Path:
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"{asset_description} not found: {path}")
        return path

    default_path = ASSETS_DIR / default_name
    if default_path.exists():
        return default_path
    try:
        import gdown
    except ImportError as error:
        raise RuntimeError(
            f"The default {asset_description.lower()} needs gdown. Install it with "
            f"'python -m pip install gdown', or provide --{option_name} PATH."
        ) from error

    default_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {default_name} -> {default_path}", file=sys.stderr)
    downloaded_path = gdown.download(
        id=google_drive_id,
        output=str(default_path),
        quiet=False,
    )
    if downloaded_path is None or not default_path.exists():
        raise RuntimeError(f"Could not download the default {asset_description.lower()} to {default_path}.")
    return default_path


def resolve_input_path(input_path: Path | None) -> Path:
    return resolve_asset_path(
        input_path,
        DEFAULT_VIDEO_NAME,
        TRAFFIC_ANALYSIS_VIDEO_ID,
        "input video",
        "input",
    )


def resolve_weights_path(weights_path: Path | None) -> Path:
    return resolve_asset_path(
        weights_path,
        DEFAULT_WEIGHTS_NAME,
        TRAFFIC_ANALYSIS_WEIGHTS_ID,
        "traffic-analysis weights",
        "weights",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Video to process. Defaults to Supervision's traffic-analysis video.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="YOLO weights. Defaults to Supervision's traffic-analysis weights.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ASSETS_DIR / DEFAULT_OUTPUT_NAME,
        help="Annotated MP4 path. Defaults under examples/.example_assets.",
    )
    args = parser.parse_args()

    try:
        input_path = resolve_input_path(args.input)
        weights_path = resolve_weights_path(args.weights)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    try:
        pipeline = build_frame_pipeline(
            weights_path=weights_path,
            entry_polygons=ENTRY_ZONE_POLYGONS,
            exit_polygons=EXIT_ZONE_POLYGONS,
        )
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    pipeline.validate()
    pipeline.describe()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    sv.process_video(
        source_path=str(input_path),
        target_path=str(args.output),
        callback=lambda frame, _: pipeline(frame),
    )
    print(f"Saved route-counting video to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
