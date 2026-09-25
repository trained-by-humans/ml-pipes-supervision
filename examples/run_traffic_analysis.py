"""Analyze tracked vehicle zone visits in Supervision's traffic-analysis video.

Run from the repo root:
    python examples/run_traffic_analysis.py
    python examples/run_traffic_analysis.py --input path/to/video.mov
    python examples/run_traffic_analysis.py --output traffic-zone-visits.mp4

The default input is the public ``traffic_analysis.mov`` video used by
Supervision's traffic-analysis example. The matching custom YOLO weights are
also downloaded when ``--weights`` is omitted. Install the optional tools:

    python -m pip install gdown "ml-pipes-ultralytics @ git+https://github.com/requiem4machines/ml-pipes-ultralytics.git"
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
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

Polygons: TypeAlias = tuple[npt.NDArray[np.int64], ...]
Zones: TypeAlias = tuple[sv.PolygonZone, ...]
ZoneVisits: TypeAlias = tuple[int, ...]

DEFAULT_VIDEO_NAME = "traffic_analysis.mov"
DEFAULT_WEIGHTS_NAME = "traffic_analysis.pt"
DEFAULT_OUTPUT_NAME = "traffic-zone-visit-analysis-result.mp4"
TRAFFIC_ANALYSIS_VIDEO_ID = "1qadBd7lgpediafCpL_yedGjQPk-FLK-W"
TRAFFIC_ANALYSIS_WEIGHTS_ID = "1y-IfToCjRXa3ZdC1JpnKRopC7mcQW-5z"
# ``traffic_analysis.pt`` defines bus, car, truck, and van as IDs 0 through 3.
VEHICLE_CLASS_IDS = frozenset({0, 1, 2, 3})
COLORS = sv.ColorPalette.from_hex(
    [
        "#E6194B",  # Entry 0 / Exit 0: top
        "#3CB44B",  # Entry 1 / Exit 1: bottom
        "#FFE119",  # Entry 2 / Exit 2: left
        "#3C76D1",  # Entry 3 / Exit 3: right
        "#E6194B",
        "#3CB44B",
        "#FFE119",
        "#3C76D1",
    ]
)

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
TRAFFIC_ZONE_POLYGONS = ENTRY_ZONE_POLYGONS + EXIT_ZONE_POLYGONS
ENTRY_ZONE_IDS = tuple(range(len(ENTRY_ZONE_POLYGONS)))
EXIT_ZONE_IDS = tuple(range(len(ENTRY_ZONE_POLYGONS), len(TRAFFIC_ZONE_POLYGONS)))


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
        detections[self.field] = marked
        return detections


@Operator
class TrackZoneVisits:
    """Attach each tracked detection's ordered zone-visit history.

    When ``start_zone_ids`` is set, a track starts collecting visits only after
    it reaches one of those zones. When it reaches an ``end_zone_ids`` zone,
    that terminal visit is recorded and its history stops changing. This lets
    an application define entry and exit zones without giving special meaning
    to the remaining zones.
    """

    def __init__(
        self,
        polygons: Polygons,
        *,
        start_zone_ids: tuple[int, ...] | None = None,
        end_zone_ids: tuple[int, ...] | None = None,
        triggering_anchors: tuple[sv.Position, ...] = (sv.Position.CENTER,),
        allow_revisit: bool = False,
        field: str = "zone_visits",
    ) -> None:
        if not polygons:
            raise ValueError("TrackZoneVisits requires at least one polygon.")
        if not triggering_anchors:
            raise ValueError("triggering_anchors must not be empty.")
        if not field:
            raise ValueError("field must not be empty.")
        if start_zone_ids is not None and any(
            zone_id < 0 or zone_id >= len(polygons) for zone_id in start_zone_ids
        ):
            raise ValueError("start_zone_ids must refer to configured polygons.")
        if end_zone_ids is not None and any(
            zone_id < 0 or zone_id >= len(polygons) for zone_id in end_zone_ids
        ):
            raise ValueError("end_zone_ids must refer to configured polygons.")
        self.zones = build_zones(polygons, triggering_anchors)
        self.start_zone_ids = (
            None if start_zone_ids is None else frozenset(start_zone_ids)
        )
        self.end_zone_ids = None if end_zone_ids is None else frozenset(end_zone_ids)
        self.allow_revisit = allow_revisit
        self.field = field
        self._zone_visits_by_tracker_id: dict[int, ZoneVisits] = {}
        self._current_zone_by_tracker_id: dict[int, int] = {}
        self._finished_tracker_ids: set[int] = set()

    def reset(self) -> None:
        """Clear zone-visit histories before processing a new video stream."""
        self._zone_visits_by_tracker_id.clear()
        self._current_zone_by_tracker_id.clear()
        self._finished_tracker_ids.clear()

    def __call__(self, detections: sv.Detections) -> sv.Detections:
        if detections.tracker_id is None:
            raise ValueError("TrackZoneVisits requires detections with tracker_id values.")

        current_zones = np.full(len(detections), -1, dtype=np.int32)
        for zone_id, zone in enumerate(self.zones):
            is_in_zone = zone.trigger(detections)
            current_zones[(current_zones == -1) & is_in_zone] = zone_id

        zone_visits = np.empty(len(detections), dtype=object)
        for index, tracker_id_value in enumerate(detections.tracker_id):
            tracker_id = int(tracker_id_value)
            if tracker_id < 0:
                zone_visits[index] = ()
                continue
            if tracker_id in self._finished_tracker_ids:
                zone_visits[index] = self._zone_visits_by_tracker_id[tracker_id]
                continue

            current_zone = int(current_zones[index])
            visits = self._zone_visits_by_tracker_id.get(tracker_id, ())
            previous_zone = self._current_zone_by_tracker_id.get(tracker_id)
            if current_zone < 0:
                self._current_zone_by_tracker_id.pop(tracker_id, None)
            elif current_zone != previous_zone:
                is_active = tracker_id in self._zone_visits_by_tracker_id
                can_start = (
                    self.start_zone_ids is None
                    or current_zone in self.start_zone_ids
                )
                if is_active or can_start:
                    if self.allow_revisit or current_zone not in visits:
                        visits = (*visits, current_zone)
                        self._zone_visits_by_tracker_id[tracker_id] = visits
                    if (
                        self.end_zone_ids is not None
                        and current_zone in self.end_zone_ids
                    ):
                        self._finished_tracker_ids.add(tracker_id)
                self._current_zone_by_tracker_id[tracker_id] = current_zone
            zone_visits[index] = visits

        detections[self.field] = zone_visits
        return detections


@dataclass(frozen=True)
class ZoneVisitMetrics:
    """Snapshot of the visit and transition analytics for one zone."""

    unique_visitor_count: int
    total_visit_count: int
    unique_arrival_count: int
    unique_departure_count: int
    unique_arrivals_from: dict[int, int]
    unique_departures_to: dict[int, int]


@Operator
class ZoneVisitAnalytics:
    """Aggregate new visits and directed transitions from zone-visit histories."""

    def __init__(self, zone_count: int, field: str = "zone_visits") -> None:
        if zone_count < 1:
            raise ValueError("zone_count must be at least one.")
        if not field:
            raise ValueError("field must not be empty.")
        self.zone_count = zone_count
        self.field = field
        self._processed_visits_by_tracker_id: dict[int, int] = {}
        self._visitor_ids_by_zone: list[set[int]] = [set() for _ in range(zone_count)]
        self._total_visit_counts = [0] * zone_count
        self._arrival_ids_by_zone: dict[int, set[int]] = {}
        self._departure_ids_by_zone: dict[int, set[int]] = {}
        self._tracker_ids_by_transition: dict[tuple[int, int], set[int]] = {}

    @property
    def metrics(self) -> tuple[ZoneVisitMetrics, ...]:
        return tuple(
            ZoneVisitMetrics(
                unique_visitor_count=len(self._visitor_ids_by_zone[zone_id]),
                total_visit_count=self._total_visit_counts[zone_id],
                unique_arrival_count=len(self._arrival_ids_by_zone.get(zone_id, set())),
                unique_departure_count=len(
                    self._departure_ids_by_zone.get(zone_id, set())
                ),
                unique_arrivals_from={
                    origin_id: len(tracker_ids)
                    for (origin_id, destination_id), tracker_ids in self._tracker_ids_by_transition.items()
                    if destination_id == zone_id
                },
                unique_departures_to={
                    destination_id: len(tracker_ids)
                    for (origin_id, destination_id), tracker_ids in self._tracker_ids_by_transition.items()
                    if origin_id == zone_id
                },
            )
            for zone_id in range(self.zone_count)
        )

    def reset(self) -> None:
        """Clear aggregate analytics before processing a new video stream."""
        self._processed_visits_by_tracker_id.clear()
        self._visitor_ids_by_zone = [set() for _ in range(self.zone_count)]
        self._total_visit_counts = [0] * self.zone_count
        self._arrival_ids_by_zone.clear()
        self._departure_ids_by_zone.clear()
        self._tracker_ids_by_transition.clear()

    def __call__(
        self, detections: sv.Detections
    ) -> tuple[sv.Detections, tuple[ZoneVisitMetrics, ...]]:
        if detections.tracker_id is None:
            raise ValueError("ZoneVisitAnalytics requires detections with tracker_id values.")
        zone_visits = detections.data.get(self.field)
        if zone_visits is None:
            raise ValueError(
                f"Detections are missing the {self.field!r} zone-visit data field."
            )
        if len(zone_visits) != len(detections):
            raise ValueError(
                f"The {self.field!r} zone-visit data field must have one value per detection."
            )

        for tracker_id_value, visits_value in zip(detections.tracker_id, zone_visits):
            tracker_id = int(tracker_id_value)
            if tracker_id < 0:
                continue
            visits = tuple(int(zone_id) for zone_id in visits_value)
            if not visits:
                continue

            processed_visits = self._processed_visits_by_tracker_id.get(tracker_id, 0)
            for visit_index in range(processed_visits, len(visits)):
                zone_id = visits[visit_index]
                if not 0 <= zone_id < self.zone_count:
                    raise ValueError(
                        f"Zone ID {zone_id} is outside the configured range 0 through "
                        f"{self.zone_count - 1}."
                    )
                self._visitor_ids_by_zone[zone_id].add(tracker_id)
                self._total_visit_counts[zone_id] += 1
                if visit_index == 0:
                    continue
                departure_zone = visits[visit_index - 1]
                arrival_zone = zone_id
                self._departure_ids_by_zone.setdefault(departure_zone, set()).add(
                    tracker_id
                )
                self._arrival_ids_by_zone.setdefault(arrival_zone, set()).add(tracker_id)
                self._tracker_ids_by_transition.setdefault(
                    (departure_zone, arrival_zone), set()
                ).add(tracker_id)
            self._processed_visits_by_tracker_id[tracker_id] = len(visits)

        return detections, self.metrics


def zone_visit_color_lookup(detection: Detection) -> int:
    """Use the first visited zone as a tracked detection's display color."""
    return int(detection.data["zone_visits"][0])


@Operator
class ZoneTransitionAnnotator:
    """Draw polygons and directed transition totals from a metrics snapshot."""

    def __init__(
        self,
        polygons: Polygons,
        zone_labels: tuple[str, ...] | None = None,
        color: sv.ColorPalette = COLORS,
        thickness: int = 2,
        text_scale: float = 0.7,
        text_thickness: int = 2,
    ) -> None:
        if zone_labels is not None and len(zone_labels) != len(polygons):
            raise ValueError("zone_labels must have one label per polygon.")
        self.polygons = polygons
        self.zone_labels = zone_labels
        self.color = color
        self.thickness = thickness
        self.text_scale = text_scale
        self.text_thickness = text_thickness

    def __call__(
        self,
        scene: npt.NDArray[np.uint8],
        detections: sv.Detections,
        metrics_by_zone: tuple[ZoneVisitMetrics, ...],
    ) -> tuple[npt.NDArray[np.uint8], sv.Detections, tuple[ZoneVisitMetrics, ...]]:
        if len(metrics_by_zone) != len(self.polygons):
            raise ValueError("The number of zone metrics must match the number of polygons.")
        annotated = scene.copy()
        for zone_id, polygon in enumerate(self.polygons):
            annotated = sv.draw_polygon(
                scene=annotated,
                polygon=polygon,
                color=self.color.by_idx(zone_id),
                thickness=self.thickness,
            )

        for destination_id, destination_polygon in enumerate(self.polygons):
            destination_center = sv.get_polygon_center(destination_polygon)
            transition_rows = sorted(
                metrics_by_zone[destination_id].unique_arrivals_from.items()
            )
            for row, (origin_id, count) in enumerate(transition_rows):
                label = (
                    str(count)
                    if self.zone_labels is None
                    else f"{self.zone_labels[origin_id]}: {count}"
                )
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
        return annotated, detections, metrics_by_zone


def build_zones(
    polygons: Polygons,
    triggering_anchors: tuple[sv.Position, ...] = (sv.Position.CENTER,),
) -> Zones:
    return tuple(
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=list(triggering_anchors),
        )
        for polygon in polygons
    )


def keep_vehicle_detections(detections: sv.Detections) -> npt.NDArray[np.bool_]:
    if detections.class_id is None:
        raise ValueError("Vehicle filtering requires detection class IDs.")
    return np.isin(np.asarray(detections.class_id), tuple(VEHICLE_CLASS_IDS))


def has_zone_visits(detections: sv.Detections) -> npt.NDArray[np.bool_]:
    zone_visits = detections.data.get("zone_visits", [()] * len(detections))
    return np.asarray([bool(visits) for visits in zone_visits], dtype=bool)


def build_frame_pipeline(
    weights_path: Path,
    zones: Polygons,
) -> Pipeline[
    npt.NDArray[np.uint8],
    tuple[npt.NDArray[np.uint8], sv.Detections, tuple[ZoneVisitMetrics, ...]],
]:
    try:
        from ml_pipes.ultralytics import yolo
    except ImportError as error:
        raise RuntimeError(
            "Traffic analysis requires ml-pipes-ultralytics. Install it with "
            "'python -m pip install \"ml-pipes-ultralytics @ "
            "git+https://github.com/requiem4machines/ml-pipes-ultralytics.git\"'."
        ) from error

    return Pipeline(
        [
            Store("source_frame"),
            yolo.Predict(model=weights_path, conf=0.3, iou=0.7),
            Select(0),
            Detections.FromUltralytics(),
            Detections.Filter(keep_vehicle_detections),
            ByteTrack(),
            TrackZoneVisits(
                zones,
                start_zone_ids=ENTRY_ZONE_IDS,
                end_zone_ids=EXIT_ZONE_IDS,
            ),
            ZoneVisitAnalytics(zone_count=len(zones)),
            Store("zone_visit_metrics", source=1),
            Pick(0),
            Detections.Filter(has_zone_visits),
            Recall("source_frame", prepend=True),
            BoxAnnotator(color=COLORS, custom_color_lookup=zone_visit_color_lookup),
            LabelAnnotator(show_tracker_id=True, color=COLORS, custom_color_lookup=zone_visit_color_lookup),
            TraceAnnotator(thickness=2, color=COLORS, custom_color_lookup=zone_visit_color_lookup),
            Recall("zone_visit_metrics"),
            ZoneTransitionAnnotator(zones),
            ImageWindow("Traffic Zone Visit Analytics", at=0),
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
            zones=TRAFFIC_ZONE_POLYGONS,
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
        callback=lambda frame, _: pipeline(frame)[0],
    )
    print(f"Saved zone-visit analysis video to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
