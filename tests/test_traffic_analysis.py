import numpy as np
import pytest
import supervision as sv

from examples.run_traffic_analysis import (
    MarkZone,
    ZoneTransitionAnnotator,
    ZoneVisitAnalytics,
    ZoneVisitMetrics,
    TrackZoneVisits,
    has_zone_visits,
    zone_visit_color_lookup,
)
from ml_pipes.supervision import BoxAnnotator


def polygon(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int64)


def tracked_detections(
    boxes: list[list[float]],
    tracker_ids: list[int],
    zone_visits: list[tuple[int, ...]] | None = None,
) -> sv.Detections:
    data: dict[str, np.ndarray] = {}
    if zone_visits is not None:
        visit_values = np.empty(len(zone_visits), dtype=object)
        visit_values[:] = zone_visits
        data["zone_visits"] = visit_values
    return sv.Detections(
        xyxy=np.asarray(boxes, dtype=np.float32).reshape((-1, 4)),
        tracker_id=np.asarray(tracker_ids, dtype=np.int32),
        data=data,
    )


def visits_for(
    tracker: TrackZoneVisits,
    box: list[float],
    tracker_id: int = 4,
) -> tuple[int, ...]:
    detections = tracker(tracked_detections([box], [tracker_id]))
    return detections.data["zone_visits"][0]


def test_mark_zone_adds_current_membership_without_filtering() -> None:
    zone = sv.PolygonZone(polygon=polygon(0, 0, 10, 10))
    marker = MarkZone((zone,), "zone")
    detections = tracked_detections([[1, 1, 3, 3], [20, 20, 22, 22]], [1, 2])

    marked = marker(detections)

    assert marked is detections
    assert len(marked) == 2
    np.testing.assert_array_equal(marked.data["zone"], [0, -1])


def test_track_zone_visits_records_transitions_not_frames() -> None:
    tracker = TrackZoneVisits((polygon(0, 0, 10, 10), polygon(20, 0, 30, 10)))

    assert visits_for(tracker, [1, 1, 3, 3]) == (0,)
    assert visits_for(tracker, [2, 1, 4, 3]) == (0,)
    assert visits_for(tracker, [12, 1, 14, 3]) == (0,)
    assert visits_for(tracker, [21, 1, 23, 3]) == (0, 1)


def test_track_zone_visits_can_allow_or_ignore_revisits() -> None:
    polygons = (polygon(0, 0, 10, 10), polygon(20, 0, 30, 10))
    without_revisits = TrackZoneVisits(polygons)
    with_revisits = TrackZoneVisits(polygons, allow_revisit=True)

    for tracker in (without_revisits, with_revisits):
        visits_for(tracker, [1, 1, 3, 3])
        visits_for(tracker, [21, 1, 23, 3])
        visits_for(tracker, [1, 1, 3, 3])

    assert visits_for(without_revisits, [1, 1, 3, 3]) == (0, 1)
    assert visits_for(with_revisits, [1, 1, 3, 3]) == (0, 1, 0)


def test_track_zone_visits_can_start_only_from_selected_zones() -> None:
    tracker = TrackZoneVisits(
        (
            polygon(0, 0, 10, 10),
            polygon(20, 0, 30, 10),
            polygon(40, 0, 50, 10),
        ),
        start_zone_ids=(0,),
    )

    assert visits_for(tracker, [21, 1, 23, 3]) == ()
    assert visits_for(tracker, [1, 1, 3, 3]) == (0,)
    assert visits_for(tracker, [41, 1, 43, 3]) == (0, 2)


def test_track_zone_visits_stops_after_an_end_zone() -> None:
    tracker = TrackZoneVisits(
        (
            polygon(0, 0, 10, 10),
            polygon(20, 0, 30, 10),
            polygon(40, 0, 50, 10),
        ),
        start_zone_ids=(0,),
        end_zone_ids=(1,),
    )

    assert visits_for(tracker, [1, 1, 3, 3]) == (0,)
    assert visits_for(tracker, [21, 1, 23, 3]) == (0, 1)
    assert visits_for(tracker, [41, 1, 43, 3]) == (0, 1)


def test_track_zone_visits_supports_custom_triggering_anchors() -> None:
    zones = (polygon(0, 0, 10, 10),)
    center_tracker = TrackZoneVisits(zones)
    bottom_right_tracker = TrackZoneVisits(
        zones,
        triggering_anchors=(sv.Position.BOTTOM_RIGHT,),
    )

    assert visits_for(center_tracker, [0, 0, 12, 8]) == (0,)
    assert visits_for(bottom_right_tracker, [0, 0, 12, 8]) == ()


def test_track_zone_visits_uses_the_first_matching_zone() -> None:
    tracker = TrackZoneVisits((polygon(0, 0, 10, 10), polygon(0, 0, 20, 20)))

    assert visits_for(tracker, [1, 1, 3, 3]) == (0,)


def test_track_zone_visits_ignores_unconfirmed_tracks_and_can_reset() -> None:
    tracker = TrackZoneVisits((polygon(0, 0, 10, 10), polygon(20, 0, 30, 10)))

    assert visits_for(tracker, [1, 1, 3, 3], tracker_id=-1) == ()
    assert visits_for(tracker, [1, 1, 3, 3]) == (0,)
    tracker.reset()
    assert visits_for(tracker, [21, 1, 23, 3]) == (1,)


def test_track_zone_visits_requires_tracker_ids() -> None:
    tracker = TrackZoneVisits((polygon(0, 0, 10, 10),))
    detections = sv.Detections(xyxy=np.zeros((1, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="tracker_id"):
        tracker(detections)


def test_has_zone_visits_treats_missing_data_as_no_matches() -> None:
    detections = tracked_detections([[1, 1, 3, 3]], [4])

    np.testing.assert_array_equal(has_zone_visits(detections), [False])


def test_zone_visit_analytics_publishes_metrics_for_each_zone() -> None:
    counter = ZoneVisitAnalytics(zone_count=3)
    entered = tracked_detections([[1, 1, 3, 3]], [4], [(0,)])

    returned, metrics = counter(entered)
    assert returned is entered
    assert metrics == (
        ZoneVisitMetrics(1, 1, 0, 0, {}, {}),
        ZoneVisitMetrics(0, 0, 0, 0, {}, {}),
        ZoneVisitMetrics(0, 0, 0, 0, {}, {}),
    )

    _, metrics = counter(tracked_detections([[21, 1, 23, 3]], [4], [(0, 1)]))
    assert metrics[0] == ZoneVisitMetrics(1, 1, 0, 1, {}, {1: 1})
    assert metrics[1] == ZoneVisitMetrics(1, 1, 1, 0, {0: 1}, {})

    _, metrics = counter(tracked_detections([[21, 1, 23, 3]], [4], [(0, 1)]))
    assert metrics[0].total_visit_count == 1
    assert metrics[1].total_visit_count == 1

    _, metrics = counter(
        tracked_detections(
            [[21, 1, 23, 3], [21, 1, 23, 3]],
            [4, 8],
            [(0, 1, 0, 1), (2, 1)],
        )
    )
    assert metrics[0] == ZoneVisitMetrics(1, 2, 1, 1, {1: 1}, {1: 1})
    assert metrics[1] == ZoneVisitMetrics(2, 3, 2, 1, {0: 1, 2: 1}, {0: 1})
    assert metrics[2] == ZoneVisitMetrics(1, 1, 0, 1, {}, {1: 1})


def test_zone_visit_analytics_requires_tracker_ids_and_zone_visit_data() -> None:
    counter = ZoneVisitAnalytics(zone_count=1)
    no_tracker_ids = sv.Detections(xyxy=np.zeros((1, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="tracker_id"):
        counter(no_tracker_ids)
    with pytest.raises(ValueError, match="zone-visit data field"):
        counter(tracked_detections([[1, 1, 3, 3]], [4]))


def test_zone_visit_analytics_reset_clears_metrics() -> None:
    counter = ZoneVisitAnalytics(zone_count=2)
    counter(tracked_detections([[1, 1, 3, 3]], [4], [(0, 1)]))

    counter.reset()

    assert counter.metrics == (
        ZoneVisitMetrics(0, 0, 0, 0, {}, {}),
        ZoneVisitMetrics(0, 0, 0, 0, {}, {}),
    )


def test_zone_transition_annotator_renders_explicit_metrics_without_counter_state() -> None:
    polygons = (polygon(0, 0, 10, 10), polygon(20, 0, 30, 10))
    detections = tracked_detections([[1, 1, 3, 3]], [4], [(0, 1)])
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    metrics = (
        ZoneVisitMetrics(1, 1, 0, 1, {}, {1: 1}),
        ZoneVisitMetrics(1, 1, 1, 0, {0: 1}, {}),
    )

    boxed, boxed_detections = BoxAnnotator(
        color=sv.ColorPalette.from_hex(["#E6194B", "#3CB44B"]),
        custom_color_lookup=zone_visit_color_lookup,
    )(frame, detections)
    annotated, returned_detections = ZoneTransitionAnnotator(
        polygons,
        zone_labels=("North entry", "North exit"),
    )(
        boxed,
        boxed_detections,
        metrics,
    )

    assert annotated.shape == frame.shape
    assert returned_detections is boxed_detections


def test_zone_transition_annotator_requires_one_label_per_zone() -> None:
    polygons = (polygon(0, 0, 10, 10), polygon(20, 0, 30, 10))

    with pytest.raises(ValueError, match="one label per polygon"):
        ZoneTransitionAnnotator(polygons, zone_labels=("North",))


def test_zone_transition_annotator_uses_color_only_when_labels_are_omitted() -> None:
    annotator = ZoneTransitionAnnotator((polygon(0, 0, 10, 10),))

    assert annotator.zone_labels is None
