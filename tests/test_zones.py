from unittest.mock import create_autospec

import numpy as np
import pytest
import supervision as sv

from ml_pipes.supervision import TrackingTimer, TriggerLineZone, TriggerZone


def tracked_detections(*tracker_ids: int) -> sv.Detections:
    count = len(tracker_ids)
    return sv.Detections(
        xyxy=np.zeros((count, 4), dtype=np.float32),
        tracker_id=np.asarray(tracker_ids, dtype=np.int32),
    )


def test_tracking_timer_tracks_continuous_dwell_time() -> None:
    timer = TrackingTimer(fps=10)

    first = timer(tracked_detections(7))
    second = timer(tracked_detections(7))
    timer(tracked_detections())
    reentered = timer(tracked_detections(7))

    np.testing.assert_allclose(first.data["tracking_time"], [0.0])
    np.testing.assert_allclose(second.data["tracking_time"], [0.1])
    np.testing.assert_allclose(reentered.data["tracking_time"], [0.0])


def test_tracking_timer_requires_tracking_ids() -> None:
    timer = TrackingTimer(fps=30)
    detections = sv.Detections(xyxy=np.zeros((1, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="tracker_id"):
        timer(detections)


def test_tracking_timer_can_retain_time_across_missing_frames() -> None:
    timer = TrackingTimer(fps=10, reset_missing_tracks=False)

    timer(tracked_detections(7))
    timer(tracked_detections())
    reentered = timer(tracked_detections(7))

    np.testing.assert_allclose(reentered.data["tracking_time"], [0.2])


def test_tracking_timer_assigns_zero_to_unconfirmed_tracks() -> None:
    timer = TrackingTimer(fps=10)

    unconfirmed = timer(tracked_detections(-1))
    confirmed = timer(tracked_detections(7))

    np.testing.assert_allclose(unconfirmed.data["tracking_time"], [0.0])
    np.testing.assert_allclose(confirmed.data["tracking_time"], [0.0])


@pytest.mark.parametrize("fps", [0, -1])
def test_tracking_timer_requires_positive_frame_rate(fps: float) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        TrackingTimer(fps=fps)


def test_line_zone_initializes_crossing_results() -> None:
    zone = sv.LineZone(
        start=sv.Point(0, 5),
        end=sv.Point(20, 5),
        triggering_anchors=(sv.Position.CENTER,),
    )
    trigger = TriggerLineZone(zone)

    assert trigger.line_zone is zone
    assert trigger.crossed_in.shape == (0,)
    assert trigger.crossed_out.shape == (0,)
    assert trigger.crossed_in.dtype == bool
    assert trigger.crossed_out.dtype == bool


def test_line_zone_forwards_detections_and_replaces_crossing_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zone = sv.LineZone(
        start=sv.Point(0, 5),
        end=sv.Point(20, 5),
        triggering_anchors=(sv.Position.CENTER,),
    )
    trigger = TriggerLineZone(zone)
    detections = tracked_detections(7, 8)
    results = [
        (np.asarray([True, False]), np.asarray([False, True])),
        (np.asarray([False, False]), np.asarray([False, False])),
    ]
    update = create_autospec(zone.trigger, side_effect=results)
    monkeypatch.setattr(zone, "trigger", update)

    for crossed_in, crossed_out in results:
        assert trigger(detections) is detections
        assert trigger.crossed_in is crossed_in
        assert trigger.crossed_out is crossed_out

    assert update.call_count == 2
    for call in update.call_args_list:
        assert call.args[0] is detections
        assert call.kwargs == {}


def test_polygon_zone_uses_upstream_membership_to_select_detections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zone = sv.PolygonZone(
        polygon=np.asarray([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.int64)
    )
    detections = tracked_detections(7, 8)
    membership = np.asarray([False, True])
    update = create_autospec(zone.trigger, return_value=membership)
    monkeypatch.setattr(zone, "trigger", update)

    filtered = TriggerZone(zone)(detections)

    update.assert_called_once_with(detections)
    assert update.call_args.args[0] is detections
    np.testing.assert_array_equal(filtered.tracker_id, [8])
    np.testing.assert_array_equal(detections.tracker_id, [7, 8])
