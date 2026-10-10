"""Tracker adapter contracts, without asserting upstream tracking algorithms."""

from typing import Any
from unittest.mock import MagicMock, create_autospec

import numpy as np
import pytest
import supervision as sv

from ml_pipes.supervision import trackers as tracker_ops


@pytest.mark.parametrize(
    ("operator_name", "defaults"),
    [
        pytest.param(
            "ByteTrack",
            {
                "lost_track_buffer": 30,
                "frame_rate": 30.0,
                "track_activation_threshold": 0.7,
                "minimum_consecutive_frames": 2,
                "minimum_iou_threshold": 0.1,
                "high_conf_det_threshold": 0.6,
                "state_estimator_class": tracker_ops.XYXYStateEstimator,
            },
            id="bytetrack",
        ),
        pytest.param(
            "BoTSORT",
            {
                "lost_track_buffer": 30,
                "frame_rate": 30.0,
                "track_activation_threshold": 0.7,
                "minimum_consecutive_frames": 2,
                "minimum_iou_threshold_first_assoc": 0.2,
                "minimum_iou_threshold_second_assoc": 0.5,
                "minimum_iou_threshold_unconfirmed_assoc": 0.3,
                "high_conf_det_threshold": 0.6,
                "enable_cmc": True,
                "cmc_method": "sparseOptFlow",
                "cmc_downscale": 2,
                "instant_first_frame_activation": True,
                "state_estimator_class": tracker_ops.XCYCWHStateEstimator,
            },
            id="botsort",
        ),
        pytest.param(
            "OCSORT",
            {
                "lost_track_buffer": 30,
                "frame_rate": 30.0,
                "minimum_consecutive_frames": 3,
                "minimum_iou_threshold": 0.3,
                "direction_consistency_weight": 0.2,
                "high_conf_det_threshold": 0.6,
                "delta_t": 3,
                "state_estimator_class": tracker_ops.XCYCSRStateEstimator,
            },
            id="ocsort",
        ),
        pytest.param(
            "SORT",
            {
                "lost_track_buffer": 30,
                "frame_rate": 30.0,
                "track_activation_threshold": 0.25,
                "minimum_consecutive_frames": 3,
                "minimum_iou_threshold": 0.3,
                "state_estimator_class": tracker_ops.XYXYStateEstimator,
            },
            id="sort",
        ),
    ],
)
@pytest.mark.parametrize("custom_options", [False, True], ids=["defaults", "custom"])
def test_tracker_constructor_forwards_options(
    operator_name: str,
    defaults: dict[str, Any],
    custom_options: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream_name = f"{operator_name}Tracker"
    upstream_class = getattr(tracker_ops, upstream_name)
    upstream = create_autospec(upstream_class, instance=True)
    construct = create_autospec(upstream_class, return_value=upstream)
    monkeypatch.setattr(tracker_ops, upstream_name, construct)
    options = (
        {
            "lost_track_buffer": 12,
            "frame_rate": 24.0,
            "state_estimator_class": tracker_ops.XCYCSRStateEstimator,
        }
        if custom_options
        else {}
    )
    if custom_options and "track_activation_threshold" in defaults:
        options["track_activation_threshold"] = 0.4

    operator = getattr(tracker_ops, operator_name)(**options)

    construct.assert_called_once_with(**{**defaults, **options})
    assert operator.tracker is upstream


@pytest.fixture
def upstream_tracker() -> MagicMock:
    return create_autospec(tracker_ops.BaseTracker, instance=True)


@pytest.mark.parametrize("entrypoint", ["update", "__call__"])
def test_update_forwards_detections_and_returns_upstream_result(
    entrypoint: str, upstream_tracker: MagicMock,
) -> None:
    operator = tracker_ops.UpdateTrackedObjects(upstream_tracker)
    detections = sv.Detections.empty()
    expected = sv.Detections.empty()
    upstream_tracker.update.return_value = expected

    result = getattr(operator, entrypoint)(detections)

    upstream_tracker.update.assert_called_once_with(detections)
    assert upstream_tracker.update.call_args.args[0] is detections
    assert result is expected


def test_update_selects_detections_from_tuple_payload(upstream_tracker: MagicMock) -> None:
    operator = tracker_ops.UpdateTrackedObjects(upstream_tracker)
    detections = sv.Detections.empty()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    result = operator((detections, frame))

    upstream_tracker.update.assert_called_once_with(detections)
    assert upstream_tracker.update.call_args.args[0] is detections
    assert result is upstream_tracker.update.return_value


def test_reset_delegates_to_upstream_tracker(upstream_tracker: MagicMock) -> None:
    operator = tracker_ops.UpdateTrackedObjects(upstream_tracker)

    assert operator.reset() is None

    upstream_tracker.reset.assert_called_once_with()


def test_readers_expose_current_upstream_tracked_objects(upstream_tracker: MagicMock) -> None:
    update = tracker_ops.UpdateTrackedObjects(upstream_tracker)
    read = tracker_ops.ReadTrackedObjects(upstream_tracker)

    for expected in [sv.Detections.empty(), sv.Detections.empty()]:
        upstream_tracker.tracked_objects = expected
        assert update.tracked_objects is expected
        assert read() is expected


@pytest.fixture
def botsort_factory(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    upstream = create_autospec(tracker_ops.BoTSORTTracker, instance=True)
    construct = create_autospec(tracker_ops.BoTSORTTracker, return_value=upstream)
    monkeypatch.setattr(tracker_ops, "BoTSORTTracker", construct)
    return construct


@pytest.mark.parametrize("entrypoint", ["update", "__call__"])
def test_botsort_forwards_detections_and_frame(
    entrypoint: str, botsort_factory: MagicMock,
) -> None:
    operator = tracker_ops.BoTSORT(enable_cmc=True)
    upstream = botsort_factory.return_value
    detections = sv.Detections.empty()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    expected = sv.Detections.empty()
    upstream.update.return_value = expected

    result = (
        operator.update(detections, frame)
        if entrypoint == "update"
        else operator((detections, frame))
    )

    upstream.update.assert_called_once_with(detections, frame)
    assert upstream.update.call_args.args[0] is detections
    assert upstream.update.call_args.args[1] is frame
    assert result is expected


@pytest.mark.parametrize("entrypoint", ["update", "__call__"])
def test_botsort_requires_frame_when_camera_motion_compensation_is_enabled(
    entrypoint: str, botsort_factory: MagicMock,
) -> None:
    operator = tracker_ops.BoTSORT(enable_cmc=True)

    with pytest.raises(ValueError, match="requires the current frame"):
        getattr(operator, entrypoint)(sv.Detections.empty())

    botsort_factory.return_value.update.assert_not_called()


@pytest.mark.parametrize("entrypoint", ["update", "__call__"])
def test_botsort_accepts_detections_only_when_camera_motion_compensation_is_disabled(
    entrypoint: str, botsort_factory: MagicMock,
) -> None:
    operator = tracker_ops.BoTSORT(enable_cmc=False)
    detections = sv.Detections.empty()
    upstream = botsort_factory.return_value

    result = getattr(operator, entrypoint)(detections)

    upstream.update.assert_called_once_with(detections, None)
    assert upstream.update.call_args.args[0] is detections
    assert result is upstream.update.return_value
