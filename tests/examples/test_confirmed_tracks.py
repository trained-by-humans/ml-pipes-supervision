"""Example pipelines discard pending IDs before identity-dependent stages."""

from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import create_autospec

import numpy as np
import pytest
import supervision as sv

from ml_pipes.supervision import TraceAnnotator
from ml_pipes.supervision import trackers as tracker_ops


@pytest.mark.parametrize(
    ("example_name", "tracker_name"),
    [
        ("run_count_objects_crossing_line", "bytetrack"),
        ("run_time_in_zone", "bytetrack"),
        ("run_track_objects", "bytetrack"),
        ("run_track_objects", "botsort"),
        ("run_track_objects", "ocsort"),
        ("run_track_objects", "sort"),
    ],
)
@pytest.mark.parametrize("tracker_ids", [(-1, 0, 7), (-1, -1, -1)])
def test_examples_filter_pending_tracks_before_counting_and_traces(
    example_name: str,
    tracker_name: str,
    tracker_ids: tuple[int, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if find_spec("inference") is None:
        pytest.skip("Optional Inference dependency is not installed.")

    from ml_pipes.supervision import inference as inference_ops

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))
    example = import_module(f"examples.{example_name}")
    model = create_autospec(inference_ops.Model, instance=True)
    model.infer.return_value = [
        {
            "image": {"width": 96, "height": 96},
            "predictions": [
                {
                    "x": x,
                    "y": 20,
                    "width": 8,
                    "height": 8,
                    "confidence": 0.9,
                    "class_id": 2,
                    "class": "car",
                }
                for x in (20, 48, 76)
            ],
        }
    ]
    monkeypatch.setattr(inference_ops, "get_model", lambda **_: model)

    tracked = sv.Detections.from_inference(model.infer.return_value[0])
    tracked.tracker_id = np.asarray(tracker_ids, dtype=int)
    upstream_classes = {
        "bytetrack": tracker_ops.ByteTrackTracker,
        "botsort": tracker_ops.BoTSORTTracker,
        "ocsort": tracker_ops.OCSORTTracker,
        "sort": tracker_ops.SORTTracker,
    }
    upstream_class = upstream_classes[tracker_name]
    upstream = create_autospec(upstream_class, instance=True)
    upstream.update.return_value = tracked
    monkeypatch.setattr(tracker_ops, upstream_class.__name__, lambda **_: upstream)
    monkeypatch.setattr(sv.ImageWindow, "show", lambda *_: None)

    expected_ids = [tracker_id for tracker_id in tracker_ids if tracker_id != -1]
    zone = None
    if example_name == "run_count_objects_crossing_line":
        zone = sv.LineZone(start=sv.Point(0, 48), end=sv.Point(96, 48))
        pipeline = example.build_frame_pipeline("test-model", None, zone)
    elif example_name == "run_time_in_zone":
        zone = sv.PolygonZone(polygon=np.asarray([[0, 0], [95, 0], [95, 95], [0, 95]]))
        pipeline = example.build_frame_pipeline("test-model", None, zone, 25.0)
    else:
        pipeline = example.build_frame_pipeline(
            "test-model", None, example.build_tracker(tracker_name)
        )

    consumer_ids = []
    if zone is not None:
        trigger = zone.trigger

        def record_trigger(detections):
            consumer_ids.append(detections.tracker_id.tolist())
            return trigger(detections)

        monkeypatch.setattr(zone, "trigger", record_trigger)

    trace = next(
        operator for operator in pipeline.operators
        if isinstance(operator, TraceAnnotator)
    )
    annotate = trace.annotator.annotate

    def record_trace(*, scene, detections, **kwargs):
        consumer_ids.append(detections.tracker_id.tolist())
        return annotate(scene=scene, detections=detections, **kwargs)

    monkeypatch.setattr(trace.annotator, "annotate", record_trace)
    frame = np.zeros((96, 96, 3), dtype=np.uint8)

    result = pipeline(frame)

    assert consumer_ids == [expected_ids] * (2 if zone is not None else 1)
    upstream.update.assert_called_once()
    assert len(upstream.update.call_args.args[0]) == 3
    assert tracked.tracker_id.tolist() == list(tracker_ids)
    model.infer.assert_called_once_with(frame)
    assert not np.any(frame)
    annotated_frame = (
        result if example_name == "run_count_objects_crossing_line" else result[0]
    )
    assert annotated_frame.shape == frame.shape
    assert annotated_frame.dtype == np.uint8
    if isinstance(result, tuple):
        assert result[1].tracker_id.tolist() == expected_ids
