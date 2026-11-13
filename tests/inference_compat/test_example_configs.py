"""Configuration and filtering choices owned by our runnable examples."""

from collections.abc import Callable
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, create_autospec

import numpy as np
import pytest
import supervision as sv

from ml_pipes.standard import Scatter
from ml_pipes.supervision import Detections, TrackingTimer
from ml_pipes.vision import Tile


@pytest.fixture
def example_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[str], tuple[ModuleType, MagicMock]]:
    if find_spec("inference") is None:
        pytest.skip("Optional Inference dependency is not installed.")

    from ml_pipes.supervision import inference as inference_ops

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))
    model = create_autospec(inference_ops.Model, instance=True)
    monkeypatch.setattr(inference_ops, "get_model", lambda **_: model)

    def load(name: str) -> tuple[ModuleType, MagicMock]:
        return import_module(f"examples.{name}"), model

    return load


def test_small_object_example_uses_slicer_defaults(example_factory) -> None:
    example, _ = example_factory("run_detect_small_objects")
    pipeline = example.build_pipeline(
        example.DEFAULT_MODEL_ID,
        None,
        example.DEFAULT_SLICE_WH,
        example.DEFAULT_OVERLAP_WH,
        example.DEFAULT_MAX_CONCURRENCY,
        example.DEFAULT_IOU_THRESHOLD,
    )

    tile = next(op for op in pipeline.operators if isinstance(op, Tile))
    scatter = next(op for op in pipeline.operators if isinstance(op, Scatter))
    nms = next(op for op in pipeline.operators if isinstance(op, Detections.NMS))

    assert example.DEFAULT_MODEL_ID == "rfdetr-medium"
    assert tile.slice_wh == (640, 640)
    assert tile.overlap_wh == (100, 100)
    assert scatter.gate.max_concurrency == 1
    assert nms.threshold == 0.5
    assert nms.overlap_metric == sv.OverlapMetric.IOU
    assert not any(isinstance(op, Detections.NMM) for op in pipeline.operators)


def test_filter_example_composes_class_and_strict_boundary_filters(example_factory) -> None:
    example, _ = example_factory("run_filter_detections")
    pipeline = example.build_pipeline(example.DEFAULT_MODEL_ID, None, (100, 100))
    detections = sv.Detections(
        xyxy=np.asarray(
            [[0, 0, 79, 100], [0, 0, 79, 100], [0, 0, 79, 100], [0, 0, 80, 100]],
            dtype=np.float32,
        ),
        class_id=np.asarray([1, 3, 1, 1]),
        confidence=np.asarray([0.9, 0.9, 0.5, 0.9], dtype=np.float32),
    )
    filtered = detections
    for operator in pipeline.operators:
        if isinstance(operator, Detections.Filter):
            filtered = operator(filtered)

    assert example.DEFAULT_MODEL_ID == "rfdetr-small"
    assert example.DEFAULT_CLASS_ID == 1
    np.testing.assert_array_equal(filtered.xyxy, detections.xyxy[:1])
    assert len(detections) == 4


def test_time_zone_example_changes_configs_not_zone_or_timer_semantics(example_factory) -> None:
    example, _ = example_factory("run_time_in_zone")
    zone = example.build_zone(100, 100)
    pipeline = example.build_frame_pipeline(example.DEFAULT_MODEL_ID, None, zone, 25.0)
    timer = next(op for op in pipeline.operators if isinstance(op, TrackingTimer))

    assert example.DEFAULT_MODEL_ID == "rfdetr-medium"
    assert tuple(zone.triggering_anchors) == (sv.Position.CENTER,)
    np.testing.assert_array_equal(zone.polygon, [[20, 20], [80, 20], [80, 80], [20, 80]])
    assert timer.reset_missing_tracks is True
    assert timer.fps == 25.0


def test_yolo_world_example_uses_strict_relative_area_limit(example_factory) -> None:
    example, model = example_factory("run_zero_shot_object_detection")
    pipeline = example.build_frame_pipeline(model, example.DEFAULT_TEXT, (100, 100))
    area_filter = next(
        op for op in pipeline.operators if isinstance(op, Detections.Filter)
    )
    detections = sv.Detections(
        xyxy=np.asarray([[0, 0, 9, 100], [0, 0, 10, 100]], dtype=np.float32),
    )

    filtered = area_filter(detections)

    np.testing.assert_array_equal(filtered.xyxy, detections.xyxy[:1])
    assert len(detections) == 2
