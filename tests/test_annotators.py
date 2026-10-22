from unittest.mock import create_autospec

import numpy as np
import pytest
import supervision as sv

from ml_pipes.supervision import (
    BoxAnnotator,
    Detection,
    Detections,
    HeatMapAnnotator,
    LabelAnnotator,
    RichLabelAnnotator,
    TraceAnnotator,
)


def test_box_annotator_preserves_source_scene() -> None:
    scene = np.zeros((32, 32, 3), dtype=np.uint8)
    source = scene.copy()
    detections = sv.Detections(
        xyxy=np.array([[4, 4, 28, 28]], dtype=np.float32),
        class_id=np.array([0], dtype=np.int32),
    )

    annotated, returned_detections = BoxAnnotator()(scene, detections)

    np.testing.assert_array_equal(scene, source)
    assert annotated is not scene
    assert not np.array_equal(annotated, source)
    assert returned_detections is detections


def test_label_annotator_passes_each_detection_to_callback() -> None:
    scene = np.zeros((32, 32, 3), dtype=np.uint8)
    detections = sv.Detections(
        xyxy=np.array([[4, 4, 28, 28]], dtype=np.float32),
        class_id=np.array([3], dtype=np.int32),
        tracker_id=np.array([12], dtype=np.int32),
        data={"time_in_zone": np.array([1.5], dtype=np.float32)},
    )
    received: list[Detection] = []

    annotated, returned_detections = LabelAnnotator(
        label_formatter=lambda detection: received.append(detection) or "01:30"
    )(scene, detections)

    assert annotated is not scene
    assert not np.array_equal(annotated, scene)
    assert returned_detections is detections
    assert received[0].tracker_id == 12
    assert received[0].data["time_in_zone"] == 1.5


def test_label_annotator_requires_string_label() -> None:
    scene = np.zeros((32, 32, 3), dtype=np.uint8)
    detections = sv.Detections(xyxy=np.array([[4, 4, 28, 28]], dtype=np.float32))

    with pytest.raises(TypeError, match="must return str"):
        LabelAnnotator(label_formatter=lambda _: 1)(scene, detections)


@pytest.mark.parametrize("operator_type", [LabelAnnotator, RichLabelAnnotator, BoxAnnotator])
def test_annotation_callbacks_receive_unwrapped_attached_row_values(operator_type) -> None:
    scene = np.zeros((32, 32, 3), dtype=np.uint8)
    detections = sv.Detections(
        xyxy=np.array([[4, 4, 16, 16], [16, 16, 28, 28]], dtype=np.float32),
    )
    fields = {
        "response": ["walking", "standing"],
        "accessories": [{"hat", "glasses"}, set()],
        "ragged": [["blue"], ["red", "green"]],
        "details": [{"visible": True}, {"visible": False}],
        "embedding": np.array([[1, 2], [3, 4]]),
    }
    detections = Detections.AttachData()(detections, fields)
    received: list[Detection] = []
    if operator_type is BoxAnnotator:
        operator = operator_type(
            custom_color_lookup=lambda detection: received.append(detection) or 0,
        )
    else:
        operator = operator_type(
            label_formatter=lambda detection: received.append(detection) or detection.data["response"],
            color_lookup=sv.ColorLookup.INDEX,
        )

    _, returned = operator(scene, detections)

    assert returned is detections
    assert len(received) == 2
    for index, record in enumerate(received):
        for key in ("response", "accessories", "ragged", "details"):
            assert record.data[key] is fields[key][index]
        np.testing.assert_array_equal(record.data["embedding"], fields["embedding"][index])
    assert detections.data["accessories"] is fields["accessories"]


def test_label_annotator_rejects_callback_with_standard_fields() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        LabelAnnotator(label_formatter=lambda _: "label", show_class=True)


def test_rich_label_annotator_passes_each_detection_to_callback() -> None:
    scene = np.zeros((32, 32, 3), dtype=np.uint8)
    detections = sv.Detections(
        xyxy=np.array([[4, 4, 28, 28]], dtype=np.float32),
        class_id=np.array([3], dtype=np.int32),
        tracker_id=np.array([12], dtype=np.int32),
    )
    received: list[Detection] = []

    annotated, returned_detections = RichLabelAnnotator(
        label_formatter=lambda detection: received.append(detection) or "車両 #12"
    )(scene, detections)

    assert annotated is not scene
    assert not np.array_equal(annotated, scene)
    assert returned_detections is detections
    assert received[0].tracker_id == 12


def test_rich_label_annotator_rejects_callback_with_standard_fields() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        RichLabelAnnotator(label_formatter=lambda _: "label", show_class=True)


@pytest.mark.parametrize("operator_type", [TraceAnnotator, HeatMapAnnotator])
def test_stateful_annotator_delegates_reset(
    operator_type: type[TraceAnnotator] | type[HeatMapAnnotator],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = operator_type()
    upstream = operator.annotator
    reset = create_autospec(upstream.reset)
    monkeypatch.setattr(upstream, "reset", reset)

    assert operator.reset() is None

    reset.assert_called_once_with()
    assert operator.annotator is upstream
