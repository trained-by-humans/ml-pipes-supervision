"""Adapter contracts, with real-library smoke tests at conversion boundaries."""

from copy import deepcopy
from typing import Any
from unittest.mock import create_autospec

import numpy as np
import pytest
import supervision as sv

from ml_pipes.supervision import Detections, DetectionsSmoother, ImageToArray
from ml_pipes.supervision import core as supervision_core
from ml_pipes.tensor import TensorRegistry
from ml_pipes.vision import ImagePayload, TileRect


@pytest.mark.parametrize(
    ("options", "compact_masks"),
    [({}, False), ({"compact_masks": False}, False), ({"compact_masks": True}, True)],
    ids=["default", "dense", "compact"],
)
def test_from_inference_forwards_payload_and_mask_option(
    options: dict[str, Any],
    compact_masks: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference_result = {"image": {"width": 8, "height": 8}, "predictions": []}
    expected = sv.Detections.empty()
    # Autospec retains the real dependency's API signature, not its algorithms.
    convert = create_autospec(sv.Detections.from_inference, return_value=expected)
    monkeypatch.setattr(sv.Detections, "from_inference", convert)

    result = Detections.FromInference(**options)(inference_result)

    convert.assert_called_once_with(inference_result, compact_masks=compact_masks)
    assert convert.call_args.args[0] is inference_result
    assert result is expected


def test_from_ultralytics_forwards_payload_and_returns_upstream_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = object()
    expected = sv.Detections.empty()
    convert = create_autospec(sv.Detections.from_ultralytics, return_value=expected)
    monkeypatch.setattr(sv.Detections, "from_ultralytics", convert)

    result = Detections.FromUltralytics()(payload)

    convert.assert_called_once_with(payload)
    assert result is expected


def test_tensor_registry_detection_adapter() -> None:
    detections = Detections.FromTensorRegistry()(
        TensorRegistry(
            {
                "boxes": np.asarray([[0, 0, 10, 10]], dtype=np.float32),
                "scores": np.asarray([0.9], dtype=np.float32),
                "classes": np.asarray([1], dtype=np.int32),
            }
        )
    )

    assert detections.xyxy.tolist() == [[0.0, 0.0, 10.0, 10.0]]
    assert detections.class_id.tolist() == [1]


def test_tensor_registry_detection_adapter_accepts_custom_mask_fields() -> None:
    detections = Detections.FromTensorRegistry(
        boxes="xyxy", scores="confidence", classes="class_id", masks="mask"
    )(
        TensorRegistry(
            {
                "xyxy": np.asarray([[0, 0, 2, 2]], dtype=np.float32),
                "confidence": np.asarray([0.75], dtype=np.float32),
                "class_id": np.asarray([3], dtype=np.int32),
                "mask": np.ones((1, 2, 2), dtype=bool),
            }
        )
    )

    assert detections.mask is not None
    assert detections.mask.shape == (1, 2, 2)


def test_detections_filter_requires_detections_result() -> None:
    detections = sv.Detections.empty()

    with pytest.raises(TypeError, match="must return supervision.Detections or a boolean"):
        Detections.Filter(lambda _: None)(detections)


def masked_detections(compact_masks: bool) -> sv.Detections:
    boxes = np.asarray([[1, 1, 3, 3]], dtype=np.float32)
    masks = np.zeros((1, 4, 4), dtype=bool)
    masks[0, 1:4, 1:4] = True
    return sv.Detections(
        xyxy=boxes,
        mask=sv.CompactMask.from_dense(masks, boxes, (4, 4))
        if compact_masks
        else masks,
        confidence=np.asarray([0.9], dtype=np.float32),
        class_id=np.asarray([3], dtype=np.int32),
    )


@pytest.mark.parametrize(
    ("first_compact", "second_compact"),
    [(False, False), (True, True), (False, True)],
    ids=["dense", "compact", "mixed"],
)
def test_stitch_places_tile_masks_in_shared_image_coordinates(
    first_compact: bool, second_compact: bool,
) -> None:
    tiles = [masked_detections(first_compact), masked_detections(second_compact)]
    source = deepcopy(tiles)

    stitched = Detections.Stitch()(
        tiles, [TileRect(0, 0, 4, 4), TileRect(4, 2, 8, 6)]
    )

    np.testing.assert_array_equal(stitched.xyxy, [[1, 1, 3, 3], [5, 3, 7, 5]])
    # The merged representation is upstream's choice; our contract is placement.
    masks = (
        stitched.mask.to_dense()
        if isinstance(stitched.mask, sv.CompactMask)
        else stitched.mask
    )
    expected = np.zeros((2, 6, 8), dtype=bool)
    expected[0, 1:4, 1:4] = True
    expected[1, 3:6, 5:8] = True
    np.testing.assert_array_equal(masks, expected)
    assert tiles == source


def test_stitch_forwards_offsets_resolution_and_tile_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tiles = [sv.Detections.empty(), sv.Detections.empty()]
    moved = [sv.Detections.empty(), sv.Detections.empty()]
    expected = sv.Detections.empty()
    move = create_autospec(supervision_core.move_detections, side_effect=moved)
    merge = create_autospec(sv.Detections.merge, return_value=expected)
    monkeypatch.setattr(supervision_core, "move_detections", move)
    monkeypatch.setattr(sv.Detections, "merge", merge)

    result = Detections.Stitch()(
        tiles, [TileRect(4, 2, 8, 6), TileRect(0, 0, 4, 4)]
    )

    assert result is expected
    assert move.call_count == 2
    for call, tile, offset in zip(move.call_args_list, tiles, [(4, 2), (0, 0)]):
        assert call.kwargs["detections"] is tile
        np.testing.assert_array_equal(call.kwargs["offset"], offset)
        assert call.kwargs["resolution_wh"] == (8, 6)
    merge.assert_called_once_with(moved)
    assert merge.call_args.args[0][0] is moved[0]
    assert merge.call_args.args[0][1] is moved[1]


def test_stitch_requires_one_rectangle_per_tile() -> None:
    with pytest.raises(ValueError, match="one tile rectangle"):
        Detections.Stitch()([sv.Detections.empty()], [])


def test_stitch_accepts_an_empty_tile_list() -> None:
    result = Detections.Stitch()([], [])

    assert isinstance(result, sv.Detections)
    assert len(result) == 0


@pytest.mark.parametrize(
    ("operator_name", "method_name", "threshold_option"),
    [("NMS", "with_nms", "threshold"), ("NMM", "with_nmm", "iou_threshold")],
)
@pytest.mark.parametrize("custom_options", [False, True], ids=["defaults", "custom"])
def test_overlap_filter_forwards_options_and_returns_upstream_result(
    operator_name: str,
    method_name: str,
    threshold_option: str,
    custom_options: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detections = sv.Detections.empty()
    expected = sv.Detections.empty()
    filter_method = create_autospec(
        getattr(detections, method_name), return_value=expected
    )
    monkeypatch.setattr(detections, method_name, filter_method)
    options = (
        {
            threshold_option: 0.25,
            "class_agnostic": True,
            "overlap_metric": sv.OverlapMetric.IOS,
        }
        if custom_options
        else {}
    )

    result = getattr(Detections, operator_name)(**options)(detections)

    filter_method.assert_called_once_with(
        threshold=0.25 if custom_options else 0.5,
        class_agnostic=custom_options,
        overlap_metric=sv.OverlapMetric.IOS if custom_options else sv.OverlapMetric.IOU,
    )
    assert result is expected


def test_smoother_forwards_window_length(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = sv.DetectionsSmoother(length=2)
    construct = create_autospec(sv.DetectionsSmoother, return_value=upstream)
    monkeypatch.setattr(sv, "DetectionsSmoother", construct)

    operator = DetectionsSmoother(length=2)

    construct.assert_called_once_with(length=2)
    assert operator.smoother is upstream


@pytest.mark.parametrize("entrypoint", ["update", "__call__"])
def test_smoother_forwards_update_and_returns_upstream_result(
    entrypoint: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = DetectionsSmoother(length=2)
    detections = sv.Detections.empty()
    expected = sv.Detections.empty()
    update = create_autospec(
        operator.smoother.update_with_detections, return_value=expected
    )
    monkeypatch.setattr(operator.smoother, "update_with_detections", update)

    result = getattr(operator, entrypoint)(detections)

    update.assert_called_once_with(detections)
    assert update.call_args.args[0] is detections
    assert result is expected


def test_smoother_delegates_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    operator = DetectionsSmoother()
    reset = create_autospec(operator.smoother.reset)
    monkeypatch.setattr(operator.smoother, "reset", reset)

    assert operator.reset() is None

    reset.assert_called_once_with()


@pytest.mark.parametrize("color_space", ["RGB", "BGR"])
def test_image_to_array_returns_an_independent_contiguous_bgr_frame(
    color_space: str,
) -> None:
    array = np.asarray([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)[:, ::-1]
    payload = ImagePayload(array=array, color_space=color_space)
    source = array.copy()

    frame = ImageToArray()(payload)

    expected = source[..., ::-1] if color_space == "RGB" else source
    np.testing.assert_array_equal(frame, expected)
    np.testing.assert_array_equal(array, source)
    assert frame.flags.c_contiguous
    assert not np.shares_memory(frame, array)
