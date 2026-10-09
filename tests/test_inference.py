"""Inference wrapper contracts; no model construction, downloads, or execution."""

from unittest.mock import MagicMock, create_autospec

import numpy as np
import pytest

from ml_pipes.supervision import inference as inference_ops
from ml_pipes.vision import ImagePayload


@pytest.fixture
def model() -> MagicMock:
    return create_autospec(inference_ops.Model, instance=True)


@pytest.mark.parametrize("api_key", [None, "test-key"], ids=["no-key", "explicit-key"])
def test_model_id_and_key_are_forwarded_to_model_factory(
    api_key: str | None, model: MagicMock, monkeypatch: pytest.MonkeyPatch,
) -> None:
    construct = create_autospec(inference_ops.get_model, return_value=model)
    monkeypatch.setattr(inference_ops, "get_model", construct)

    operator = inference_ops.RoboflowInference(
        model_id="test-model/1", api_key=api_key, confidence=0.25
    )
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    result = operator(image)

    expected = {"model_id": "test-model/1"}
    if api_key is not None:
        expected["api_key"] = api_key
    construct.assert_called_once_with(**expected)
    model.infer.assert_called_once_with(image, confidence=0.25)
    assert result is model.infer.return_value


@pytest.mark.parametrize("keyword_model", [False, True], ids=["positional", "keyword"])
def test_injected_model_bypasses_factory_and_receives_inference_options(
    keyword_model: bool, model: MagicMock, monkeypatch: pytest.MonkeyPatch,
) -> None:
    construct = create_autospec(inference_ops.get_model)
    monkeypatch.setattr(inference_ops, "get_model", construct)
    options = {"confidence": 0.25, "iou_threshold": 0.4}
    operator = (
        inference_ops.RoboflowInference(model=model, **options)
        if keyword_model
        else inference_ops.RoboflowInference(model, **options)
    )
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    result = operator(image)

    construct.assert_not_called()
    model.infer.assert_called_once_with(image, **options)
    assert model.infer.call_args.args[0] is image
    assert result is model.infer.return_value


@pytest.mark.parametrize("color_space", ["RGB", "BGR"])
def test_image_payload_is_normalized_to_contiguous_bgr_without_mutation(
    color_space: str, model: MagicMock,
) -> None:
    operator = inference_ops.RoboflowInference(model=model)
    array = np.asarray([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)[:, ::-1]
    source = array.copy()
    payload = ImagePayload(array=array, color_space=color_space)

    result = operator(payload)

    model.infer.assert_called_once()
    frame = model.infer.call_args.args[0]
    expected = source[..., ::-1] if color_space == "RGB" else source
    np.testing.assert_array_equal(frame, expected)
    np.testing.assert_array_equal(array, source)
    assert frame.flags.c_contiguous
    assert result is model.infer.return_value


@pytest.mark.parametrize("image_kind", ["array", "path"])
def test_non_payload_input_is_forwarded_unchanged(
    image_kind: str, model: MagicMock,
) -> None:
    operator = inference_ops.RoboflowInference(model=model)
    image = (
        np.zeros((2, 2, 3), dtype=np.uint8)
        if image_kind == "array"
        else "not-loaded-by-the-wrapper.png"
    )

    result = operator(image)

    model.infer.assert_called_once_with(image)
    assert model.infer.call_args.args[0] is image
    assert result is model.infer.return_value


def test_unsupported_payload_layout_is_rejected_before_inference(model: MagicMock) -> None:
    operator = inference_ops.RoboflowInference(model=model)
    payload = ImagePayload(
        array=np.zeros((3, 2, 2), dtype=np.uint8), color_space="RGB", layout="CHW"
    )

    with pytest.raises(ValueError, match="expects HWC"):
        operator(payload)

    model.infer.assert_not_called()


@pytest.mark.parametrize(
    ("scenario", "message"),
    [
        ("missing-model", "requires model_id or model"),
        ("model-and-id", "either model_id or model, not both"),
        ("key-with-model", "api_key only when initialized with a model id"),
    ],
)
def test_invalid_model_selection_is_rejected_without_calling_factory(
    scenario: str, message: str, model: MagicMock, monkeypatch: pytest.MonkeyPatch,
) -> None:
    construct = create_autospec(inference_ops.get_model)
    monkeypatch.setattr(inference_ops, "get_model", construct)
    options = {
        "missing-model": {},
        "model-and-id": {"model_id": "test-model/1", "model": model},
        "key-with-model": {"model": model, "api_key": "test-key"},
    }[scenario]

    with pytest.raises(ValueError, match=message):
        inference_ops.RoboflowInference(**options)

    construct.assert_not_called()
