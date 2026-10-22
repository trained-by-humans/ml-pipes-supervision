"""Example-owned local Qwen contracts and generation without model downloads."""
from contextlib import nullcontext
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from examples import qwen_local
from examples.qwen_local import DEFAULT_MODEL_ID, INSTALL_COMMAND, LocalQwen3VL, require_local_qwen
from ml_pipes.vision import ImagePayload


@pytest.fixture
def dependencies(monkeypatch):
    """Keep the optional runtime absent from example tests, mocking only its API."""
    torch = SimpleNamespace(
        device=MagicMock(side_effect=lambda value: SimpleNamespace(type=value.split(":")[0])),
        float32="float32", float16="float16", bfloat16="bfloat16",
        inference_mode=MagicMock(side_effect=nullcontext),
        cuda=SimpleNamespace(
            device=MagicMock(side_effect=lambda _: nullcontext()),
            is_available=MagicMock(return_value=True),
            is_bf16_supported=MagicMock(return_value=True),
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=MagicMock(return_value=True))),
    )
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model
    model.generate.return_value = np.array([[11, 12, 31, 32, 99]])
    model.generation_config.eos_token_id = [99, 100]
    model_factory = MagicMock(return_value=model)
    processor = MagicMock()
    processor.apply_chat_template.return_value = "<chat-template>"
    inputs = {"input_ids": np.array([[11, 12]]), "pixel_values": np.ones((1, 3))}
    processor.return_value.to.return_value = inputs
    processor.batch_decode.return_value = ["  casual\nJeans and sneakers.\n"]
    processor_factory = MagicMock(return_value=processor)
    fromarray = MagicMock(side_effect=lambda array: array.copy())
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=SimpleNamespace(fromarray=fromarray)))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=processor_factory),
        Qwen3VLForConditionalGeneration=SimpleNamespace(from_pretrained=model_factory),
    ))
    return SimpleNamespace(
        torch=torch, model=model, model_factory=model_factory, processor=processor,
        processor_factory=processor_factory, inputs=inputs, fromarray=fromarray,
    )


@pytest.fixture
def image():
    return ImagePayload(np.full((8, 12, 3), [230, 20, 10], dtype=np.uint8), color_space="RGB")


def test_rgb_pixels_prompt_and_generation_options_are_forwarded_without_swapping_or_resizing(dependencies, image):
    operator = LocalQwen3VL("Describe the outfit.", max_tokens=100)
    original = image.array.copy()

    assert operator(image) == "casual\nJeans and sneakers."

    dependencies.model_factory.assert_called_once_with(
        DEFAULT_MODEL_ID, dtype="float32", attn_implementation="eager",
    )
    dependencies.processor_factory.assert_called_once_with(DEFAULT_MODEL_ID)
    np.testing.assert_array_equal(dependencies.fromarray.call_args.args[0], original)
    request = dependencies.processor.apply_chat_template.call_args
    assert request.kwargs == {"tokenize": False, "add_generation_prompt": True}
    assert request.args[0][0]["role"] == "user"
    content = request.args[0][0]["content"]
    assert content[1] == {"type": "text", "text": "Describe the outfit."}
    np.testing.assert_array_equal(content[0]["image"], original)
    preprocessing = dependencies.processor.call_args.kwargs
    assert preprocessing["text"] == ["<chat-template>"]
    assert preprocessing["return_tensors"] == "pt"
    assert preprocessing["padding"] is True
    np.testing.assert_array_equal(preprocessing["images"][0], original)
    generated = dependencies.model.generate.call_args.kwargs
    assert generated["max_new_tokens"] == 100
    assert generated["do_sample"] is False
    assert generated["return_dict_in_generate"] is False
    assert generated["input_ids"] is dependencies.inputs["input_ids"]
    np.testing.assert_array_equal(dependencies.processor.batch_decode.call_args.args[0], [[31, 32, 99]])
    assert dependencies.processor.batch_decode.call_args.kwargs == {
        "skip_special_tokens": True, "clean_up_tokenization_spaces": False,
    }
    dependencies.model.eval.assert_called_once()
    dependencies.torch.inference_mode.assert_called_once()
    np.testing.assert_array_equal(image.array, original)


def test_model_and_processor_are_loaded_lazily_once_and_reused(dependencies, image):
    operator = LocalQwen3VL("style", model_id="/models/qwen")
    assert operator._model is operator._processor is None
    dependencies.model_factory.assert_not_called()
    dependencies.processor_factory.assert_not_called()

    operator(image)
    operator(image)

    dependencies.model_factory.assert_called_once_with(
        "/models/qwen", dtype="float32", attn_implementation="eager",
    )
    dependencies.processor_factory.assert_called_once_with("/models/qwen")
    assert dependencies.model.generate.call_count == 2


@pytest.mark.parametrize("invalid", [
    np.zeros((8, 12, 3), dtype=np.uint8),
    ImagePayload(np.zeros((8, 12, 3), dtype=np.uint8)),  # BGR
    ImagePayload(np.zeros((3, 8, 12), dtype=np.uint8), color_space="RGB", layout="CHW"),
    ImagePayload(np.zeros((8, 12), dtype=np.uint8), color_space="RGB"),
    ImagePayload(np.zeros((8, 12, 4), dtype=np.uint8), color_space="RGB"),
    ImagePayload(np.zeros((8, 12, 3), dtype=np.float32), color_space="RGB"),
    ImagePayload(np.empty((0, 12, 3), dtype=np.uint8), color_space="RGB"),
])
def test_invalid_images_are_rejected_before_loading_optional_dependencies(dependencies, invalid):
    with pytest.raises(ValueError, match="RGB HWC uint8 ImagePayload"):
        LocalQwen3VL("style")(invalid)
    dependencies.model_factory.assert_not_called()
    dependencies.processor_factory.assert_not_called()


@pytest.mark.parametrize("options", [
    {"prompt": " "}, {"model_id": " "}, {"device": "auto"}, {"device": "cuda:abc"},
    {"max_tokens": 0}, {"max_tokens": -1}, {"max_tokens": 1.5}, {"max_tokens": True},
])
def test_invalid_configuration_is_rejected_without_loading(dependencies, options):
    with pytest.raises(ValueError):
        LocalQwen3VL(**{"prompt": "style", **options})
    dependencies.model_factory.assert_not_called()


@pytest.mark.parametrize("missing", ["torch", "PIL", "transformers"])
def test_runtime_dependencies_are_optional_until_first_crop(dependencies, image, monkeypatch, missing):
    monkeypatch.setitem(sys.modules, missing, None)
    operator = LocalQwen3VL("style")
    assert operator._model is None
    with pytest.raises(ImportError, match="Install them first") as error:
        operator(image)
    assert INSTALL_COMMAND in str(error.value)
    assert isinstance(error.value.__cause__, ModuleNotFoundError)


def test_cli_dependency_check_does_not_download_or_load_weights(dependencies, capsys):
    require_local_qwen()

    assert capsys.readouterr() == ("", "")
    dependencies.model_factory.assert_not_called()
    dependencies.processor_factory.assert_not_called()
    dependencies.torch.device.assert_not_called()


@pytest.mark.parametrize("missing", ["torch", "PIL", "transformers"])
def test_cli_dependency_check_reports_missing_runtime_and_exits_cleanly(dependencies, monkeypatch, capsys, missing):
    monkeypatch.setitem(sys.modules, missing, None)

    with pytest.raises(SystemExit) as error:
        require_local_qwen()

    assert error.value.code == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "Local Qwen3-VL requires" in output.err
    assert INSTALL_COMMAND in output.err
    assert "Traceback" not in output.err
    dependencies.model_factory.assert_not_called()
    dependencies.processor_factory.assert_not_called()


def test_transformers_without_qwen_support_has_install_guidance(dependencies, image, monkeypatch, capsys):
    monkeypatch.delattr(sys.modules["transformers"], "Qwen3VLForConditionalGeneration")

    with pytest.raises(ImportError, match="Qwen3VLForConditionalGeneration") as error:
        LocalQwen3VL("style")(image)
    assert INSTALL_COMMAND in str(error.value)

    with pytest.raises(SystemExit) as error:
        require_local_qwen()
    assert error.value.code == 1
    assert INSTALL_COMMAND in capsys.readouterr().err
    dependencies.model_factory.assert_not_called()


def test_cli_dependency_check_does_not_hide_non_import_failures(monkeypatch):
    def broken_runtime():
        raise RuntimeError("broken native runtime")

    monkeypatch.setattr(qwen_local, "_load_dependencies", broken_runtime)

    with pytest.raises(RuntimeError, match="broken native runtime"):
        require_local_qwen()


@pytest.mark.parametrize("device, bf16, expected", [
    ("cpu", True, "float32"), ("mps", True, "float16"),
    ("cuda", True, "bfloat16"), ("cuda:1", False, "float16"),
])
def test_requested_device_and_supported_dtype_are_used(dependencies, image, device, bf16, expected):
    dependencies.torch.cuda.is_bf16_supported.return_value = bf16

    LocalQwen3VL("style", device=device)(image)

    dependencies.torch.device.assert_called_once_with(device)
    assert dependencies.model_factory.call_args.kwargs["dtype"] == expected
    target = dependencies.model.to.call_args.args[0]
    assert target.type == device.split(":")[0]
    dependencies.processor.return_value.to.assert_called_once_with(target)


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_unavailable_acceleration_fails_without_silent_fallback_or_weight_loading(dependencies, image, device):
    dependencies.torch.cuda.is_available.return_value = False
    dependencies.torch.backends.mps.is_available.return_value = False
    with pytest.raises(RuntimeError, match="not available"):
        LocalQwen3VL("style", device=device)(image)
    dependencies.model_factory.assert_not_called()
    dependencies.processor_factory.assert_not_called()


@pytest.mark.parametrize("eos, ending", [(99, 99), ([99, 100], 100)])
def test_answer_finishing_exactly_at_token_limit_is_accepted(dependencies, image, eos, ending):
    dependencies.model.generation_config.eos_token_id = eos
    dependencies.model.generate.return_value = np.array([[11, 12, 31, 32, ending]])
    assert LocalQwen3VL("style", max_tokens=3)(image) == "casual\nJeans and sneakers."


def test_truncated_answer_is_decoded_and_returned(dependencies, image):
    dependencies.model.generate.return_value = np.array([[11, 12, 31, 32, 33]])
    dependencies.processor.batch_decode.return_value = ["  casual\nJeans and  "]

    assert LocalQwen3VL("style", max_tokens=3)(image) == "casual\nJeans and"

    np.testing.assert_array_equal(dependencies.processor.batch_decode.call_args.args[0], [[31, 32, 33]])


@pytest.mark.parametrize("max_tokens", [3, 256])
def test_empty_answer_is_rejected_even_at_token_limit(dependencies, image, max_tokens):
    dependencies.model.generate.return_value = np.array([[11, 12, 31, 32, 33]])
    dependencies.processor.batch_decode.return_value = [" \n"]
    with pytest.raises(RuntimeError, match="no text"):
        LocalQwen3VL("style", max_tokens=max_tokens)(image)


def test_model_failures_propagate(dependencies, image):
    dependencies.model.generate.side_effect = RuntimeError("out of memory")
    with pytest.raises(RuntimeError, match="out of memory"):
        LocalQwen3VL("style")(image)
