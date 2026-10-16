"""Bedrock request/response and image boundaries, without remote inference."""
import sys
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from ml_pipes.supervision.bedrock import DEFAULT_MODEL_ID, Qwen3VL


@pytest.fixture
def client():
    mock = MagicMock(spec=["converse"])
    mock.converse.return_value = {
        "output": {"message": {"content": [
            {"text": "casual"}, {"reasoningContent": {}}, {"text": "jeans and sneakers"},
        ]}},
        "stopReason": "end_turn",
    }
    return mock


def decode_request_image(client):
    content = client.converse.call_args.kwargs["messages"][0]["content"]
    image = content[0]["image"]
    assert image["format"] == "png"
    return cv2.imdecode(
        np.frombuffer(image["source"]["bytes"], dtype=np.uint8), cv2.IMREAD_COLOR,
    )


def test_converse_receives_lossless_bgr_crop_prompt_and_options(client):
    image = np.full((8, 12, 3), [10, 20, 230], dtype=np.uint8)[:, ::-1]
    original = image.copy()
    operator = Qwen3VL("Describe the outfit.", client=client, max_tokens=100)

    answer = operator(image)

    assert answer == "casual\njeans and sneakers"
    request = client.converse.call_args.kwargs
    assert request["modelId"] == DEFAULT_MODEL_ID == "qwen.qwen3-vl-235b-a22b"
    assert request["inferenceConfig"] == {"maxTokens": 100, "temperature": 0.0}
    assert request["messages"][0]["role"] == "user"
    assert request["messages"][0]["content"][1] == {"text": "Describe the outfit."}
    np.testing.assert_array_equal(decode_request_image(client), original)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("shape", [
    (2048, 512, 3),
    (512, 2048, 3),
    (2048, 1, 3),
])
def test_operator_preserves_large_images_without_resizing(client, shape):
    image = np.full(shape, [10, 20, 230], dtype=np.uint8)
    Qwen3VL("style", client=client)(image)
    np.testing.assert_array_equal(decode_request_image(client), image)


@pytest.mark.parametrize("image", [
    np.empty((0, 10, 3), dtype=np.uint8),
    np.zeros((10, 10), dtype=np.uint8),
    np.zeros((10, 10, 4), dtype=np.uint8),
    np.zeros((10, 10, 3), dtype=np.float32),
])
def test_invalid_images_are_rejected_before_api_calls(client, image):
    with pytest.raises(ValueError, match="BGR HWC uint8"):
        Qwen3VL("style", client=client)(image)
    client.converse.assert_not_called()


@pytest.mark.parametrize("stop_reason", [
    "max_tokens", "guardrail_intervened", "content_filtered", None,
])
def test_incomplete_or_blocked_answers_are_not_silently_accepted(client, stop_reason):
    client.converse.return_value["stopReason"] = stop_reason
    with pytest.raises(RuntimeError, match="did not finish"):
        Qwen3VL("style", client=client)(np.zeros((4, 4, 3), dtype=np.uint8))


def test_empty_text_is_reported(client):
    client.converse.return_value["output"]["message"]["content"] = [{"text": " "}]
    with pytest.raises(RuntimeError, match="no text"):
        Qwen3VL("style", client=client)(np.zeros((4, 4, 3), dtype=np.uint8))


@pytest.mark.parametrize("kwargs", [{"prompt": " "}, {"prompt": "style", "max_tokens": 0}])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        Qwen3VL(**kwargs)


def test_sdk_is_optional_until_first_inference(monkeypatch):
    monkeypatch.setitem(sys.modules, "boto3", None)
    operator = Qwen3VL("style")
    assert operator._client is None
    with pytest.raises(ModuleNotFoundError, match=r"ml-pipes-supervision\[bedrock\]"):
        operator(np.zeros((4, 4, 3), dtype=np.uint8))


def test_sdk_client_is_created_lazily_with_region_and_reused(monkeypatch, client):
    boto3 = pytest.importorskip("boto3")
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(boto3, "client", factory)
    operator = Qwen3VL("style", region_name="us-west-2")
    factory.assert_not_called()

    for _ in range(2):
        operator(np.zeros((4, 4, 3), dtype=np.uint8))

    factory.assert_called_once()
    assert factory.call_args.args == ("bedrock-runtime",)
    assert factory.call_args.kwargs["region_name"] == "us-west-2"
    assert factory.call_args.kwargs["config"].read_timeout == 120
    assert client.converse.call_count == 2


def test_bedrock_failures_propagate_instead_of_becoming_style_labels(client):
    client.converse.side_effect = RuntimeError("Access denied")
    with pytest.raises(RuntimeError, match="Access denied"):
        Qwen3VL("style", client=client)(np.zeros((4, 4, 3), dtype=np.uint8))


def test_request_matches_real_bedrock_sdk_schema(monkeypatch):
    boto3 = pytest.importorskip("boto3")
    from botocore.stub import Stubber

    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    sdk_client = boto3.client(
        "bedrock-runtime", region_name="us-east-1",
        aws_access_key_id="testing", aws_secret_access_key="testing",
    )
    with Stubber(sdk_client) as stubber:
        stubber.add_response("converse", {
            "output": {"message": {"role": "assistant", "content": [{"text": "casual"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 1, "totalTokens": 11},
            "metrics": {"latencyMs": 1},
        })
        result = Qwen3VL("style", client=sdk_client)(np.zeros((4, 4, 3), dtype=np.uint8))
        assert result == "casual"
        stubber.assert_no_pending_responses()
