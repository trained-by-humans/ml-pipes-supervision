"""Optional Amazon Bedrock vision-language inference operators."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from ml_pipes.operator import Operator

__all__ = ["Qwen3VL"]

DEFAULT_MODEL_ID = "qwen.qwen3-vl-235b-a22b"


@Operator
class Qwen3VL:
    """Send a BGR uint8 crop and a prompt to Qwen3-VL through Bedrock Converse.

    Authentication uses the SDK's normal credential chain, including
    ``AWS_BEARER_TOKEN_BEDROCK`` for Bedrock API keys. The optional SDK and
    client are loaded only on the first call. Images are PNG-encoded without
    resizing; configure any image preprocessing explicitly in the pipeline.
    """

    def __init__(
        self,
        prompt: str,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        region_name: str | None = None,
        max_tokens: int = 256,
        client: Any | None = None,
    ) -> None:
        if not prompt.strip():
            raise ValueError("Qwen3VL requires a non-empty prompt.")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive.")
        self.prompt = prompt
        self.model_id = model_id
        self.region_name = region_name
        self.max_tokens = max_tokens
        self._client = client

    def __call__(self, image: npt.NDArray[np.uint8]) -> str:
        if (
            image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
            or image.size == 0
        ):
            raise ValueError("Qwen3VL expects a non-empty BGR HWC uint8 image.")

        encoded, png = cv2.imencode(".png", image)
        if not encoded:
            raise ValueError("Failed to encode the image for Bedrock.")

        if self._client is None:
            try:
                import boto3
                from botocore.config import Config
            except ModuleNotFoundError as error:
                if error.name in ("boto3", "botocore"):
                    raise ModuleNotFoundError(
                        "Qwen3VL requires the optional Bedrock dependency. "
                        "Install 'ml-pipes-supervision[bedrock]'."
                    ) from error
                raise
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.region_name,
                config=Config(read_timeout=120),
            )

        response = self._client.converse(
            modelId=self.model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"image": {"format": "png", "source": {"bytes": png.tobytes()}}},
                    {"text": self.prompt},
                ],
            }],
            inferenceConfig={"maxTokens": self.max_tokens, "temperature": 0.0},
        )
        if response.get("stopReason") not in ("end_turn", "stop_sequence"):
            raise RuntimeError(
                f"Qwen3VL did not finish its answer: {response.get('stopReason')!r}."
            )
        content = response["output"]["message"]["content"]
        answer = "\n".join(block["text"] for block in content if "text" in block).strip()
        if not answer:
            raise RuntimeError("Qwen3VL returned no text.")
        return answer
