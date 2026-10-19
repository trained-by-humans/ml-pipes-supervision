"""Example-only local Qwen3-VL inference through Hugging Face Transformers."""
from __future__ import annotations

import sys
from typing import Any

import numpy as np

from ml_pipes.operator import Operator
from ml_pipes.vision import ImagePayload

__all__ = ["LocalQwen3VL", "require_local_qwen"]

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
INSTALL_COMMAND = 'python -m pip install "transformers>=4.57.0,<6" "torch>=2.6" "Pillow>=10"'


def _load_dependencies() -> tuple[Any, Any, Any, Any]:
    """Import the example's runtime without downloading or loading weights."""
    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as error:
        raise ImportError(
            "Local Qwen3-VL requires Transformers with Qwen3-VL support, PyTorch, "
            f"and Pillow. A required dependency is missing or incompatible: {error}\n"
            f"Install them first with: {INSTALL_COMMAND}"
        ) from error
    return torch, Image, AutoProcessor, Qwen3VLForConditionalGeneration


def require_local_qwen() -> None:
    """Stop a CLI example with an actionable message if Qwen cannot be imported."""
    try:
        _load_dependencies()
    except ImportError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from None


@Operator
class LocalQwen3VL:
    """Answer a prompt about an RGB image, loading weights once on first use.

    Color conversion and crop resizing belong in the pipeline. The model's
    processor handles native normalization, patch alignment, and tokenization.
    CPU uses float32; CUDA uses bfloat16 when supported, otherwise float16;
    MPS uses float16. No automatic device fallback or quantization is applied.
    Calls must be sequential; reuse one instance across crops and frames.
    Non-empty text is returned even when the generation token limit is reached.
    """

    def __init__(
        self,
        prompt: str,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cpu",
        max_tokens: int = 256,
    ) -> None:
        if not prompt.strip():
            raise ValueError("LocalQwen3VL requires a non-empty prompt.")
        if not model_id.strip():
            raise ValueError("LocalQwen3VL requires a non-empty model ID or local path.")
        if device not in ("cpu", "mps", "cuda") and not (
            device.startswith("cuda:") and device[5:].isdigit()
        ):
            raise ValueError("device must be 'cpu', 'mps', 'cuda', or 'cuda:N'.")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer.")
        self.prompt = prompt
        self.model_id = model_id
        self.device = device
        self.max_tokens = max_tokens
        self._model: Any | None = None
        self._processor: Any | None = None

    def __call__(self, image: ImagePayload) -> str:
        if (
            not isinstance(image, ImagePayload)
            or image.color_space != "RGB"
            or image.layout != "HWC"
            or image.array.ndim != 3
            or image.array.shape[2] != 3
            or image.array.dtype != np.uint8
            or image.array.size == 0
        ):
            raise ValueError("LocalQwen3VL expects a non-empty RGB HWC uint8 ImagePayload.")

        torch, Image, AutoProcessor, Qwen3VLForConditionalGeneration = _load_dependencies()

        device = torch.device(self.device)
        if self._model is None:
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available; select --device cpu or install a CUDA-enabled PyTorch.")
            if device.type == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("MPS is not available; select --device cpu.")
            dtype = torch.float32
            if device.type == "cuda":
                with torch.cuda.device(device):
                    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            elif device.type == "mps":
                dtype = torch.float16
            processor = AutoProcessor.from_pretrained(self.model_id)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id, dtype=dtype, attn_implementation="eager",
            ).to(device).eval()
            self._processor = processor
            self._model = model

        rgb_image = Image.fromarray(image.array)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": rgb_image},
                {"type": "text", "text": self.prompt},
            ],
        }]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[text], images=[rgb_image], padding=True, return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            sequences = self._model.generate(
                **inputs, max_new_tokens=self.max_tokens, do_sample=False,
                return_dict_in_generate=False,
            )
        generated = sequences[:, inputs["input_ids"].shape[-1]:]
        answer = self._processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()
        if not answer:
            raise RuntimeError("LocalQwen3VL returned no text.")
        return answer
