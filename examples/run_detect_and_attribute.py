"""Detect objects and select a supplied attribute label per crop with local Qwen.

The pipeline returns (annotated_image, detections), with attribute, label, and
the full model response in detections.data. Annotation uses padded, image-clipped
boxes; invalid crop regions are dropped. People and the people-walking image
are selected by default, with fashion-style candidates; change class IDs and
candidates for other subjects.
Attribute and label arguments accept 'name: description'; annotations use
only the selected label name, while Qwen uses the descriptive text in its prompt.
Unrecognized answers are labeled 'unknown'; CLIP remains available separately.

Run from the repository root:
    python examples/run_detect_and_attribute.py
    python examples/run_detect_and_attribute.py --input photo.jpg --attribute activity --labels walking standing sitting
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
import sys
from typing import Any

import cv2
from inference.core.env import CLIP_MAX_BATCH_SIZE
import numpy as np
import numpy.typing as npt
import supervision as sv
from supervision.assets import download_assets

if __package__:
    from .common import ASSETS_DIR
    from .qwen_local import DEFAULT_MODEL_ID as DEFAULT_QWEN_MODEL_ID
    from .qwen_local import LocalQwen3VL, require_local_qwen
    from .run_detect_and_describe import (
        DEFAULT_CLASS_IDS,
        DEFAULT_DETECTION_MODEL_ID,
        DEFAULT_IMAGE_ASSET,
        build_prompt,
    )
else:
    from common import ASSETS_DIR
    from qwen_local import DEFAULT_MODEL_ID as DEFAULT_QWEN_MODEL_ID
    from qwen_local import LocalQwen3VL, require_local_qwen
    from run_detect_and_describe import (
        DEFAULT_CLASS_IDS,
        DEFAULT_DETECTION_MODEL_ID,
        DEFAULT_IMAGE_ASSET,
        build_prompt,
    )

from ml_pipes.core import Pipeline
from ml_pipes.operator import Operator
from ml_pipes.standard import Gather, Pick, Recall, Scatter, Select, Store
from ml_pipes.supervision import BoxAnnotator, Detections, ImageToArray, LabelAnnotator, PlotImage
from ml_pipes.supervision import inference as inference_ops
from ml_pipes.supervision.inference import RoboflowInference
from ml_pipes.vision import ConvertColorSpace, Decode, LoadFile

DEFAULT_CLIP_MODEL_ID = "clip/ViT-B-32"
CLIP_INSTALL_COMMAND = f'python -m pip install "inference[clip]=={version("inference")}"'
DEFAULT_ATTRIBUTE = "fashion style: overall visual style of the outfit"
DEFAULT_LABELS = (
    "casual: relaxed everyday clothing",
    "formal: office attire, such as a suit or dress shirt",
    "sportswear: athletic clothing, such as shorts or a windbreaker",
)


def parse_named_text(value: str, kind: str) -> tuple[str, str]:
    """Split the first colon into a result name and optional matching text."""
    name, separator, description = value.partition(":")
    name = name.strip()
    if not name:
        raise ValueError(f"A non-empty {kind} is required.")
    text = description.strip() if separator else name
    if not text:
        raise ValueError(f"A non-empty {kind} description is required after ':'.")
    return name, text


def prepare_candidates(attribute: str, labels: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    """Validate names/descriptions without loading models or discarding text."""
    attribute = attribute.strip()
    parse_named_text(attribute, "attribute")
    if isinstance(labels, str) or not labels:
        raise ValueError("At least one candidate label is required.")
    labels = tuple(label.strip() for label in labels)
    names = [parse_named_text(label, "candidate label")[0] for label in labels]
    if len(set(names)) != len(names):
        raise ValueError("Candidate labels must be unique.")
    return attribute, labels


def build_attribute_prompt(attribute: str, labels: tuple[str, ...]) -> str:
    """Give Qwen the label names and descriptions, separate from output parsing."""
    attribute, labels = prepare_candidates(attribute, labels)
    name, description = parse_named_text(attribute, "attribute")
    candidates = [
        {"name": label_name, "description": label_description}
        for label_name, label_description in (
            parse_named_text(label, "candidate label") for label in labels
        )
    ]
    return build_prompt(
        "Assign the requested attribute to the visible subject in this image crop.\n"
        f"Attribute: {json.dumps({'name': name, 'description': description}, ensure_ascii=False)}\n"
        f"Allowed labels: {json.dumps(candidates, ensure_ascii=False)}\n"
        "Choose one supplied label name and copy it exactly onto the first line, "
        "without quotes, prefixes, or the label description. Do not invent labels. "
        "If none is supported by visible evidence, answer 'unknown'."
    )


@dataclass(frozen=True)
class Classification:
    label: str
    similarity: float


def _load_clip_dependencies() -> None:
    """Check CLIP's backend without loading weights or affecting Qwen usage."""
    try:
        from inference.models.clip import Clip
    except ImportError as error:
        raise ImportError(
            "CLIPClassify requires the optional Roboflow Inference CLIP dependencies. "
            f"A required dependency is missing or incompatible: {error}\n"
            f"Install them first with: {CLIP_INSTALL_COMMAND}"
        ) from error


def require_clip() -> None:
    """Check an explicit CLIP experiment's dependencies with a clean CLI exit."""
    try:
        _load_clip_dependencies()
    except ImportError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from None


def unit_embeddings(embeddings: npt.ArrayLike, rows: int) -> npt.NDArray[np.float64]:
    """Validate model output and normalize vectors for cosine similarity."""
    vectors = np.asarray(embeddings, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] != rows or vectors.shape[1] == 0:
        raise ValueError("CLIP must return one non-empty embedding vector per input.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("CLIP embeddings must be finite and non-zero.")
    return vectors / norms


@Operator
class CLIPClassify:
    """Select a candidate by cosine similarity using Roboflow Inference CLIP.

    The model and normalized candidate text embeddings are cached on first use.
    BGR arrays are passed unchanged to Roboflow Inference, whose model
    preprocessing handles BGR-to-RGB conversion.
    Native CLIP preprocessing handles image normalization; any additional crop
    resizing belongs in the pipeline. Scores are not calibrated probabilities.
    Install CLIP separately; missing backend dependencies raise an actionable
    ImportError on first use. CLI experiments can call require_clip() upfront.
    """

    def __init__(
        self,
        attribute: str,
        labels: tuple[str, ...],
        *,
        model_id: str = DEFAULT_CLIP_MODEL_ID,
        api_key: str | None = None,
    ) -> None:
        attribute, labels = prepare_candidates(attribute, labels)
        self.attribute, attribute_text = parse_named_text(attribute, "attribute")
        candidates = [parse_named_text(label, "candidate label") for label in labels]
        self.labels = tuple(name for name, _ in candidates)
        self.prompts = tuple(
            f"A photo of a subject. {attribute_text}: {text}." for _, text in candidates
        )
        self.model_id = model_id
        self.api_key = api_key
        self._model: Any | None = None
        self._text_embeddings: npt.NDArray[np.float64] | None = None

    def __call__(self, image: npt.NDArray[np.uint8]) -> Classification:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or image.size == 0:
            raise ValueError("CLIPClassify expects a non-empty BGR HWC uint8 image.")
        if self._model is None:
            _load_clip_dependencies()
            kwargs = {"model_id": self.model_id}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._model = inference_ops.get_model(**kwargs)
        if self._text_embeddings is None:
            batches = []
            for start in range(0, len(self.prompts), CLIP_MAX_BATCH_SIZE):
                prompts = list(self.prompts[start:start + CLIP_MAX_BATCH_SIZE])
                batches.append(unit_embeddings(self._model.embed_text(prompts), rows=len(prompts)))
            self._text_embeddings = np.concatenate(batches, axis=0)

        image_embeddings = unit_embeddings(self._model.embed_image(image), rows=1)
        if image_embeddings.shape[1] != self._text_embeddings.shape[1]:
            raise ValueError("CLIP image and text embeddings must have the same dimensions.")
        similarities = self._text_embeddings @ image_embeddings[0]
        best = int(np.argmax(similarities))
        return Classification(self.labels[best], float(np.clip(similarities[best], -1, 1)))


@Operator
class ParseAttributeLabels:
    """Parse crop responses into aligned data fields without accessing detections."""

    def __init__(self, attribute: str, labels: tuple[str, ...]) -> None:
        attribute, labels = prepare_candidates(attribute, labels)
        self.attribute, _ = parse_named_text(attribute, "attribute")
        self.labels = {parse_named_text(label, "candidate label")[0] for label in labels}

    def __call__(self, responses: list[str]) -> dict[str, list[str]]:
        labels = []
        for response in responses:
            first_line = response.strip().splitlines()[0] if response.strip() else ""
            label = first_line.strip()
            labels.append(label if label in self.labels else "unknown")
        return {
            "attribute": [self.attribute] * len(responses),
            "label": labels,
            "response": responses,
        }


def build_pipeline(
    *,
    attribute: str = DEFAULT_ATTRIBUTE,
    labels: tuple[str, ...] = DEFAULT_LABELS,
    detection_model_id: str = DEFAULT_DETECTION_MODEL_ID,
    class_ids: tuple[int, ...] = DEFAULT_CLASS_IDS,
    confidence: float = 0.5,
    padding: float = 0.05,
    qwen_model_id: str = DEFAULT_QWEN_MODEL_ID,
    device: str = "cpu",
    max_tokens: int = 256,
    api_key: str | None = None,
) -> Pipeline[str | Path, tuple[npt.NDArray[np.uint8], sv.Detections]]:
    """Detect, assign a supplied label with local Qwen, annotate, and display."""
    attribute, labels = prepare_candidates(attribute, labels)
    if not class_ids:
        raise ValueError("At least one class ID is required.")
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1.")
    if not np.isfinite(padding) or padding < 0:
        raise ValueError("padding must be finite and non-negative.")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer.")
    prepared_prompt = build_attribute_prompt(attribute, labels)
    return Pipeline(
        [
            LoadFile(),
            Decode(),
            ImageToArray(),
            Store("source_image"),
            RoboflowInference(model_id=detection_model_id, api_key=api_key, confidence=confidence),
            Select(0),
            Detections.FromInference(),
            Detections.Filter(lambda detections: np.isin(detections.class_id, class_ids)),
            Detections.Filter(lambda detections: detections.confidence >= confidence),
            Detections.NMS(threshold=0.5),
            Detections.ScaleBoxes(1 + 2 * padding),
            Recall("source_image", prepend=True),
            Detections.ClipBoxes(rounding="outward", invalid="drop"),
            Detections.Crop(),
            Store("detections", source=0),
            Pick(1),
            Scatter(max_concurrency=1),
            ConvertColorSpace(output_color_space="RGB"),
            LocalQwen3VL(
                prompt=prepared_prompt, model_id=qwen_model_id,
                device=device, max_tokens=max_tokens,
            ),
            Gather(),
            ParseAttributeLabels(attribute=attribute, labels=labels),
            Recall("detections", prepend=True),
            Detections.AttachData(),
            Recall("source_image", prepend=True),
            BoxAnnotator(),
            LabelAnnotator(
                label_formatter=lambda detection: detection.data["label"],
            ),
            PlotImage(at=0),
        ],
        auto_validate=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Source image path. Defaults to Supervision's PEOPLE_WALKING sample (downloaded if missing).",
    )
    parser.add_argument(
        "--attribute", default=DEFAULT_ATTRIBUTE, metavar="NAME[: DESCRIPTION]",
        help="Attribute name, optionally followed by ':' and a description; quote the entire argument. "
             f"Default: {DEFAULT_ATTRIBUTE}.",
    )
    parser.add_argument(
        "--labels", nargs="+", default=DEFAULT_LABELS, metavar="LABEL[: DESCRIPTION]",
        help="Candidate labels with optional descriptions; quote each item, without separating commas. "
             f"Default: {'; '.join(DEFAULT_LABELS)}.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON result path.")
    parser.add_argument("--output-image", type=Path, help="Optional annotated image path.")
    parser.add_argument("--detection-model-id", default=DEFAULT_DETECTION_MODEL_ID)
    parser.add_argument("--class-ids", type=int, nargs="+", default=DEFAULT_CLASS_IDS)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--padding", type=float, default=0.05)
    parser.add_argument(
        "--qwen-model-id", default=DEFAULT_QWEN_MODEL_ID,
        help=f"Local model ID or path (default: {DEFAULT_QWEN_MODEL_ID}).",
    )
    parser.add_argument("--device", default="cpu", help="Local Qwen device: cpu (default), mps, cuda, or cuda:N.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum generated answer tokens (default: 256).")
    parser.add_argument("--api-key", help="Optional Roboflow key for detection; ROBOFLOW_API_KEY also works.")
    args = parser.parse_args()
    require_local_qwen()
    input_path = args.input or Path(download_assets(DEFAULT_IMAGE_ASSET, directory=ASSETS_DIR))

    pipeline = build_pipeline(
        attribute=args.attribute, labels=tuple(args.labels),
        detection_model_id=args.detection_model_id, class_ids=tuple(args.class_ids),
        confidence=args.confidence, padding=args.padding,
        qwen_model_id=args.qwen_model_id, device=args.device, max_tokens=args.max_tokens,
        api_key=args.api_key,
    )
    pipeline.validate()
    annotated_image, detections = pipeline(input_path)
    results = [
        {
            "detection_index": index,
            "xyxy": detections.xyxy[index].tolist(),
            "class_id": int(detections.class_id[index]),
            "confidence": float(detections.confidence[index]),
            "attribute": detections.data["attribute"][index],
            "label": detections.data["label"][index],
            "response": detections.data["response"][index],
        }
        for index in range(len(detections))
    ]
    output = json.dumps(results, indent=2, ensure_ascii=False)
    print(output)
    if args.output is not None:
        args.output.write_text(output + "\n", encoding="utf-8")
    if args.output_image is not None:
        if not cv2.imwrite(str(args.output_image), annotated_image):
            raise OSError(f"Failed to save annotated image to {args.output_image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
