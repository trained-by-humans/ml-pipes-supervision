"""Detect objects and assign multiple visible attributes per crop with local Qwen.

One VLM query per crop requests a JSON object. ParseAttributes converts the
responses to aligned columns named after the attributes, plus the raw response.
Single-value fields contain label strings; multi-value fields preserve lists.
Missing or invalid values become 'unknown' (['unknown'] for multi-value fields).
JSON is requested by the prompt, not enforced through constrained decoding.
Annotation uses padded, image-clipped boxes; invalid crop regions are dropped.
The defaults select people and assign fashion style, activity, and accessories.

Run beside the downloaded helper files (or under examples/ in a checkout):
    python run_detect_and_attribute.py
    python run_detect_and_attribute.py --input photo.jpg --attributes attributes.json
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import cv2
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
    )
else:
    from common import ASSETS_DIR
    from qwen_local import DEFAULT_MODEL_ID as DEFAULT_QWEN_MODEL_ID
    from qwen_local import LocalQwen3VL, require_local_qwen
    from run_detect_and_describe import (
        DEFAULT_CLASS_IDS,
        DEFAULT_DETECTION_MODEL_ID,
        DEFAULT_IMAGE_ASSET,
    )

from ml_pipes.core import Pipeline
from ml_pipes.operator import Operator
from ml_pipes.standard import Gather, Pick, Recall, Scatter, Select, Store
from ml_pipes.supervision import BoxAnnotator, Detections, ImageToArray, LabelAnnotator, PlotImage
from ml_pipes.supervision.inference import RoboflowInference
from ml_pipes.vision import ConvertColorSpace, Decode, LoadFile


def parse_named_text(value: str, kind: str) -> tuple[str, str]:
    """Split the first colon into a result name and optional matching text."""
    if not isinstance(value, str):
        raise ValueError(f"Each {kind} must be a string.")
    name, separator, description = value.partition(":")
    name = name.strip()
    if not name:
        raise ValueError(f"A non-empty {kind} is required.")
    text = description.strip() if separator else name
    if not text:
        raise ValueError(f"A non-empty {kind} description is required after ':'.")
    return name, text


@dataclass(frozen=True)
class Attribute:
    """Example-only attribute configuration, not a package operator.

    Labels accept 'name: description'. ``multiple=True`` requests a list of
    labels, with [] meaning none visible and ['unknown'] meaning indeterminate.
    """

    labels: tuple[str, ...]
    description: str = ""
    multiple: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.labels, (tuple, list)) or not self.labels:
            raise ValueError("At least one candidate label is required.")
        names = [parse_named_text(label, "candidate label")[0] for label in self.labels]
        if len(set(names)) != len(names):
            raise ValueError("Candidate labels must be unique.")
        if not isinstance(self.description, str):
            raise ValueError("Attribute description must be a string.")
        if not isinstance(self.multiple, bool):
            raise ValueError("multiple must be a bool.")
        object.__setattr__(self, "labels", tuple(label.strip() for label in self.labels))
        object.__setattr__(self, "description", self.description.strip())


DEFAULT_ATTRIBUTES = {
    "fashion_style": Attribute(
        description="Overall visual style of the outfit",
        labels=(
            "casual: relaxed everyday clothing",
            "formal: office attire, such as a suit or dress shirt",
            "sportswear: athletic clothing, such as shorts or a windbreaker",
        ),
    ),
    "activity": Attribute(
        description="Visible body posture or movement",
        labels=("walking: moving on foot", "standing: upright and stationary", "sitting: seated"),
    ),
    "accessories": Attribute(
        labels=("backpack", "shoulder_bag", "handbag", "shopping_bag", "hat", "glasses", "phone"),
        multiple=True,
    ),
}


def prepare_attributes(attributes: Mapping[str, Attribute]) -> dict[str, Attribute]:
    """Validate output field names before loading any models."""
    if not isinstance(attributes, Mapping) or not attributes:
        raise ValueError("At least one attribute is required.")
    for name, attribute in attributes.items():
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise ValueError("Attribute names must be non-empty strings without surrounding whitespace.")
        if name == "response":
            raise ValueError("'response' is reserved for raw model responses.")
        if not isinstance(attribute, Attribute):
            raise ValueError("Each attribute must be an Attribute configuration.")
    return dict(attributes)


def build_attribute_prompt(attributes: Mapping[str, Attribute]) -> str:
    """Request one JSON object per subject; generation remains unconstrained."""
    attributes = prepare_attributes(attributes)
    specifications = {
        name: {
            "description": attribute.description or name,
            "multiple": attribute.multiple,
            "labels": [
                {"name": label_name, "description": label_description}
                for label_name, label_description in (
                    parse_named_text(label, "candidate label") for label in attribute.labels
                )
            ],
        }
        for name, attribute in attributes.items()
    }
    example = {
        name: ["unknown"] if attribute.multiple else "unknown"
        for name, attribute in attributes.items()
    }
    return (
        "Assign every requested attribute to the visible subject centered in this image crop.\n"
        f"Attributes: {json.dumps(specifications, ensure_ascii=False)}\n"
        "Return exactly one JSON object using the attribute names as keys. "
        "For each multiple=false attribute, return one exact supplied label name as a string. "
        "For each multiple=true attribute, return an array of all visibly supported label names; "
        "use [] only when no listed items are visible. Do not invent label names or return descriptions. "
        "Use 'unknown' for an indeterminate single value, or ['unknown'] for an indeterminate array. "
        "Assess each attribute independently. Use only visible evidence, ignore neighboring people, "
        "and do not guess hidden personal attributes from appearance. "
        "If the crop is too small, blurry, or obscured, use the unknown value for affected attributes. "
        "Do not add markdown, commentary, or other keys.\n"
        f"JSON shape (replace unknown values with supported labels): {json.dumps(example)}"
    )


def unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys instead of silently accepting the last value."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


@Operator
class ParseAttributes:
    """Turn ordered JSON responses into named, aligned detection-data columns.

    Missing/invalid fields become unknown independently. Malformed JSON or
    duplicate keys make the entire row unknown. Extra fields are ignored.
    Multi-value lists retain their order and nesting. Raw responses are retained.
    """

    def __init__(self, attributes: Mapping[str, Attribute]) -> None:
        self.attributes = prepare_attributes(attributes)
        self.labels = {
            name: {parse_named_text(label, "candidate label")[0] for label in attribute.labels}
            for name, attribute in self.attributes.items()
        }

    def __call__(self, responses: list[str]) -> dict[str, list[Any]]:
        fields: dict[str, list[Any]] = {name: [] for name in self.attributes}
        for response in responses:
            text = response.strip()
            if text.startswith("```") and text.endswith("```"):
                lines = text.splitlines()
                if lines[0].strip() in ("```", "```json"):
                    text = "\n".join(lines[1:-1])
            try:
                payload = json.loads(text, object_pairs_hook=unique_json_object)
                if not isinstance(payload, dict):
                    payload = {}
            except ValueError:
                payload = {}
            for name, attribute in self.attributes.items():
                value = payload.get(name)
                allowed = self.labels[name] | {"unknown"}
                if attribute.multiple:
                    valid = isinstance(value, list) and all(
                        isinstance(item, str) and item in allowed for item in value
                    ) and ("unknown" not in value or value == ["unknown"])
                    fields[name].append(value if valid else ["unknown"])
                else:
                    fields[name].append(value if isinstance(value, str) and value in allowed else "unknown")
        fields["response"] = responses
        return fields


def format_attributes(data: Mapping[str, Any], attributes: Mapping[str, Attribute]) -> str:
    """Compose an annotation without flattening stored multi-value data."""
    parts = []
    for name in attributes:
        value = data[name]
        if isinstance(value, list):
            text = ", ".join(value) if value else "none"
        else:
            text = str(value)
        parts.append(text if len(attributes) == 1 else f"{name}: {text}")
    return " | ".join(parts)


def build_pipeline(
    *,
    attributes: Mapping[str, Attribute] | None = None,
    detection_model_id: str = DEFAULT_DETECTION_MODEL_ID,
    class_ids: tuple[int, ...] = DEFAULT_CLASS_IDS,
    confidence: float = 0.5,
    padding: float = 0.05,
    qwen_model_id: str = DEFAULT_QWEN_MODEL_ID,
    device: str = "cpu",
    max_tokens: int = 256,
    api_key: str | None = None,
) -> Pipeline[str | Path, tuple[npt.NDArray[np.uint8], sv.Detections]]:
    """Detect, query all attributes once per crop, parse, attach, and annotate."""
    attributes = prepare_attributes(DEFAULT_ATTRIBUTES if attributes is None else attributes)
    if not class_ids:
        raise ValueError("At least one class ID is required.")
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1.")
    if not np.isfinite(padding) or padding < 0:
        raise ValueError("padding must be finite and non-negative.")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer.")
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
            Detections.ClipBoxes(rounding="outward", invalid_policy="drop"),
            Detections.Crop(),
            Store("detections", source=0),
            Pick(1),
            Scatter(max_concurrency=1),
            ConvertColorSpace(output_color_space="RGB"),
            LocalQwen3VL(
                prompt=build_attribute_prompt(attributes), model_id=qwen_model_id,
                device=device, max_tokens=max_tokens,
            ),
            Gather(),
            ParseAttributes(attributes),
            Recall("detections", prepend=True),
            Detections.AttachData(),
            Recall("source_image", prepend=True),
            BoxAnnotator(),
            LabelAnnotator(
                label_formatter=lambda detection: format_attributes(detection.data, attributes),
                smart_position=True, max_line_length=40,
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
        "--attributes", type=Path, metavar="CONFIG.json",
        help="JSON mapping attribute names to labels, description, and multiple (default: fashion_style, activity, accessories).",
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
    attributes = DEFAULT_ATTRIBUTES
    if args.attributes is not None:
        try:
            config = json.loads(args.attributes.read_text(encoding="utf-8"), object_pairs_hook=unique_json_object)
            if not isinstance(config, dict) or not config:
                raise ValueError("Attribute config must be a non-empty JSON object.")
            attributes = prepare_attributes({name: Attribute(**options) for name, options in config.items()})
        except (OSError, TypeError, ValueError) as error:
            parser.error(str(error))
    require_local_qwen()
    input_path = args.input or Path(download_assets(DEFAULT_IMAGE_ASSET, directory=ASSETS_DIR))

    pipeline = build_pipeline(
        attributes=attributes,
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
            "attributes": {name: detections.data[name][index] for name in attributes},
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
