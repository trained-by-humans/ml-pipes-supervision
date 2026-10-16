"""Detect objects, answer a prompt per crop with Qwen3-VL, and annotate them.

The pipeline returns (annotated_image, detections), with full answers in
detections.data["response"]. People are selected by default; change class IDs
and the prompt for other subjects and tasks.

Run from the repository root (see README_detect_and_describe.md for setup):
    python examples/run_detect_and_describe.py
    python examples/run_detect_and_describe.py --input path/to/photo.jpg
    python examples/run_detect_and_describe.py --input photo.jpg --prompt "Describe the clothing colors."
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import supervision as sv
from supervision.assets import ImageAssets, download_assets

if __package__:
    from .common import ASSETS_DIR
else:
    from common import ASSETS_DIR

from ml_pipes.core import Pipeline
from ml_pipes.operator import Operator
from ml_pipes.standard import Gather, Pick, Recall, Scatter, Select, Store
from ml_pipes.supervision import (
    BoxAnnotator,
    Detections,
    ImageToArray,
    LabelAnnotator,
    PlotImage,
)
from ml_pipes.supervision.bedrock import DEFAULT_MODEL_ID as DEFAULT_QWEN_MODEL_ID
from ml_pipes.supervision.bedrock import Qwen3VL
from ml_pipes.supervision.inference import RoboflowInference
from ml_pipes.vision import Decode, ImagePayload, LoadFile, Resize

DEFAULT_IMAGE_ASSET = ImageAssets.PEOPLE_WALKING
DEFAULT_DETECTION_MODEL_ID = "rfdetr-small"
# Pretrained RF-DETR detection aliases use COCO category ID 1 for person.
DEFAULT_CLASS_IDS = (1,)
DEFAULT_PROMPT = "Describe the visible subject's appearance and activity."
RESPONSE_INSTRUCTIONS = (
    "Use only visible evidence in this image crop. "
    "Put a short answer or label on the first line, followed by a brief explanation "
    "on subsequent lines. "
    "If the requested attribute cannot be determined visually, or the crop is too "
    "small, blurry, or obscured, put 'unknown' on the first line and explain why. "
    "Do not guess hidden personal attributes from appearance."
)


def build_prompt(prompt: str) -> str:
    """Combine a task with shared grounding and annotation-format instructions."""
    if not prompt.strip():
        raise ValueError("A non-empty task prompt is required.")
    return f"Task:\n{prompt.strip()}\n\n{RESPONSE_INSTRUCTIONS}"


@Operator
class PadBoxes:
    """Expand separate crop boxes, preserving the original detections.

    Padding is a fraction of each box's width/height on each side. This step
    does not need an image or constrain coordinates to image boundaries.
    """

    def __init__(self, padding: float = 0.05) -> None:
        if not np.isfinite(padding) or padding < 0:
            raise ValueError("padding must be finite and non-negative.")
        self.padding = padding

    def __call__(
        self,
        detections: sv.Detections,
    ) -> tuple[sv.Detections, npt.NDArray[np.float64]]:
        boxes = detections.xyxy.astype(np.float64, copy=True)
        padding = (boxes[:, 2:] - boxes[:, :2]) * self.padding
        boxes[:, :2] -= padding
        boxes[:, 2:] += padding
        return detections, boxes


@Operator
class ClipBoxes:
    """Clip crop boxes to the image and drop unusable boxes with their detections.

    This step also works without padding: pass detections.xyxy as crop_boxes.
    Original detection coordinates remain unchanged for annotation.
    """

    def __call__(
        self,
        image: npt.NDArray[np.uint8],
        detections: sv.Detections,
        crop_boxes: npt.NDArray[np.number],
    ) -> tuple[npt.NDArray[np.uint8], sv.Detections, npt.NDArray[np.number]]:
        if crop_boxes.shape != (len(detections), 4):
            raise ValueError("Each detection must have exactly one crop box.")
        height, width = image.shape[:2]
        boxes = sv.clip_boxes(crop_boxes, resolution_wh=(width, height))
        keep = (
            np.isfinite(crop_boxes).all(axis=1)
            & (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )
        return image, detections[keep], boxes[keep]


@Operator
class CropDetections:
    """Slice prepared in-bounds boxes into independent BGR image payloads.

    Expects finite, non-empty crop boxes from ClipBoxes (or equivalent input).
    This step performs no padding or clipping. Fractional edges are rounded
    outward so the crop includes the whole prepared box.
    """

    def __call__(
        self,
        image: npt.NDArray[np.uint8],
        detections: sv.Detections,
        crop_boxes: npt.NDArray[np.number],
    ) -> tuple[sv.Detections, list[ImagePayload]]:
        if crop_boxes.shape != (len(detections), 4):
            raise ValueError("Each detection must have exactly one crop box.")
        crops = []
        for x1, y1, x2, y2 in crop_boxes:
            left, top = int(np.floor(x1)), int(np.floor(y1))
            right, bottom = int(np.ceil(x2)), int(np.ceil(y2))
            crops.append(ImagePayload(array=image[top:bottom, left:right].copy()))
        return detections, crops


@Operator
class AttachResponses:
    """Attach aligned full responses for use by the label formatter."""

    def __call__(
        self,
        detections: sv.Detections,
        responses: list[str],
    ) -> sv.Detections:
        if len(detections) != len(responses):
            raise ValueError("Each detection must have exactly one model response.")
        return replace(
            detections,
            data={**detections.data, "response": np.asarray(responses, dtype=str)},
        )


def build_pipeline(
    *,
    detection_model_id: str = DEFAULT_DETECTION_MODEL_ID,
    class_ids: tuple[int, ...] = DEFAULT_CLASS_IDS,
    confidence: float = 0.5,
    padding: float = 0.05,
    prompt: str = DEFAULT_PROMPT,
    qwen_model_id: str = DEFAULT_QWEN_MODEL_ID,
    region_name: str | None = None,
) -> Pipeline[str | Path, tuple[npt.NDArray[np.uint8], sv.Detections]]:
    """Detect, answer a task per crop, annotate first-line labels, and display."""
    if not class_ids:
        raise ValueError("At least one class ID is required.")
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1.")
    prepared_prompt = build_prompt(prompt)
    pad_boxes = PadBoxes(padding=padding)
    return Pipeline(
        [
            LoadFile(),
            Decode(),
            ImageToArray(),
            Store("source_image"),
            RoboflowInference(model_id=detection_model_id, confidence=confidence),
            Select(0),
            Detections.FromInference(),
            Detections.Filter(lambda detections: np.isin(detections.class_id, class_ids)),
            Detections.Filter(lambda detections: detections.confidence >= confidence),
            Detections.NMS(threshold=0.5),
            pad_boxes,
            Recall("source_image", prepend=True),
            ClipBoxes(),
            CropDetections(),
            Store("detections", source=0),
            Pick(1),
            Scatter(max_concurrency=1),
            Resize(
                target_size=(1024, 1024),
                mode="letterbox",
                allow_scale_up=False,
                interpolation="area",
            ),
            Pick(0),  # Discard the crop resize transform; source boxes stay unchanged.
            ImageToArray(),
            Qwen3VL(prompt=prepared_prompt, model_id=qwen_model_id, region_name=region_name),
            Gather(),
            Recall("detections", prepend=True),
            AttachResponses(),
            Recall("source_image", prepend=True),
            BoxAnnotator(),
            LabelAnnotator(
                label_formatter=lambda detection: detection.data["response"].splitlines()[0],
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
    parser.add_argument("--output", type=Path, help="Optional JSON result path.")
    parser.add_argument("--output-image", type=Path, help="Optional annotated image path.")
    parser.add_argument("--detection-model-id", default=DEFAULT_DETECTION_MODEL_ID)
    parser.add_argument(
        "--class-ids", type=int, nargs="+", default=DEFAULT_CLASS_IDS,
        help="Detector class IDs to keep (default: 1, person for RF-DETR).",
    )
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument(
        "--padding", type=float, default=0.05,
        help="Extra context per side, as a fraction of box width/height (default: 0.05).",
    )
    parser.add_argument("--qwen-model-id", default=DEFAULT_QWEN_MODEL_ID)
    parser.add_argument(
        "--prompt", default=DEFAULT_PROMPT,
        help="Task to answer for each crop; answer-format instructions are appended.",
    )
    parser.add_argument(
        "--region",
        default=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1",
        help="Bedrock region (default: AWS region environment variable, or us-east-1).",
    )
    args = parser.parse_args()
    input_path = args.input or Path(download_assets(DEFAULT_IMAGE_ASSET, directory=ASSETS_DIR))

    pipeline = build_pipeline(
        detection_model_id=args.detection_model_id,
        class_ids=tuple(args.class_ids),
        confidence=args.confidence,
        padding=args.padding,
        prompt=args.prompt,
        qwen_model_id=args.qwen_model_id,
        region_name=args.region,
    )
    pipeline.validate()
    annotated_image, detections = pipeline(input_path)
    results = [
        {
            "detection_index": index,
            "xyxy": detections.xyxy[index].tolist(),
            "class_id": int(detections.class_id[index]),
            "confidence": float(detections.confidence[index]),
            "response": response,
        }
        for index, response in enumerate(detections.data["response"])
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
