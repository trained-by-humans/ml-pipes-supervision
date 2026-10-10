"""
YOLO-World zero-shot video detection through Roboflow Inference and Supervision.

Port of Supervision's "Zero-Shot Object Detection with YOLO-World" video
workflow. Yolo-World requires the optional Roboflow Inference extra:

    python -m pip install 'inference[yolo-world]'

Run from the repo root:
    python examples/run_zero_shot_object_detection.py
    python examples/run_zero_shot_object_detection.py --text "blue bottle"
    python examples/run_zero_shot_object_detection.py --input path/to/video.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import cast

import numpy as np
import numpy.typing as npt
import supervision as sv
from inference.core.models.base import Model

from common import ASSETS_DIR, resolve_input_path
from ml_pipes.supervision.inference import RoboflowInference
from ml_pipes.supervision import (
    BoxAnnotator,
    Detections,
    ImageWindow,
    LabelAnnotator,
)
from ml_pipes.core import Pipeline
from ml_pipes.standard import Pick, Recall, Store

DEFAULT_MODEL_ID = "yolo_world/l"
DEFAULT_TEXT = "yellow filling"
DEFAULT_CONFIDENCE = 0.002
DEFAULT_NMS_THRESHOLD = 0.1
DEFAULT_MAX_RELATIVE_AREA = 0.10
DEFAULT_VIDEO_NAME = "yellow-filling.mp4"
DEFAULT_VIDEO_URL = "https://media.roboflow.com/supervision/cookbooks/yellow-filling.mp4"
DEFAULT_OUTPUT_NAME = "yellow-filling-output.mp4"


def _build_yolo_world(model_id: str, api_key: str | None) -> Model:
    try:
        from inference.models.yolo_world.yolo_world import YOLOWorld
    except ImportError:
        print(
            "YOLO-World requires the optional Roboflow Inference dependency. "
            "Install it with: python -m pip install 'inference[yolo-world]'",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    return cast(Model, YOLOWorld(model_id=model_id, api_key=api_key))


def build_frame_pipeline(
    model: Model,
    text: str,
    image_shape: tuple[int, int],
) -> Pipeline[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    return Pipeline(
        [
            Store("source_frame"),
            RoboflowInference(
                model,
                text=[text],
                confidence=DEFAULT_CONFIDENCE,
            ),
            Detections.FromInference(),
            Detections.NMS(threshold=DEFAULT_NMS_THRESHOLD),
            Detections.Filter(
                lambda detections: (
                    detections.area
                    < DEFAULT_MAX_RELATIVE_AREA * image_shape[0] * image_shape[1]
                )
            ),
            Recall("source_frame", prepend=True),
            BoxAnnotator(thickness=2),
            LabelAnnotator(
                text_thickness=2,
                text_scale=1,
                text_color=sv.Color.BLACK,
            ),
            ImageWindow("YOLO-World Zero-Shot Detection", at=0),
            Pick(0),
        ],
        auto_validate=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Roboflow Inference Yolo-World model id. Defaults to yolo_world/l.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Roboflow API key. Only needed for private or account-scoped model ids.",
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Zero-shot class prompt. Defaults to 'yellow filling'.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input video path. Defaults to Supervision's yellow-filling video.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path. Defaults to yellow-filling-output.mp4.",
    )
    args = parser.parse_args()

    model = _build_yolo_world(args.model_id, args.api_key)
    input_path = resolve_input_path(
        args.input,
        ASSETS_DIR / DEFAULT_VIDEO_NAME,
        DEFAULT_VIDEO_URL,
    )
    output_path = args.output or ASSETS_DIR / DEFAULT_OUTPUT_NAME
    frame_width, frame_height = sv.VideoInfo.from_video_path(str(input_path)).resolution_wh
    pipeline = build_frame_pipeline(model, args.text, (frame_height, frame_width))
    pipeline.validate()
    pipeline.describe()

    def process_frame(frame: npt.NDArray[np.uint8], _: int) -> npt.NDArray[np.uint8]:
        return pipeline(frame)

    sv.process_video(
        source_path=str(input_path),
        target_path=str(output_path),
        callback=process_frame,
    )
    print(f"Saved zero-shot detection video to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
