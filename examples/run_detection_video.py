"""
RF-DETR video detection through Roboflow Inference and Supervision.

Run from the repo root:
    python examples/run_detection_video.py
    python examples/run_detection_video.py --input path/to/video.mp4
    python examples/run_detection_video.py --model-id rfdetr-medium
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
import supervision as sv
from supervision.assets import VideoAssets, download_assets

from common import ASSETS_DIR
from ml_pipes.supervision.inference import RoboflowInference
from ml_pipes.supervision import (
    BoxAnnotator,
    FPSAnnotator,
    Detections,
    ImageWindow,
    LabelAnnotator,
)

from ml_pipes.core import Pipeline
from ml_pipes.standard import Recall, Select, Store

DEFAULT_MODEL_ID = "rfdetr-small"
DEFAULT_VIDEO_ASSET = VideoAssets.PEOPLE_WALKING


def build_frame_pipeline(
    model_id: str,
    api_key: str | None,
) -> Pipeline[npt.NDArray[np.uint8], tuple[npt.NDArray[np.uint8], sv.Detections]]:
    return Pipeline(
        [
            Store("source_frame"),
            RoboflowInference(model_id=model_id, api_key=api_key),
            Select(0),
            Detections.FromInference(),
            Recall("source_frame", prepend=True),
            BoxAnnotator(),
            LabelAnnotator(),
            FPSAnnotator(),
            ImageWindow("ml-pipes", at=0),
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
        help="Roboflow Inference model id. Defaults to the RF-DETR small pretrained alias.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Roboflow API key. Only needed for private or account-scoped model ids.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input video path. Defaults to Supervision's VideoAssets.PEOPLE_WALKING asset.",
    )
    args = parser.parse_args()

    input_path = args.input or Path(
        download_assets(DEFAULT_VIDEO_ASSET, directory=ASSETS_DIR)
    )

    pipeline = build_frame_pipeline(args.model_id, args.api_key)
    pipeline.validate()
    pipeline.describe()

    frames_generator = sv.get_video_frames_generator(str(input_path))
    for frame in frames_generator:
        pipeline(frame)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
