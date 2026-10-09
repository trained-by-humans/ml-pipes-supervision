"""
RF-DETR video detection saved to CSV through Roboflow Inference and Supervision.

Run from the repo root:
    python examples/run_save_detections.py
    python examples/run_save_detections.py --input path/to/video.mp4
    python examples/run_save_detections.py --output detections.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import numpy.typing as npt
import supervision as sv
from supervision.assets import VideoAssets, download_assets

from common import ASSETS_DIR
from ml_pipes.supervision.inference import RoboflowInference
from ml_pipes.supervision import (
    Detections,
)

from ml_pipes.core import Pipeline
from ml_pipes.standard import Select

DEFAULT_MODEL_ID = "rfdetr-small"
DEFAULT_VIDEO_ASSET = VideoAssets.PEOPLE_WALKING


def build_frame_pipeline(
    model_id: str,
    api_key: str | None,
) -> Pipeline[npt.NDArray[np.uint8], sv.Detections]:
    return Pipeline(
        [
            RoboflowInference(model_id=model_id, api_key=api_key),
            Select(0),
            Detections.FromInference(),
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
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <input>_detections.csv.",
    )
    args = parser.parse_args()

    input_path = args.input or Path(
        download_assets(DEFAULT_VIDEO_ASSET, directory=ASSETS_DIR)
    )
    output_path = args.output or input_path.with_stem(input_path.stem + "_detections").with_suffix(".csv")

    pipeline = build_frame_pipeline(args.model_id, args.api_key)
    pipeline.validate()
    pipeline.describe()

    frames_generator = sv.get_video_frames_generator(str(input_path))
    with sv.CSVSink(str(output_path)) as sink:
        for frame in frames_generator:
            detections = pipeline(frame)
            sink.append(detections, {})

    print(f"Saved detections to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
