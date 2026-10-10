"""
Count objects crossing a line with Roboflow Inference and Supervision.

Port of Supervision's "Count Objects Crossing the Line" notebook process-video
section. The upstream RF-DETR model is run through
RoboflowInference(model_id="rfdetr-medium").

Run from the repo root:
    python examples/run_count_objects_crossing_line.py
    python examples/run_count_objects_crossing_line.py --input path/to/video.mp4
    python examples/run_count_objects_crossing_line.py --output result.mp4
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
from ml_pipes.supervision.trackers import ByteTrack
from ml_pipes.supervision import (
    BoxAnnotator,
    FPSAnnotator,
    Detections,
    ImageWindow,
    LabelAnnotator,
    LineZoneAnnotator,
    TraceAnnotator,
    TriggerLineZone,
)
from ml_pipes.core import Pipeline
from ml_pipes.standard import Pick, Recall, Select, Store

DEFAULT_MODEL_ID = "rfdetr-medium"
DEFAULT_VIDEO_ASSET = VideoAssets.VEHICLES
DEFAULT_OUTPUT_NAME = "count-objects-crossing-the-line-result.mp4"
LINE_START = sv.Point(0, 1500)
LINE_END = sv.Point(3840, 1500)


def build_frame_pipeline(
    model_id: str,
    api_key: str | None,
    line_zone: sv.LineZone,
) -> Pipeline[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    return Pipeline(
        [
            Store("source_frame"),
            RoboflowInference(model_id=model_id, api_key=api_key),
            Select(0),
            Detections.FromInference(),
            ByteTrack(),
            Detections.Filter(lambda detections: detections.tracker_id != -1),
            TriggerLineZone(line_zone),
            Recall("source_frame", prepend=True),
            TraceAnnotator(thickness=4),
            BoxAnnotator(thickness=4),
            LabelAnnotator(
                text_thickness=4,
                text_scale=2,
                show_tracker_id=True,
                show_class=True,
                show_confidence=True,
            ),
            LineZoneAnnotator(
                line_zone=line_zone,
                thickness=4,
                text_thickness=4,
                text_scale=2,
            ),
            FPSAnnotator(),
            ImageWindow("Count Objects Crossing the Line", at=0),
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
        help="Roboflow Inference model id. Defaults to the RF-DETR medium pretrained alias.",
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
        help="Input video path. Defaults to Supervision's VideoAssets.VEHICLES asset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path. Defaults to count-objects-crossing-the-line-result.mp4.",
    )
    args = parser.parse_args()

    input_path = args.input or Path(
        download_assets(DEFAULT_VIDEO_ASSET, directory=ASSETS_DIR)
    )
    output_path = args.output or ASSETS_DIR / DEFAULT_OUTPUT_NAME
    line_zone = sv.LineZone(start=LINE_START, end=LINE_END)
    pipeline = build_frame_pipeline(
        model_id=args.model_id,
        api_key=args.api_key,
        line_zone=line_zone,
    )
    pipeline.validate()
    pipeline.describe()

    def process_frame(frame: npt.NDArray[np.uint8], _: int) -> npt.NDArray[np.uint8]:
        return pipeline(frame)

    sv.process_video(
        source_path=str(input_path),
        target_path=str(output_path),
        callback=process_frame,
    )
    print(f"Saved line-crossing video to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
