"""Track the continuous time each object spends in a polygon zone.

Run from the repo root:
    python examples/run_time_in_zone.py
    python examples/run_time_in_zone.py --input path/to/video.mp4
    python examples/run_time_in_zone.py --tracker botsort
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import supervision as sv
from supervision.assets import VideoAssets, download_assets

from common import ASSETS_DIR
from ml_pipes.supervision.inference import RoboflowInference
from ml_pipes.supervision.trackers import ByteTrack
from ml_pipes.supervision import (
    BoxAnnotator,
    Detections,
    ImageWindow,
    LabelAnnotator,
    PolygonZoneAnnotator,
    TrackingTimer,
    TraceAnnotator,
    TriggerZone,
)
from ml_pipes.core import Pipeline
from ml_pipes.standard import Recall, Select, Store

DEFAULT_MODEL_ID = "rfdetr-small"
DEFAULT_VIDEO_ASSET = VideoAssets.PEOPLE_WALKING
ZONE_FRACTIONS = ((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8))


def build_zone(width: int, height: int) -> sv.PolygonZone:
    """Create a central zone that scales with the input video resolution."""
    polygon = np.asarray(
        [[x * width, y * height] for x, y in ZONE_FRACTIONS], dtype=np.int64
    )
    return sv.PolygonZone(polygon=polygon)


def build_frame_pipeline(
    model_id: str,
    api_key: str | None,
    zone: sv.PolygonZone,
    fps: float,
) -> Pipeline[npt.NDArray[np.uint8], tuple[npt.NDArray[np.uint8], sv.Detections]]:
    return Pipeline(
        [
            Store("source_frame"),
            RoboflowInference(model_id=model_id, api_key=api_key),
            Select(0),
            Detections.FromInference(),
            ByteTrack(),
            Detections.Filter(lambda detections: detections.tracker_id != -1),
            TriggerZone(zone),
            TrackingTimer(fps, field="time_in_zone"),
            Recall("source_frame", prepend=True),
            TraceAnnotator(),
            BoxAnnotator(),
            LabelAnnotator(
                label_formatter=lambda detection: (
                    f"#{int(detection.tracker_id) if detection.tracker_id is not None else -1} "
                    f"{int(float(detection.data['time_in_zone'])) // 60:02d}:"
                    f"{int(float(detection.data['time_in_zone'])) % 60:02d}"
                )
            ),
            PolygonZoneAnnotator(zone=zone),
            ImageWindow("Time In Zone", at=0),
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

    video_info = sv.VideoInfo.from_video_path(str(input_path))
    zone = build_zone(video_info.width, video_info.height)
    pipeline = build_frame_pipeline(
        model_id=args.model_id,
        api_key=args.api_key,
        zone=zone,
        fps=video_info.fps,
    )
    pipeline.validate()
    pipeline.describe()

    frames_generator = sv.get_video_frames_generator(str(input_path))
    for frame in frames_generator:
        pipeline(frame)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
