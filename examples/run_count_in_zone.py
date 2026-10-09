"""
RF-DETR count-in-zone video processing through Roboflow Inference and Supervision.

Run from the repo root:
    python examples/run_count_in_zone.py
    python examples/run_count_in_zone.py --input path/to/video.mp4
    python examples/run_count_in_zone.py --output result.mp4
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
    BoxAnnotator,
    ImageWindow,
    PolygonZoneAnnotator,
    Detections,
    TriggerZone,
)
from ml_pipes.core import Pipeline
from ml_pipes.standard import Pick, Recall, Select, Store

DEFAULT_MODEL_ID = "rfdetr-medium"
DEFAULT_VIDEO_ASSET = VideoAssets.VEHICLES_2
DEFAULT_OUTPUT_NAME = "result.mp4"
DEFAULT_ZONE_POLYGONS = (
    np.asarray(
        [
            [718, 595],
            [927, 592],
            [851, 1062],
            [42, 1059],
        ],
        dtype=np.int32,
    ),
    np.asarray(
        [
            [987, 595],
            [1199, 595],
            [1893, 1056],
            [1015, 1062],
        ],
        dtype=np.int32,
    ),
)


def build_zones() -> tuple[sv.PolygonZone, sv.PolygonZone]:
    return (
        sv.PolygonZone(polygon=DEFAULT_ZONE_POLYGONS[0]),
        sv.PolygonZone(polygon=DEFAULT_ZONE_POLYGONS[1]),
    )


def build_frame_pipeline(
    model_id: str,
    api_key: str | None,
    zone_one: sv.PolygonZone,
    zone_two: sv.PolygonZone,
) -> Pipeline[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    return Pipeline(
        [
            Store("source_frame"),
            RoboflowInference(model_id=model_id, api_key=api_key),
            Select(0),
            Detections.FromInference(),
            Store("detections"),
            TriggerZone(zone_one),
            Recall("source_frame", prepend=True),
            BoxAnnotator(color=sv.ColorPalette.DEFAULT.by_idx(0), thickness=4),
            PolygonZoneAnnotator(
                zone=zone_one,
                color=sv.ColorPalette.DEFAULT.by_idx(0),
                thickness=4,
                text_color=sv.Color.WHITE,
                text_scale=4,
                text_thickness=8,
            ),
            Pick(0),
            Store("first_zone_scene"),
            Recall("detections", prepend=True),
            Select(0),
            TriggerZone(zone_two),
            Recall("first_zone_scene", prepend=True),
            BoxAnnotator(color=sv.ColorPalette.DEFAULT.by_idx(1), thickness=4),
            PolygonZoneAnnotator(
                zone=zone_two,
                color=sv.ColorPalette.DEFAULT.by_idx(1),
                thickness=4,
                text_color=sv.Color.WHITE,
                text_scale=4,
                text_thickness=8,
            ),
            ImageWindow("Count In Zone", at=0),
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
        help="Input video path. Defaults to Supervision's VideoAssets.VEHICLES_2 asset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path. Defaults to result.mp4 under examples/.example_assets.",
    )
    args = parser.parse_args()

    input_path = args.input or Path(
        download_assets(DEFAULT_VIDEO_ASSET, directory=ASSETS_DIR)
    )
    output_path = args.output or ASSETS_DIR / DEFAULT_OUTPUT_NAME

    zone_one, zone_two = build_zones()
    pipeline = build_frame_pipeline(
        model_id=args.model_id,
        api_key=args.api_key,
        zone_one=zone_one,
        zone_two=zone_two,
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
    print(f"Saved zone-count video to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
