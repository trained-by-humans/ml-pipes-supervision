"""
Small-object detection through tiled Roboflow Inference and Supervision.

Run from the repo root:
    python examples/run_detect_small_objects.py
    python examples/run_detect_small_objects.py --input path/to/photo.jpg
    python examples/run_detect_small_objects.py --model-id rfdetr-small --slice-wh 320 320 --overlap-wh 80 80
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
import supervision as sv
from supervision.assets import ImageAssets, download_assets

from common import ASSETS_DIR
from ml_pipes.supervision.inference import RoboflowInference
from ml_pipes.supervision import (
    BoxAnnotator,
    Detections,
    ImageToArray,
    LabelAnnotator,
    PlotImage,
)
from ml_pipes.core import Pipeline
from ml_pipes.standard import Gather, Pick, Recall, Scatter, Select, Store
from ml_pipes.vision import Decode, LoadFile, Tile

DEFAULT_MODEL_ID = "rfdetr-medium"
DEFAULT_SLICE_WH = (320, 320)
DEFAULT_OVERLAP_WH = (80, 80)
DEFAULT_MAX_CONCURRENCY = 4
DEFAULT_IOU_THRESHOLD = 0.5


def build_pipeline(
    model_id: str,
    api_key: str | None,
    slice_wh: tuple[int, int],
    overlap_wh: tuple[int, int],
    max_concurrency: int,
    iou_threshold: float,
) -> Pipeline[str | Path, tuple[npt.NDArray[np.uint8], sv.Detections]]:
    return Pipeline(
        [
            LoadFile(),
            Decode(),
            Store("source_image"),
            Tile(slice_wh=slice_wh, overlap_wh=overlap_wh),
            Store("tile_rects", source=1),
            Pick(0),
            Scatter(max_concurrency=max_concurrency),
            RoboflowInference(model_id=model_id, api_key=api_key),
            Select(0),
            Detections.FromInference(),
            Gather(),
            Recall("tile_rects"),
            Detections.Stitch(),
            Detections.NMM(iou_threshold=iou_threshold),
            Store("detections"),
            Recall("source_image"),
            Pick(1),
            ImageToArray(),
            Recall("detections"),
            BoxAnnotator(),
            LabelAnnotator(show_class=True, show_confidence=True),
            PlotImage(at=0),
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
        help="Input image path. Defaults to Supervision's ImageAssets.PEOPLE_WALKING asset.",
    )
    parser.add_argument(
        "--slice-wh",
        type=int,
        nargs=2,
        default=list(DEFAULT_SLICE_WH),
        metavar=("W", "H"),
        help="Tile width and height in pixels. Defaults to 320 320.",
    )
    parser.add_argument(
        "--overlap-wh",
        type=int,
        nargs=2,
        default=list(DEFAULT_OVERLAP_WH),
        metavar=("W", "H"),
        help="Overlap between tiles in pixels. Defaults to 80 80.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help="Maximum parallel tile inference workers. Defaults to 4.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help="IoU threshold for post-stitch NMM merge. Defaults to 0.5.",
    )
    args = parser.parse_args()

    image_path = args.input or Path(
        download_assets(ImageAssets.PEOPLE_WALKING, directory=ASSETS_DIR)
    )

    pipeline = build_pipeline(
        model_id=args.model_id,
        api_key=args.api_key,
        slice_wh=(args.slice_wh[0], args.slice_wh[1]),
        overlap_wh=(args.overlap_wh[0], args.overlap_wh[1]),
        max_concurrency=args.max_concurrency,
        iou_threshold=args.iou_threshold,
    )
    pipeline.validate()
    pipeline.describe()
    pipeline(image_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
