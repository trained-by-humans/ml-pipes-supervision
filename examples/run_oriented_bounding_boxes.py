"""
Oriented bounding-box detection through Ultralytics and Supervision.

Port of Supervision's "Oriented Bounding Boxes" notebook. This example uses
YOLO11-OBB, trained on DOTA aerial classes, to detect ships in the marina image.
It requires ml-pipes-ultralytics:

    python -m pip install ml-pipes-ultralytics

Run from the repo root:
    python examples/run_oriented_bounding_boxes.py
    python examples/run_oriented_bounding_boxes.py --input path/to/photo.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
import supervision as sv

from common import ASSETS_DIR, resolve_input_path
from ml_pipes.core import Pipeline
from ml_pipes.standard import Recall, Select, Store
from ml_pipes.supervision import (
    Detections,
    ImageToArray,
    OrientedBoxAnnotator,
    PlotImage,
)
from ml_pipes.ultralytics import yolo
from ml_pipes.vision import Decode, LoadFile

DEFAULT_MODEL_ID = "yolo11n-obb.pt"
DEFAULT_IMAGE_NAME = "boats.jpg"
DEFAULT_IMAGE_URL = "https://ultralytics.com/images/boats.jpg"
DEFAULT_IMAGE_SIZE = 1024
DEFAULT_CLASS_NAME = "ship"


def build_pipeline(
    model_id: str,
    image_size: int,
) -> Pipeline[str | Path, tuple[npt.NDArray[np.uint8], sv.Detections]]:
    """Build the final OBB pipeline with OBB-aware NMS and annotation."""
    return Pipeline(
        [
            LoadFile(),
            Decode(),
            ImageToArray(),
            Store("source_image"),
            yolo.Predict(model=model_id, imgsz=image_size, iou=0.9),
            Select(0),
            Detections.FromUltralytics(),
            Detections.Filter(
                lambda detections: detections.data["class_name"] == DEFAULT_CLASS_NAME
            ),
            Detections.NMS(threshold=0.3),
            Recall("source_image", prepend=True),
            OrientedBoxAnnotator(color=sv.Color.GREEN, thickness=2),
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
        help="Ultralytics OBB model id. Defaults to yolo11n-obb.pt.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input image path. Defaults to the notebook's boats.jpg source.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Ultralytics inference image size. Defaults to 1024.",
    )
    args = parser.parse_args()

    pipeline = build_pipeline(args.model_id, args.imgsz)
    input_path = resolve_input_path(
        args.input,
        ASSETS_DIR / DEFAULT_IMAGE_NAME,
        DEFAULT_IMAGE_URL,
    )
    pipeline.validate()
    pipeline.describe()
    pipeline(input_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
