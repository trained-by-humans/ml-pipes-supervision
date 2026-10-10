"""
RF-DETR detection filtering through built-in ml-pipes prediction filters and Supervision.

Run from the repo root:
    python examples/run_filter_detections.py
    python examples/run_filter_detections.py --input path/to/photo.jpg
    python examples/run_filter_detections.py --model-id rfdetr-medium
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
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
from ml_pipes.standard import Recall, Select, Store
from ml_pipes.vision import Decode, LoadFile

DEFAULT_MODEL_ID = "rfdetr-small"
# Pretrained RF-DETR aliases use COCO category ID 1 for person.
DEFAULT_CLASS_ID = 1
DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_MAX_RELATIVE_AREA = 0.8


def build_pipeline(
    model_id: str,
    api_key: str | None,
    image_shape: tuple[int, int],
) -> Pipeline[str | Path, tuple[npt.NDArray[np.uint8], sv.Detections]]:
    return Pipeline(
        [
            LoadFile(),
            Decode(),
            ImageToArray(),
            Store("source_image"),
            RoboflowInference(model_id=model_id, api_key=api_key),
            Select(0),
            Detections.FromInference(),
            Detections.Filter(lambda detections: detections.class_id == DEFAULT_CLASS_ID),
            Detections.Filter(lambda detections: detections.confidence > DEFAULT_MIN_CONFIDENCE),
            Detections.Filter(
                lambda detections: (
                    detections.area
                    < DEFAULT_MAX_RELATIVE_AREA * image_shape[0] * image_shape[1]
                )
            ),
            Recall("source_image", prepend=True),
            BoxAnnotator(),
            LabelAnnotator(),
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
        help="Input image path. Defaults to Supervision's PEOPLE_WALKING image asset.",
    )
    args = parser.parse_args()

    image_path = args.input or Path(
        download_assets(ImageAssets.PEOPLE_WALKING, directory=ASSETS_DIR)
    )

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image dimensions from {image_path}")
    image_shape = int(image.shape[0]), int(image.shape[1])

    pipeline = build_pipeline(args.model_id, args.api_key, image_shape)
    pipeline.validate()
    pipeline.describe()
    pipeline(image_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
