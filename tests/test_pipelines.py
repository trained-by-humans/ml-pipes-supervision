"""Real-library smoke tests across multiple ml-pipes operators."""

from copy import deepcopy
from typing import Any

import numpy as np
import pytest
import supervision as sv

from ml_pipes.core import Pipeline
from ml_pipes.supervision import Detections, MaskAnnotator
from ml_pipes.tensor import TensorRegistry


@pytest.fixture
def inference_result() -> dict[str, Any]:
    return {
        "image": {"width": 8, "height": 8},
        "predictions": [
            {
                "x": 4,
                "y": 4,
                "width": 4,
                "height": 4,
                "confidence": 0.9,
                "class_id": 3,
                "class": "vehicle",
                "points": [
                    {"x": 2, "y": 2},
                    {"x": 5, "y": 2},
                    {"x": 5, "y": 5},
                    {"x": 2, "y": 5},
                ],
            }
        ],
    }


@pytest.mark.parametrize("compact_masks", [False, True], ids=["dense", "compact"])
def test_inference_masks_flow_through_filter_and_annotation_pipeline(
    inference_result: dict[str, Any], compact_masks: bool,
) -> None:
    source = deepcopy(inference_result)
    expected = sv.Detections.from_inference(
        inference_result, compact_masks=compact_masks
    )
    pipeline = Pipeline(
        [
            Detections.FromInference(compact_masks=compact_masks),
            Detections.Filter(lambda detections: detections.class_id == 3),
        ],
        auto_validate=True,
    )

    detections = pipeline(inference_result)
    scene = np.zeros((8, 8, 3), dtype=np.uint8)
    annotated, returned = MaskAnnotator()(scene, detections)

    assert detections == expected
    assert isinstance(detections.mask, sv.CompactMask if compact_masks else np.ndarray)
    assert returned is detections
    assert annotated.shape == scene.shape
    assert annotated.dtype == scene.dtype
    assert annotated is not scene
    assert np.any(annotated)
    assert not np.any(scene)
    assert inference_result == source


def test_tensor_conversion_and_unannotated_filter_validate_as_a_pipeline() -> None:
    pipeline = Pipeline(
        [
            Detections.FromTensorRegistry(),
            Detections.Filter(lambda detections: detections.class_id == 1),
        ],
        auto_validate=True,
    )
    registry = TensorRegistry(
        {
            "boxes": np.asarray([[0, 0, 10, 10], [1, 1, 2, 2]], dtype=np.float32),
            "scores": np.asarray([0.9, 0.8], dtype=np.float32),
            "classes": np.asarray([1, 2], dtype=np.int32),
        }
    )

    result = pipeline(registry)

    assert result.class_id.tolist() == [1]
