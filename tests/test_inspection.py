"""Registration of Supervision's image convention with ml-pipes inspection."""

import numpy as np

from ml_pipes import supervision
from ml_pipes.inspection import ImageBlock, PipelineInspector


def test_raw_ndarray_inspection_uses_supervision_bgr_convention() -> None:
    # Importing our integration registers the BGR ndarray formatter.
    assert supervision.ImageToArray
    bgr = np.array([[[0, 0, 255]]], dtype=np.uint8)

    blocks = PipelineInspector()._value_to_blocks(bgr)

    assert isinstance(blocks[0], ImageBlock)
    assert blocks[0].title == "ndarray  1×1  BGR"
    np.testing.assert_array_equal(blocks[0].array, np.array([[[255, 0, 0]]], dtype=np.uint8))
