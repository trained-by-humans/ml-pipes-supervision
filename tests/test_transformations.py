"""Detection geometry, raster crop policies, and pipeline integration."""
from copy import deepcopy
from unittest.mock import create_autospec

import numpy as np
import pytest
import supervision as sv

from ml_pipes.core import Pipeline
from ml_pipes.supervision import Detections
from ml_pipes.vision import ImagePayload


@pytest.fixture
def detections():
    return sv.Detections(
        xyxy=np.asarray([[1, 2, 5, 6], [10, 8, 14, 12]], dtype=np.float32),
        mask=np.ones((2, 16, 20), dtype=bool),
        confidence=np.asarray([0.9, 0.8]),
        class_id=np.asarray([2, 3]),
        tracker_id=np.asarray([11, 12]),
        data={
            "name": np.asarray(["first", "second"]),
            "xyxyxyxy": np.arange(16).reshape(2, 4, 2),
            "keypoints": np.arange(12).reshape(2, 3, 2),
        },
        metadata={"source": "test"},
    )


@pytest.fixture
def image():
    return np.arange(16 * 20 * 3, dtype=np.uint8).reshape(16, 20, 3)


def assert_other_fields_equal(actual, expected):
    for field in ("mask", "confidence", "class_id", "tracker_id"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    assert actual.data.keys() == expected.data.keys()
    for key in actual.data:
        np.testing.assert_array_equal(actual.data[key], expected.data[key])
    assert actual.metadata == expected.metadata


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint32, np.float32, np.float64])
@pytest.mark.parametrize("operator, helper, kwargs", [
    (Detections.PadBoxes(3), "pad_boxes", {"px": 3, "py": None}),
    (Detections.PadBoxes(3, 5), "pad_boxes", {"px": 3, "py": 5}),
    (Detections.PadBoxes(-1, -1), "pad_boxes", {"px": -1, "py": -1}),
    (Detections.ScaleBoxes(1.5), "scale_boxes", {"factor": 1.5}),
    (Detections.ScaleBoxes(0.5), "scale_boxes", {"factor": 0.5}),
    (Detections.MoveBoxes((-3, 4)), "move_boxes", {"offset": (-3, 4)}),
])
def test_geometry_matches_source_and_preserves_other_fields(detections, dtype, operator, helper, kwargs):
    detections.xyxy = detections.xyxy.astype(dtype)
    original = deepcopy(detections)

    result = operator(detections)
    expected = getattr(sv, helper)(original.xyxy, **kwargs)

    assert result is not detections
    np.testing.assert_array_equal(result.xyxy, expected)
    assert result.xyxy.dtype == expected.dtype
    assert not np.shares_memory(result.xyxy, detections.xyxy)
    assert_other_fields_equal(result, original)
    assert detections == original


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
@pytest.mark.parametrize("factors", [(2, 0.5), (0.5, 2), (1, 3)])
def test_scale_axes_are_independent_and_centered(detections, dtype, factors):
    detections.xyxy = detections.xyxy.astype(dtype)
    original = deepcopy(detections)

    result = Detections.ScaleBoxes(*factors)(detections)

    horizontal = sv.scale_boxes(original.xyxy, float(factors[0]))
    vertical = sv.scale_boxes(original.xyxy, float(factors[1]))
    np.testing.assert_array_equal(result.xyxy[:, [0, 2]], horizontal[:, [0, 2]])
    np.testing.assert_array_equal(result.xyxy[:, [1, 3]], vertical[:, [1, 3]])
    np.testing.assert_allclose(result.xyxy[:, :2] + result.xyxy[:, 2:],
                               original.xyxy[:, :2] + original.xyxy[:, 2:])
    assert_other_fields_equal(result, original)
    assert detections == original


def test_scale_large_integer_axes_keep_source_precision():
    boxes = np.asarray([[2**60 + 3, 2**59 + 7, 2**60 + 301, 2**59 + 607]], dtype=np.int64)
    result = Detections.ScaleBoxes(1.25, 0.75)(sv.Detections(xyxy=boxes))
    np.testing.assert_array_equal(result.xyxy[:, [0, 2]], sv.scale_boxes(boxes, 1.25)[:, [0, 2]])
    np.testing.assert_array_equal(result.xyxy[:, [1, 3]], sv.scale_boxes(boxes, 0.75)[:, [1, 3]])


@pytest.mark.parametrize("factors", [(0,), (-1,), (np.nan,), (np.inf,), (1, 0), (1, -1), (1, np.nan)])
def test_scale_rejects_nonfinite_or_nonpositive_factors(factors):
    with pytest.raises(ValueError, match="finite and positive"):
        Detections.ScaleBoxes(*factors)


@pytest.mark.parametrize("name, kwargs", [
    ("pad_boxes", {"px": 3, "py": 5}),
    ("scale_boxes", {"factor": 2.0}),
    ("move_boxes", {"offset": (-3, 4)}),
])
def test_geometry_delegates_to_upstream(detections, monkeypatch, name, kwargs):
    expected = detections.xyxy.copy()
    helper = create_autospec(getattr(sv, name), return_value=expected)
    monkeypatch.setattr(sv, name, helper)
    operator = {
        "pad_boxes": Detections.PadBoxes(3, 5),
        "scale_boxes": Detections.ScaleBoxes(2),
        "move_boxes": Detections.MoveBoxes((-3, 4)),
    }[name]

    assert operator(detections).xyxy is expected
    assert helper.call_args.args[0] is detections.xyxy
    assert helper.call_args.kwargs == kwargs


@pytest.mark.parametrize("rounding, expected", [
    ("none", [[1.5, 2.5, 6.5, 7.5]]),
    ("nearest", [[2, 2, 6, 8]]),
    ("outward", [[1, 2, 7, 8]]),
    ("inward", [[2, 3, 6, 7]]),
])
def test_clip_rounding_modes(image, rounding, expected):
    detections = sv.Detections(xyxy=np.asarray([[1.5, 2.5, 6.5, 7.5]], dtype=np.float32))
    original = deepcopy(detections)

    output_image, result = Detections.ClipBoxes(rounding=rounding)(image, detections)

    assert output_image is image
    np.testing.assert_array_equal(result.xyxy, expected)
    assert result.xyxy.dtype == np.float32
    assert detections == original


def invalid_detections():
    boxes = [
        [-2, -3, 5, 6], [25, 1, 30, 8], [2, 2, 2, 6],
        [1.8, 1, 1.2, 5], [np.nan, 1, 4, 5], [-np.inf, 1, 4, 5],
        [1.1, 2.1, 1.4, 2.4], [10, 8, 14, 12],
    ]
    return sv.Detections(
        xyxy=np.asarray(boxes, dtype=np.float64),
        mask=np.ones((len(boxes), 16, 20), dtype=bool),
        confidence=np.arange(len(boxes), dtype=float),
        class_id=np.arange(len(boxes)),
        tracker_id=np.arange(len(boxes)) + 10,
        data={"label": np.asarray([f"row-{i}" for i in range(len(boxes))]),
              "source_xyxy": np.asarray(boxes, dtype=np.float64)},
        metadata={"source": "invalids"},
    )


def test_clip_defaults_match_upstream_including_invalid_rows(image):
    detections = invalid_detections()
    original = detections.xyxy.copy()
    _, result = Detections.ClipBoxes()(image, detections)

    np.testing.assert_array_equal(result.xyxy, sv.clip_boxes(original, (20, 16)))
    assert len(result) == len(detections)
    assert_other_fields_equal(result, detections)
    np.testing.assert_array_equal(detections.xyxy, original)


@pytest.mark.parametrize("rounding, keep", [
    ("none", [0, 6, 7]), ("nearest", [0, 7]),
    ("outward", [0, 6, 7]), ("inward", [0, 7]),
])
def test_clip_drop_keeps_every_field_aligned_and_rejects_invalid_originals(image, rounding, keep):
    detections = invalid_detections()
    original = detections.xyxy.copy()

    _, result = Detections.ClipBoxes(rounding=rounding, invalid="drop")(image, detections)

    assert_other_fields_equal(result, detections[keep])
    assert result.class_id.tolist() == keep
    np.testing.assert_array_equal(detections.xyxy, original)


@pytest.mark.parametrize("rounding", ["none", "nearest", "outward", "inward"])
def test_clip_raise_rejects_invalid_boxes_without_mutating_input(image, rounding):
    detections = invalid_detections()
    original = detections.xyxy.copy()
    with pytest.raises(ValueError, match="invalid boxes"):
        Detections.ClipBoxes(rounding=rounding, invalid="raise")(image, detections)
    np.testing.assert_array_equal(detections.xyxy, original)


def test_clip_raise_accepts_valid_boxes(image, detections):
    _, result = Detections.ClipBoxes(invalid="raise")(image, detections)
    np.testing.assert_array_equal(result.xyxy, detections.xyxy)


@pytest.mark.parametrize("kwargs", [{"rounding": "floor"}, {"rounding": None},
                                   {"invalid": "ignore"}, {"invalid": None}])
def test_clip_rejects_unknown_policies(kwargs):
    with pytest.raises(ValueError):
        Detections.ClipBoxes(**kwargs)


@pytest.mark.parametrize("color_space", [None, "BGR", "RGB"])
@pytest.mark.parametrize("copy", [True, False])
def test_crop_preserves_color_alignment_source_coordinates_and_copy_policy(image, color_space, copy):
    input_image = image if color_space is None else ImagePayload(image, color_space=color_space)
    detections = sv.Detections(xyxy=np.asarray([[-1.2, 2.5, 6.5, 7.5], [10, 8, 25, 20]]))
    original = deepcopy(detections)

    kept, crops = Detections.Crop(copy=copy)(input_image, detections)

    assert kept is detections
    assert detections == original
    assert len(crops) == 2
    for box, crop in zip(detections.xyxy, crops):
        np.testing.assert_array_equal(crop.array, sv.crop_image(image, box))
        assert crop.color_space == (color_space or "BGR")
        assert crop.layout == "HWC"
        assert np.shares_memory(crop.array, image) is not copy
    if copy:
        crops[0].array[:] = 0
        assert np.any(image[2:8, :6])


def test_crop_delegates_prepared_regions_to_supervision(image, detections, monkeypatch):
    crop_image = create_autospec(sv.crop_image, side_effect=sv.crop_image)
    monkeypatch.setattr(sv, "crop_image", crop_image)
    Detections.Crop()(image, detections)
    assert crop_image.call_count == len(detections)
    assert all(call.args[0] is image for call in crop_image.call_args_list)


@pytest.mark.parametrize("box", [
    [1, 2, 1, 6], [4, 2, 1, 6], [np.nan, 2, 6, 8], [-np.inf, 2, 6, 8],
    [25, 2, 30, 8], [1.1, 2.1, 1.4, 2.4], [1e20, 2, 2e20, 8],
])
def test_crop_raises_instead_of_silently_dropping_invalid_regions(image, box):
    detections = sv.Detections(xyxy=np.asarray([box], dtype=np.float64))
    with pytest.raises(ValueError, match="non-empty pixel regions"):
        Detections.Crop()(image, detections)


def test_crop_clips_extreme_finite_coordinates_before_integer_conversion(image):
    detections = sv.Detections(xyxy=np.asarray([[-1e20, -1e20, 1e20, 1e20]]))
    _, crops = Detections.Crop()(image, detections)
    np.testing.assert_array_equal(crops[0].array, image)


@pytest.mark.parametrize("operator", [Detections.ClipBoxes(), Detections.Crop()])
@pytest.mark.parametrize("bad_image", [
    np.zeros((0, 20, 3), dtype=np.uint8), np.zeros((16, 20), dtype=np.uint8),
    np.zeros((16, 20, 4), dtype=np.uint8), np.zeros((16, 20, 3), dtype=np.float32),
    ImagePayload(np.zeros((3, 16, 20), dtype=np.uint8), layout="CHW"),
    ImagePayload(np.zeros((16, 20, 3), dtype=np.uint8), color_space="HSV"),
])
def test_image_operators_reject_unsupported_images(operator, bad_image):
    with pytest.raises(ValueError, match="HWC uint8"):
        operator(bad_image, sv.Detections.empty())


@pytest.mark.parametrize("operator", [Detections.PadBoxes(3), Detections.ScaleBoxes(2, 0.5),
                                     Detections.MoveBoxes((-3, 4))])
def test_empty_geometry(operator):
    result = operator(sv.Detections.empty())
    assert len(result) == 0
    assert result.xyxy.shape == (0, 4)


@pytest.mark.parametrize("rounding", ["none", "nearest", "outward", "inward"])
@pytest.mark.parametrize("invalid", ["keep", "drop", "raise"])
def test_empty_clip_and_crop(image, rounding, invalid):
    output_image, detections = Detections.ClipBoxes(rounding=rounding, invalid=invalid)(
        image, sv.Detections.empty(),
    )
    kept, crops = Detections.Crop()(output_image, detections)
    assert output_image is image
    assert kept is detections
    assert crops == []


def test_merge_matches_supervision_and_is_not_box_union(detections):
    parts = [detections[:1], detections[1:]]
    original = deepcopy(parts)
    result = Detections.Merge()(parts)
    assert result == sv.Detections.merge(parts)
    assert len(result) == 2
    assert parts == original
    assert Detections.Merge()([]) == sv.Detections.empty()


def test_merge_delegates_validation_and_result_to_supervision(monkeypatch):
    expected = sv.Detections.empty()
    merge = create_autospec(sv.Detections.merge, return_value=expected)
    monkeypatch.setattr(sv.Detections, "merge", merge)
    parts = [expected]
    assert Detections.Merge()(parts) is expected
    merge.assert_called_once_with(parts)


@pytest.mark.parametrize("payload", [False, True])
def test_clip_crop_pipeline_validates_and_preserves_alignment(image, payload):
    input_image = ImagePayload(image, color_space="RGB") if payload else image
    pipeline = Pipeline([
        Detections.ClipBoxes(rounding="outward", invalid="drop"), Detections.Crop(),
    ])
    # A multi-argument entry needs a declared tuple boundary in ml-pipes-core.
    pipeline.validate(pipeline_input_type=tuple[ImagePayload if payload else np.ndarray, sv.Detections])
    kept, crops = pipeline((input_image, invalid_detections()))
    assert kept.class_id.tolist() == [0, 6, 7]
    assert len(crops) == len(kept)
    assert [crop.shape for crop in crops] == [(6, 5, 3), (1, 1, 3), (4, 4, 3)]


def test_geometry_pipeline_validates(detections):
    pipeline = Pipeline([
        Detections.Merge(), Detections.PadBoxes(2, 3),
        Detections.ScaleBoxes(2, 0.5), Detections.MoveBoxes((-1, 2)),
    ], auto_validate=True)
    result = pipeline([detections])
    assert len(result) == len(detections)
    assert_other_fields_equal(result, detections)
