"""Per-detection data attachment and typed pipeline integration."""
from copy import deepcopy
from types import MappingProxyType

import numpy as np
import pytest
import supervision as sv

from ml_pipes.core import Pipeline
from ml_pipes.supervision import Detections


@pytest.fixture
def detections():
    return sv.Detections(
        xyxy=np.asarray([[1, 2, 5, 6], [10, 8, 14, 12]], dtype=np.float32),
        mask=np.ones((2, 16, 20), dtype=bool),
        confidence=np.asarray([0.9, 0.8]),
        class_id=np.asarray([2, 3]),
        tracker_id=np.asarray([11, 12]),
        data={"name": np.asarray(["first", "second"]),
              "keypoints": np.arange(12).reshape(2, 3, 2)},
        metadata={"source": "test"},
    )


@pytest.mark.parametrize("values", [
    ["walking", "standing"], [0.2, 0.8], [True, False],
    np.asarray(["walking", "standing"]), np.asarray([1, 2], dtype=np.int32),
    np.asarray([[1, 2], [3, 4]], dtype=np.float32),
    np.arange(12).reshape(2, 3, 2),
    [{"label": "walking"}, {"label": "standing"}],
    [[1], [2, 3]], [{"hat", "glasses"}, set()],
    [("hat",), ("hat", "glasses")], [np.array([1]), np.array([2, 3])],
])
def test_attach_field_preserves_values_and_every_other_field(detections, values):
    original = deepcopy(detections)

    result = Detections.AttachDataField("response")(detections, values)

    assert result.data.keys() == detections.data.keys() | {"response"}
    assert detections == original
    assert result is not detections
    assert result.data is not detections.data
    assert "response" not in detections.data
    assert result.metadata == {"source": "test"}
    assert result.data["response"] is values
    for name in ("xyxy", "mask", "confidence", "class_id", "tracker_id", "metadata"):
        assert getattr(result, name) is getattr(detections, name)
    for key in detections.data:
        assert result.data[key] is detections.data[key]


@pytest.mark.parametrize("values", [[], np.asarray([], dtype=str), np.empty((0, 3))])
def test_empty_values_can_be_attached_to_empty_detections(values):
    detections = sv.Detections.empty()

    result = Detections.AttachDataField("response")(detections, values)

    assert len(result) == 0
    assert result.data["response"] is values
    assert "response" not in detections.data


@pytest.mark.parametrize("values", [[], ["one"], ["one", "two", "three"], np.empty((3, 2))])
def test_misaligned_data_raises_without_mutating_detections(detections, values):
    original = deepcopy(detections)
    with pytest.raises(ValueError, match="exactly one"):
        Detections.AttachDataField("response")(detections, values)
    assert detections == original


@pytest.mark.parametrize("values", ["walking", 1, None, {"one": 1}, ("one", "two"), iter([1, 2])])
def test_unsupported_values_are_not_broadcast_or_implicitly_iterated(detections, values):
    with pytest.raises(TypeError, match="list or NumPy array"):
        Detections.AttachDataField("response")(detections, values)
    assert "response" not in detections.data


def test_zero_dimensional_array_is_not_broadcast(detections):
    with pytest.raises(ValueError, match="scalars are not broadcast"):
        Detections.AttachDataField("response")(detections, np.asarray("walking"))
    assert "response" not in detections.data


def test_existing_key_requires_explicit_overwrite(detections):
    original = deepcopy(detections)
    with pytest.raises(ValueError, match="already exists.*overwrite=True"):
        Detections.AttachDataField("name")(detections, ["new first", "new second"])
    assert detections == original

    values = ["new first", "new second"]
    result = Detections.AttachDataField("name", overwrite=True)(detections, values)

    assert result.data["name"] is values
    assert detections == original
    np.testing.assert_array_equal(result.data["keypoints"], detections.data["keypoints"])


@pytest.mark.parametrize("key", ["", " \t", None, 1])
def test_field_key_must_be_a_nonempty_string(key):
    with pytest.raises(ValueError, match="non-empty string"):
        Detections.AttachDataField(key)


@pytest.mark.parametrize("overwrite", [None, "replace", 1])
@pytest.mark.parametrize("operator", [Detections.AttachData, lambda **kw: Detections.AttachDataField("response", **kw)])
def test_overwrite_must_be_explicit_boolean(overwrite, operator):
    with pytest.raises(TypeError, match="bool"):
        operator(overwrite=overwrite)


def test_key_is_not_normalized_and_metadata_is_a_separate_namespace(detections):
    result = Detections.AttachDataField("source")(detections, ["crop-1", "crop-2"])
    result = Detections.AttachDataField(" name ")(result, ["one", "two"])

    assert result.data["source"] == ["crop-1", "crop-2"]
    assert result.metadata == detections.metadata == {"source": "test"}
    assert result.data[" name "] == ["one", "two"]
    assert list(result.data["name"]) == ["first", "second"]


@pytest.mark.parametrize("mapping_type", [dict, MappingProxyType])
def test_attach_mapping_preserves_columns_and_nested_collections(detections, mapping_type):
    original = deepcopy(detections)
    accessories = [{"hat", "glasses"}, set()]
    fields = mapping_type({
        "age": np.array([30, 40]),
        "occupation": ["teacher", "engineer"],
        "accessories": accessories,
        "ragged": [[1], [2, 3]],
        "details": [{"colors": ["blue"]}, {"colors": []}],
    })
    keys = list(fields)

    result = Detections.AttachData()(detections, fields)

    assert detections == original
    assert result is not detections
    assert result.data is not detections.data and result.data is not fields
    assert list(fields) == keys
    for key, values in fields.items():
        assert result.data[key] is values
    assert result.data["accessories"][0] is accessories[0]
    for name in ("xyxy", "mask", "confidence", "class_id", "tracker_id", "metadata"):
        assert getattr(result, name) is getattr(detections, name)
    for key in detections.data:
        assert result.data[key] is detections.data[key]

    # Supervision row selection retains each per-detection collection.
    selected = result[np.array([False, True])]
    assert selected.data["accessories"] == [set()]
    assert selected.data["ragged"] == [[2, 3]]


@pytest.mark.parametrize("data", [None, [], [("label", ["a", "b"])], np.array([1, 2]), "label"])
def test_attach_requires_a_mapping(detections, data):
    with pytest.raises(TypeError, match="mapping"):
        Detections.AttachData()(detections, data)


@pytest.mark.parametrize("key", ["", " \t", None, 1])
def test_mapping_keys_must_be_nonempty_strings(detections, key):
    with pytest.raises(ValueError, match="non-empty string"):
        Detections.AttachData()(detections, {key: [1, 2]})


@pytest.mark.parametrize("bad_field, error, message", [
    ([], ValueError, "exactly one"),
    (["one"], ValueError, "exactly one"),
    (np.empty((3, 2)), ValueError, "exactly one"),
    (np.asarray(1), ValueError, "scalars are not broadcast"),
    ("walking", TypeError, "list or NumPy array"),
    (("one", "two"), TypeError, "list or NumPy array"),
    (None, TypeError, "list or NumPy array"),
])
def test_mapping_validates_every_field_atomically(detections, bad_field, error, message):
    original = deepcopy(detections)
    fields = {"name": ["replacement", "names"], "invalid": bad_field}

    with pytest.raises(error, match=message):
        Detections.AttachData(overwrite=True)(detections, fields)

    assert detections == original
    assert list(fields) == ["name", "invalid"]


def test_mapping_collision_does_not_attach_earlier_fields(detections):
    original = deepcopy(detections)
    fields = {"new": [1, 2], "name": ["new first", "new second"]}

    with pytest.raises(ValueError, match="already exists.*overwrite=True"):
        Detections.AttachData()(detections, fields)

    assert detections == original
    result = Detections.AttachData(overwrite=True)(detections, fields)
    assert result.data["name"] is fields["name"]
    assert result.data["new"] is fields["new"]
    assert detections == original


@pytest.mark.parametrize("empty", [False, True])
def test_empty_mapping_returns_new_detections_without_changing_data(detections, empty):
    if empty:
        detections = sv.Detections.empty()
    result = Detections.AttachData()(detections, {})
    assert result == detections
    assert result is not detections
    assert result.data is not detections.data


def test_mapping_of_empty_fields_can_be_attached_to_empty_detections():
    fields = {"label": [], "embedding": np.empty((0, 3))}
    result = Detections.AttachData()(sv.Detections.empty(), fields)
    for key, values in fields.items():
        assert result.data[key] is values


@pytest.mark.parametrize("array", [False, True])
def test_attach_field_pipeline_validates_and_keeps_row_order(detections, array):
    values = np.asarray(["standing", "walking"]) if array else ["standing", "walking"]
    pipeline = Pipeline([Detections.AttachDataField("response")])
    pipeline.validate(pipeline_input_type=tuple[sv.Detections, np.ndarray if array else list[str]])

    result = pipeline((detections, values))

    assert list(result.data["response"]) == ["standing", "walking"]
    np.testing.assert_array_equal(result.tracker_id, detections.tracker_id)


@pytest.mark.parametrize("array", [False, True])
def test_attach_mapping_pipeline_validates_and_keeps_row_order(detections, array):
    values = np.asarray(["standing", "walking"]) if array else ["standing", "walking"]
    pipeline = Pipeline([Detections.AttachData()])
    pipeline.validate(
        pipeline_input_type=tuple[sv.Detections, dict[str, np.ndarray if array else list[str]]],
    )

    result = pipeline((detections, {"response": values}))

    assert result.data["response"] is values
    np.testing.assert_array_equal(result.tracker_id, detections.tracker_id)
