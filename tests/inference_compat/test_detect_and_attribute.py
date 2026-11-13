"""VLM attribute configuration, JSON parsing, aligned attachment, and CLI."""
from importlib import import_module
from importlib.util import find_spec
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec

import cv2
import numpy as np
import pytest
import supervision as sv

from ml_pipes.standard import Gather, Scatter
from ml_pipes.supervision import Detections, ImageToArray, LabelAnnotator
from ml_pipes.vision import ConvertColorSpace, ImagePayload
from examples.qwen_local import INSTALL_COMMAND, LocalQwen3VL, require_local_qwen

LABELS = ("casual", "formal", "sportswear")


def prediction(class_id=1, confidence=0.9, *, x=20):
    return {
        "x": x, "y": 40, "width": 20, "height": 40,
        "class_id": class_id, "class": "person" if class_id == 1 else "bicycle",
        "confidence": confidence,
    }


def response(predictions):
    return [{"image": {"width": 120, "height": 100}, "predictions": predictions}]


@pytest.fixture
def models(monkeypatch):
    if find_spec("inference") is None:
        pytest.skip("Optional Inference dependency is not installed.")
    from ml_pipes.supervision import inference as inference_ops

    example = import_module("examples.run_detect_and_attribute")
    detector = create_autospec(inference_ops.Model, instance=True)
    detector.infer.return_value = response([prediction()])
    construct = create_autospec(inference_ops.get_model, return_value=detector)
    monkeypatch.setattr(inference_ops, "get_model", construct)
    generate = MagicMock(return_value=json.dumps({
        "fashion_style": "formal", "activity": "standing", "accessories": ["backpack"],
    }))

    def describe(self, image: ImagePayload) -> str:
        return generate(image)

    monkeypatch.setattr(LocalQwen3VL, "__call__", describe)
    qwen_factory = create_autospec(LocalQwen3VL, side_effect=lambda **kwargs: LocalQwen3VL(**kwargs))
    monkeypatch.setattr(example, "LocalQwen3VL", qwen_factory)
    monkeypatch.setattr(example, "require_local_qwen", MagicMock())
    plot = create_autospec(sv.plot_image)
    monkeypatch.setattr(sv, "plot_image", plot)
    return SimpleNamespace(
        example=example, detector=detector, construct=construct, plot=plot,
        generate=generate, qwen_factory=qwen_factory,
    )


@pytest.fixture
def attributes(models):
    Attribute = models.example.Attribute
    return {
        "fashion_style": Attribute(LABELS, description="Visible outfit style"),
        "activity": Attribute(("walking", "standing", "sitting")),
        "accessories": Attribute(("backpack", "hat", "handbag"), multiple=True),
    }


@pytest.fixture
def source_image(tmp_path):
    image = np.arange(100 * 120 * 3, dtype=np.uint8).reshape(100, 120, 3)
    path = tmp_path / "people.png"
    assert cv2.imwrite(str(path), image)
    return path, image


def test_invalid_regions_are_dropped_before_any_attribute_queries(models, attributes, source_image):
    models.detector.infer.return_value = response([
        prediction(x=200, confidence=0.99), prediction(), prediction(x=116, confidence=0.8),
    ])
    answers = [
        json.dumps({"fashion_style": "formal", "activity": "standing", "accessories": ["hat", "backpack"]}),
        json.dumps({"fashion_style": "casual", "activity": "walking", "accessories": []}),
    ]
    models.generate.side_effect = answers

    _, detections = models.example.build_pipeline(attributes=attributes)(source_image[0])

    np.testing.assert_array_equal(detections.xyxy, [[9, 18, 31, 62], [105, 18, 120, 62]])
    assert detections.data["fashion_style"] == ["formal", "casual"]
    assert detections.data["activity"] == ["standing", "walking"]
    assert detections.data["accessories"] == [["hat", "backpack"], []]
    assert detections.data["response"] == answers
    assert "label" not in detections.data and "attribute" not in detections.data
    assert models.generate.call_count == 2  # One query per crop, not per attribute.


def test_filtered_crops_and_all_attribute_columns_stay_aligned(models, attributes, source_image, monkeypatch):
    models.detector.infer.return_value = response([
        prediction(), prediction(confidence=0.7), prediction(class_id=2, x=50),
        prediction(confidence=0.4, x=70), prediction(x=100, confidence=0.8),
    ])
    models.generate.side_effect = [
        json.dumps({"fashion_style": "formal", "activity": "standing", "accessories": ["hat"]}),
        json.dumps({"fashion_style": "casual", "activity": "walking", "accessories": []}),
    ]
    pipeline = models.example.build_pipeline(attributes=attributes)
    pipeline.validate()
    annotator = next(op for op in pipeline.operators if isinstance(op, LabelAnnotator))
    labels = MagicMock(wraps=annotator.annotator.annotate)
    monkeypatch.setattr(annotator.annotator, "annotate", labels)

    annotated, detections = pipeline(source_image[0])

    models.construct.assert_called_once_with(model_id=models.example.DEFAULT_DETECTION_MODEL_ID)
    assert models.detector.infer.call_args.kwargs == {"confidence": 0.5}
    np.testing.assert_array_equal(models.detector.infer.call_args.args[0], source_image[1])
    assert detections.class_id.tolist() == [1, 1]
    assert list(detections.data["class_name"]) == ["person", "person"]
    assert detections.data["fashion_style"] == ["formal", "casual"]
    assert detections.data["activity"] == ["standing", "walking"]
    assert detections.data["accessories"] == [["hat"], []]
    for call, expected in zip(models.generate.call_args_list, (
        source_image[1][18:62, 9:31], source_image[1][18:62, 89:111],
    )):
        crop = call.args[0]
        assert crop.color_space == "RGB" and crop.layout == "HWC"
        np.testing.assert_array_equal(crop.array, expected[..., ::-1])
    assert labels.call_args.kwargs["labels"] == [
        "fashion_style: formal | activity: standing | accessories: hat",
        "fashion_style: casual | activity: walking | accessories: none",
    ]
    assert annotated.shape == source_image[1].shape
    assert not np.array_equal(annotated, source_image[1])
    assert models.plot.call_args.args[0] is annotated


@pytest.mark.parametrize("predictions", [
    [], [prediction(class_id=2)], [prediction(confidence=0.1)], [prediction(x=200)],
])
def test_no_usable_detections_leave_every_column_empty_and_qwen_uninitialized(
    models, attributes, source_image, predictions,
):
    models.detector.infer.return_value = response(predictions)
    pipeline = models.example.build_pipeline(attributes=attributes)
    qwen = next(op for op in pipeline.operators if isinstance(op, LocalQwen3VL))

    annotated, detections = pipeline(source_image[0])

    assert len(detections) == 0
    for name in (*attributes, "response"):
        assert detections.data[name] == []
    assert qwen._model is qwen._processor is None
    models.generate.assert_not_called()
    np.testing.assert_array_equal(annotated, source_image[1])


def test_models_are_reused_and_attribute_results_are_fresh(models, source_image):
    pipeline = models.example.build_pipeline(class_ids=(2,))
    models.detector.infer.return_value = response([prediction(class_id=2)])
    _, first = pipeline(source_image[0])
    models.generate.return_value = json.dumps({"fashion_style": "sportswear", "activity": "walking"})
    _, second = pipeline(source_image[0])
    models.detector.infer.return_value = response([])
    _, empty = pipeline(source_image[0])

    assert first.data["fashion_style"] == ["formal"]
    assert second.data["fashion_style"] == ["sportswear"]
    assert empty.data["fashion_style"] == []
    models.construct.assert_called_once()
    models.qwen_factory.assert_called_once()
    assert models.generate.call_count == 2


def test_descriptions_and_cardinality_reach_the_prompt(models, attributes):
    attributes["fashion_style"] = models.example.Attribute(
        (" casual : everyday clothing ", "formal: office attire: a suit or tie"),
        description=" visible outfit style ",
    )
    pipeline = models.example.build_pipeline(attributes=attributes)
    qwen = next(op for op in pipeline.operators if isinstance(op, LocalQwen3VL))

    assert '"fashion_style": {"description": "visible outfit style"' in qwen.prompt
    assert '"name": "casual", "description": "everyday clothing"' in qwen.prompt
    assert '"name": "formal", "description": "office attire: a suit or tie"' in qwen.prompt
    assert '"multiple": true' in qwen.prompt
    assert "one JSON object" in qwen.prompt
    assert "each attribute independently" in qwen.prompt
    assert "hidden personal attributes" in qwen.prompt
    assert "first line" not in qwen.prompt  # No conflicting describe-format instructions.
    models.generate.assert_not_called()


@pytest.mark.parametrize("value, expected", [
    ("casual", ("casual", "casual")),
    (" formal : office attire: a suit ", ("formal", "office attire: a suit")),
])
def test_candidate_names_and_descriptions_are_trimmed(models, value, expected):
    assert models.example.parse_named_text(value, "candidate label") == expected


@pytest.mark.parametrize("options, message", [
    ({"labels": ()}, "candidate label"),
    ({"labels": "casual"}, "candidate label"),
    ({"labels": ("",)}, "non-empty"),
    ({"labels": (1,)}, "string"),
    ({"labels": ("casual", " casual ")}, "unique"),
    ({"labels": (": everyday clothes",)}, "non-empty"),
    ({"labels": ("casual: ",)}, "description"),
    ({"description": 1}, "description"),
    ({"multiple": 1}, "bool"),
])
def test_invalid_attribute_specs_fail_before_model_loading(models, options, message):
    with pytest.raises(ValueError, match=message):
        models.example.Attribute(**{"labels": LABELS, **options})
    models.construct.assert_not_called()


@pytest.mark.parametrize("attributes, message", [
    ({}, "At least one"),
    ([], "At least one"),
    ({"": None}, "non-empty"),
    ({" activity ": None}, "whitespace"),
    ({1: None}, "strings"),
    ({"response": None}, "reserved"),
    ({"activity": None}, "Attribute configuration"),
])
def test_invalid_attribute_mappings_fail_before_loading_models(models, attributes, message):
    with pytest.raises(ValueError, match=message):
        models.example.build_pipeline(attributes=attributes)
    models.construct.assert_not_called()


@pytest.mark.parametrize("options, message", [
    ({"class_ids": ()}, "class ID"),
    ({"confidence": 1.1}, "between 0 and 1"),
    ({"padding": -0.1}, "finite and non-negative"),
    ({"padding": np.nan}, "finite and non-negative"),
    ({"max_tokens": 0}, "positive integer"),
    ({"max_tokens": 1.5}, "positive integer"),
    ({"max_tokens": True}, "positive integer"),
])
def test_invalid_pipeline_options_fail_before_loading_models(models, options, message):
    with pytest.raises(ValueError, match=message):
        models.example.build_pipeline(**options)
    models.construct.assert_not_called()


def test_configuration_and_native_resolution_crops_are_forwarded(models, source_image):
    attributes = {"condition": models.example.Attribute(("new", "worn"))}
    models.generate.return_value = '{"condition": "worn"}'
    pipeline = models.example.build_pipeline(
        attributes=attributes, detection_model_id="custom-detector", qwen_model_id="/models/qwen",
        device="mps", max_tokens=80, api_key="test-key", confidence=0.7, padding=0,
    )
    operators = pipeline.operators
    index = next(i for i, op in enumerate(operators) if isinstance(op, Scatter))
    assert isinstance(operators[index + 1], ConvertColorSpace)
    assert isinstance(operators[index + 2], LocalQwen3VL)
    assert isinstance(operators[index + 3], Gather)
    assert isinstance(operators[index + 4], models.example.ParseAttributes)

    _, detections = pipeline(source_image[0])

    models.construct.assert_called_once_with(model_id="custom-detector", api_key="test-key")
    models.qwen_factory.assert_called_once_with(
        prompt=models.example.build_attribute_prompt(attributes),
        model_id="/models/qwen", device="mps", max_tokens=80,
    )
    crop = models.generate.call_args.args[0]
    np.testing.assert_array_equal(crop.array, source_image[1][20:60, 10:30, ::-1])
    np.testing.assert_array_equal(detections.xyxy, [[10, 20, 30, 60]])
    assert sum(isinstance(op, ImageToArray) for op in operators) == 1
    assert detections.data["condition"] == ["worn"]


def test_parser_transposes_rows_without_flattening_or_mutating(models, attributes):
    answers = [
        '{"fashion_style":"casual","activity":"walking","accessories":["hat","backpack"]}',
        '{"fashion_style":"formal","activity":"standing","accessories":[]}',
    ]
    original = answers.copy()

    fields = models.example.ParseAttributes(attributes)(answers)

    assert fields == {
        "fashion_style": ["casual", "formal"],
        "activity": ["walking", "standing"],
        "accessories": [["hat", "backpack"], []],
        "response": answers,
    }
    assert fields["response"] is answers and answers == original
    assert models.example.ParseAttributes(attributes)([]) == {
        "fashion_style": [], "activity": [], "accessories": [], "response": [],
    }


@pytest.mark.parametrize("answer", [
    "", "not JSON", '{"fashion_style":', "[]", "null", '"casual"', "1",
    '{"fashion_style":"casual","fashion_style":"formal","activity":"walking"}',
])
def test_malformed_or_non_object_responses_keep_alignment_and_raw_text(models, attributes, answer):
    fields = models.example.ParseAttributes(attributes)([answer])

    assert fields == {
        "fashion_style": ["unknown"], "activity": ["unknown"],
        "accessories": [["unknown"]], "response": [answer],
    }


@pytest.mark.parametrize("value", [
    None, "Formal", "streetwear", "formal: office attire", ["formal"], 1, {}, True,
])
def test_invalid_scalar_fields_do_not_discard_other_attributes(models, attributes, value):
    answer = json.dumps({"fashion_style": value, "activity": "walking", "accessories": []})
    fields = models.example.ParseAttributes(attributes)([answer])

    assert fields["fashion_style"] == ["unknown"]
    assert fields["activity"] == ["walking"]
    assert fields["accessories"] == [[]]


@pytest.mark.parametrize("value, expected", [
    ([], []), (["hat", "backpack"], ["hat", "backpack"]), (["unknown"], ["unknown"]),
    ("hat", ["unknown"]), (None, ["unknown"]), ([["hat"]], ["unknown"]),
    (["hat", "watch"], ["unknown"]), (["hat", "unknown"], ["unknown"]), ([1], ["unknown"]),
])
def test_multi_value_cardinality_and_unknown_semantics(models, attributes, value, expected):
    answer = json.dumps({"accessories": value})
    fields = models.example.ParseAttributes(attributes)([answer])
    assert fields["accessories"] == [expected]
    assert fields["fashion_style"] == ["unknown"]


def test_extra_keys_are_ignored_and_complete_json_fences_are_accepted(models, attributes):
    fence = chr(96) * 3
    answer = fence + 'json\n{"fashion_style":"casual","activity":"unknown","extra":"ignored"}\n' + fence
    fields = models.example.ParseAttributes(attributes)([answer])

    assert set(fields) == {*attributes, "response"}
    assert fields["fashion_style"] == ["casual"]
    assert fields["activity"] == ["unknown"]
    assert fields["response"] == [answer]


def test_parsed_columns_attach_without_mutating_geometry_metadata_or_nested_values(models, attributes):
    detections = sv.Detections(
        xyxy=np.asarray([[0, 0, 10, 10]], dtype=float),
        class_id=np.array([1]), confidence=np.array([0.9]),
        data={"class_name": np.array(["person"])}, metadata={"source": "test"},
    )
    answers = ['{"fashion_style":"casual","activity":"walking","accessories":["hat","backpack"]}']
    fields = models.example.ParseAttributes(attributes)(answers)

    result = Detections.AttachData()(detections, fields)

    assert list(result.data["class_name"]) == ["person"]
    assert result.data["accessories"] is fields["accessories"]
    assert result.data["accessories"][0] == ["hat", "backpack"]
    assert result.data["response"] is answers
    assert result.metadata == {"source": "test"}
    np.testing.assert_array_equal(result.xyxy, detections.xyxy)
    np.testing.assert_array_equal(result.class_id, detections.class_id)
    np.testing.assert_array_equal(result.confidence, detections.confidence)
    assert set(detections.data) == {"class_name"}


@pytest.mark.parametrize("count", [0, 2])
def test_attachment_still_checks_column_alignment(models, attributes, count):
    detections = sv.Detections(xyxy=np.asarray([[0, 0, 10, 10]], dtype=float))
    fields = models.example.ParseAttributes(attributes)(["{}"] * count)
    with pytest.raises(ValueError, match="exactly one"):
        Detections.AttachData()(detections, fields)


def test_annotation_formats_collections_without_changing_the_data(models, attributes):
    data = {"fashion_style": "casual", "activity": "walking", "accessories": ["hat", "backpack"]}
    text = models.example.format_attributes(data, attributes)
    assert text == "fashion_style: casual | activity: walking | accessories: hat, backpack"
    assert data["accessories"] == ["hat", "backpack"]
    assert models.example.format_attributes({"style": "casual"}, {"style": attributes["fashion_style"]}) == "casual"


def test_cli_missing_qwen_checks_runtime_before_download_or_model_loading(models, monkeypatch, capsys):
    monkeypatch.setattr(models.example, "require_local_qwen", require_local_qwen)
    monkeypatch.setitem(sys.modules, "torch", None)
    download = MagicMock()
    monkeypatch.setattr(models.example, "download_assets", download)
    monkeypatch.setattr("sys.argv", ["run_detect_and_attribute.py"])

    with pytest.raises(SystemExit) as error:
        models.example.main()

    assert error.value.code == 1
    assert INSTALL_COMMAND in capsys.readouterr().err
    download.assert_not_called()
    models.construct.assert_not_called()


def test_cli_help_works_without_qwen(models, monkeypatch, capsys):
    monkeypatch.setattr(models.example, "require_local_qwen", require_local_qwen)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr("sys.argv", ["run_detect_and_attribute.py", "--help"])
    with pytest.raises(SystemExit) as error:
        models.example.main()
    assert error.value.code == 0
    assert "--attributes" in capsys.readouterr().out
    models.construct.assert_not_called()


def test_cli_forwards_qwen_and_detector_options(models, source_image, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", [
        "run_detect_and_attribute.py", "--input", str(source_image[0]),
        "--qwen-model-id", "/models/qwen", "--device", "cuda:1", "--max-tokens", "80", "--api-key", "test-key",
    ])
    assert models.example.main() == 0
    models.qwen_factory.assert_called_once_with(
        prompt=models.example.build_attribute_prompt(models.example.DEFAULT_ATTRIBUTES),
        model_id="/models/qwen", device="cuda:1", max_tokens=80,
    )
    models.construct.assert_called_once_with(model_id=models.example.DEFAULT_DETECTION_MODEL_ID, api_key="test-key")
    assert json.loads(capsys.readouterr().out)[0]["attributes"] == {
        "fashion_style": "formal", "activity": "standing", "accessories": ["backpack"],
    }


def test_cli_defaults_to_people_walking_and_all_default_attributes(models, source_image, monkeypatch, capsys):
    download = MagicMock(return_value=str(source_image[0]))
    monkeypatch.setattr(models.example, "download_assets", download)
    monkeypatch.setattr("sys.argv", ["run_detect_and_attribute.py"])

    assert models.example.main() == 0

    download.assert_called_once_with(sv.assets.ImageAssets.PEOPLE_WALKING, directory=models.example.ASSETS_DIR)
    row = json.loads(capsys.readouterr().out)[0]
    assert set(models.example.DEFAULT_ATTRIBUTES) == {"fashion_style", "activity", "accessories"}
    assert models.example.DEFAULT_ATTRIBUTES["accessories"].multiple is True
    assert row["attributes"] == {
        "fashion_style": "formal", "activity": "standing", "accessories": ["backpack"],
    }
    assert "label" not in row and "attribute" not in row
    models.plot.assert_called_once()


def test_cli_loads_multi_attribute_config_and_saves_nested_results(models, source_image, tmp_path, monkeypatch, capsys):
    config = {
        "condition": {"labels": ["new: unused", "worn: visibly used"], "description": "Visible condition"},
        "accessories": {"labels": ["hat", "backpack"], "multiple": True},
    }
    config_path = tmp_path / "attributes.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "results.json"
    output_image = tmp_path / "annotated.png"
    answer = '{"condition":"worn","accessories":["hat","backpack"]}'
    models.generate.return_value = answer
    monkeypatch.setattr("sys.argv", [
        "run_detect_and_attribute.py", "--input", str(source_image[0]), "--attributes", str(config_path),
        "--padding", "0", "--output", str(output), "--output-image", str(output_image),
    ])

    assert models.example.main() == 0

    rows = json.loads(capsys.readouterr().out)
    assert rows == json.loads(output.read_text())
    assert rows[0]["attributes"] == {"condition": "worn", "accessories": ["hat", "backpack"]}
    assert rows[0]["response"] == answer
    assert rows[0]["xyxy"] == [10.0, 20.0, 30.0, 60.0]
    np.testing.assert_array_equal(cv2.imread(str(output_image)), models.plot.call_args.args[0])
    prompt = models.qwen_factory.call_args.kwargs["prompt"]
    assert "Visible condition" in prompt and "visibly used" in prompt


@pytest.mark.parametrize("config", [
    "{", "[]", "{}", '{"response":{"labels":["x"]}}',
    '{"activity":{"labels":[]}}', '{"activity":{"labels":["x"],"multiple":"yes"}}',
    '{"activity":{"labels":["x"],"extra":true}}', '{"activity":null}',
    '{"activity":{"labels":["x"]},"activity":{"labels":["y"]}}',
])
def test_invalid_cli_config_fails_before_runtime_and_model_loading(models, tmp_path, monkeypatch, config):
    path = tmp_path / "attributes.json"
    path.write_text(config)
    monkeypatch.setattr("sys.argv", ["run_detect_and_attribute.py", "--attributes", str(path)])
    with pytest.raises(SystemExit) as error:
        models.example.main()
    assert error.value.code == 2
    models.example.require_local_qwen.assert_not_called()
    models.construct.assert_not_called()


def test_cli_reports_image_write_failure(models, source_image, tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "run_detect_and_attribute.py", "--input", str(source_image[0]),
        "--output-image", str(tmp_path / "annotated.png"),
    ])
    monkeypatch.setattr(cv2, "imwrite", MagicMock(return_value=False))
    with pytest.raises(OSError, match="Failed to save annotated image"):
        models.example.main()
