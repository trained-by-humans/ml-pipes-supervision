"""Local Qwen attribute pipelines and the retained standalone CLIP operator."""
from importlib import import_module
from importlib.metadata import version
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
    clip = MagicMock(spec=["embed_text", "embed_image"])
    # Unequal lengths catch accidental dot-product ranking without normalization.
    clip.embed_text.return_value = np.diag([100.0, 2.0, 3.0])
    clip.embed_image.return_value = np.array([[1.0, 2.0, 0.0]])
    construct = create_autospec(
        inference_ops.get_model,
        side_effect=lambda model_id, **_: clip if model_id.startswith("clip/") else detector,
    )
    monkeypatch.setattr(inference_ops, "get_model", construct)
    generate = MagicMock(return_value="formal\nOffice attire.")

    def describe(self, image: ImagePayload) -> str:
        return generate(image)

    monkeypatch.setattr(LocalQwen3VL, "__call__", describe)
    qwen_factory = create_autospec(
        LocalQwen3VL, side_effect=lambda **kwargs: LocalQwen3VL(**kwargs),
    )
    monkeypatch.setattr(example, "LocalQwen3VL", qwen_factory)
    monkeypatch.setattr(example, "require_local_qwen", MagicMock())
    load_clip = example._load_clip_dependencies
    clip_check = MagicMock()
    monkeypatch.setattr(example, "_load_clip_dependencies", clip_check)
    plot = create_autospec(sv.plot_image)
    monkeypatch.setattr(sv, "plot_image", plot)
    return SimpleNamespace(
        example=example, detector=detector, clip=clip, construct=construct, plot=plot,
        generate=generate, qwen_factory=qwen_factory,
        clip_check=clip_check, load_clip=load_clip,
    )


@pytest.fixture
def source_image(tmp_path):
    image = np.arange(100 * 120 * 3, dtype=np.uint8).reshape(100, 120, 3)
    path = tmp_path / "people.png"
    assert cv2.imwrite(str(path), image)
    return path, image


def test_invalid_regions_are_dropped_and_labels_annotate_surviving_roi_boxes(models, source_image):
    models.detector.infer.return_value = response([
        prediction(x=200, confidence=0.99), prediction(), prediction(x=116, confidence=0.8),
    ])
    models.generate.side_effect = ["formal", "casual"]

    _, detections = models.example.build_pipeline(attribute="fashion style", labels=LABELS)(source_image[0])

    np.testing.assert_array_equal(detections.xyxy, [[9, 18, 31, 62], [105, 18, 120, 62]])
    assert list(detections.data["label"]) == ["formal", "casual"]
    assert list(detections.data["response"]) == ["formal", "casual"]
    assert "source_xyxy" not in detections.data
    assert models.generate.call_count == 2


def test_only_filtered_crops_reach_qwen_and_labels_stay_aligned(models, source_image, monkeypatch):
    models.detector.infer.return_value = response([
        prediction(), prediction(confidence=0.7), prediction(class_id=2, x=50),
        prediction(confidence=0.4, x=70), prediction(x=100, confidence=0.8),
    ])
    answers = ["formal\nOffice attire.", "casual\nEveryday clothes."]
    models.generate.side_effect = answers
    pipeline = models.example.build_pipeline(attribute="fashion style", labels=LABELS)
    pipeline.validate()
    annotator = next(op for op in pipeline.operators if isinstance(op, LabelAnnotator))
    label_spy = MagicMock(wraps=annotator.annotator.annotate)
    monkeypatch.setattr(annotator.annotator, "annotate", label_spy)

    annotated, detections = pipeline(source_image[0])

    models.construct.assert_called_once_with(model_id=models.example.DEFAULT_DETECTION_MODEL_ID)
    assert models.detector.infer.call_args.kwargs == {"confidence": 0.5}
    np.testing.assert_array_equal(models.detector.infer.call_args.args[0], source_image[1])
    assert detections.class_id.tolist() == [1, 1]
    assert list(detections.data["attribute"]) == ["fashion style", "fashion style"]
    assert list(detections.data["label"]) == ["formal", "casual"]
    assert list(detections.data["response"]) == answers
    assert "similarity" not in detections.data
    np.testing.assert_array_equal(detections.xyxy, [[9, 18, 31, 62], [89, 18, 111, 62]])
    assert list(detections.data["class_name"]) == ["person", "person"]
    assert models.generate.call_count == 2
    models.clip.embed_text.assert_not_called()
    models.clip.embed_image.assert_not_called()
    models.clip_check.assert_not_called()
    for call, expected in zip(models.generate.call_args_list, (
        source_image[1][18:62, 9:31], source_image[1][18:62, 89:111],
    )):
        crop = call.args[0]
        assert crop.color_space == "RGB" and crop.layout == "HWC"
        assert crop.array.dtype == np.uint8
        np.testing.assert_array_equal(crop.array, expected[..., ::-1])
    assert label_spy.call_args.kwargs["labels"] == ["formal", "casual"]
    assert annotated.shape == source_image[1].shape
    assert not np.array_equal(annotated, source_image[1])
    models.plot.assert_called_once()
    assert models.plot.call_args.args[0] is annotated


@pytest.mark.parametrize("predictions", [
    [], [prediction(class_id=2)], [prediction(confidence=0.1)], [prediction(x=200)],
])
def test_no_usable_detections_leave_qwen_uninitialized(models, source_image, predictions):
    models.detector.infer.return_value = response(predictions)
    pipeline = models.example.build_pipeline(attribute="style", labels=LABELS)
    qwen = next(op for op in pipeline.operators if isinstance(op, LocalQwen3VL))

    annotated, detections = pipeline(source_image[0])

    assert len(detections) == 0
    assert list(detections.data["label"]) == []
    assert list(detections.data["response"]) == []
    assert list(detections.data["attribute"]) == []
    assert qwen._model is qwen._processor is None
    models.construct.assert_called_once_with(model_id=models.example.DEFAULT_DETECTION_MODEL_ID)
    models.clip.embed_text.assert_not_called()
    models.clip.embed_image.assert_not_called()
    models.generate.assert_not_called()
    np.testing.assert_array_equal(annotated, source_image[1])
    models.plot.assert_called_once()


def test_qwen_operator_is_reused_across_images_and_results_are_fresh(models, source_image):
    pipeline = models.example.build_pipeline(attribute="style", labels=LABELS, class_ids=(2,))
    models.detector.infer.return_value = response([prediction(class_id=2), prediction(class_id=1, x=100)])
    _, first = pipeline(source_image[0])
    models.generate.return_value = "sportswear\nAthletic clothes."
    _, second = pipeline(source_image[0])
    models.detector.infer.return_value = response([])
    _, empty = pipeline(source_image[0])

    assert first.class_id.tolist() == second.class_id.tolist() == [2]
    assert list(first.data["label"]) == ["formal"]
    assert list(second.data["label"]) == ["sportswear"]
    assert list(empty.data["label"]) == []
    models.construct.assert_called_once_with(model_id=models.example.DEFAULT_DETECTION_MODEL_ID)
    models.qwen_factory.assert_called_once()
    assert models.generate.call_count == 2
    models.clip.embed_text.assert_not_called()
    models.clip.embed_image.assert_not_called()


def test_clip_operator_forwards_bgr_pixels_unchanged_to_model_preprocessing(models):
    image = np.full((12, 8, 3), [10, 20, 230], dtype=np.uint8)

    models.example.CLIPClassify("style", LABELS)(image)

    assert models.clip.embed_image.call_args.args[0] is image
    np.testing.assert_array_equal(image[0, 0], [10, 20, 230])


def test_clip_dependencies_and_model_loading_are_lazy_and_checked_once(models):
    operator = models.example.CLIPClassify("style", LABELS)
    models.clip_check.assert_not_called()
    models.construct.assert_not_called()

    image = np.zeros((12, 8, 3), dtype=np.uint8)
    operator(image)
    operator(image)

    models.clip_check.assert_called_once_with()
    models.construct.assert_called_once_with(model_id="clip/ViT-B-32")
    models.clip.embed_text.assert_called_once()
    assert models.clip.embed_image.call_count == 2


def test_clip_install_guidance_preserves_the_installed_inference_release(models):
    assert models.example.CLIP_INSTALL_COMMAND == f'python -m pip install "inference[clip]=={version("inference")}"'


def test_clip_missing_runtime_has_install_guidance_before_model_loading(models, monkeypatch):
    monkeypatch.setattr(models.example, "_load_clip_dependencies", models.load_clip)
    monkeypatch.setitem(sys.modules, "inference.models.clip", None)
    operator = models.example.CLIPClassify("style", LABELS)

    with pytest.raises(ImportError, match="CLIPClassify requires") as error:
        operator(np.zeros((12, 8, 3), dtype=np.uint8))

    assert models.example.CLIP_INSTALL_COMMAND in str(error.value)
    assert isinstance(error.value.__cause__, ModuleNotFoundError)
    models.construct.assert_not_called()
    models.clip.embed_text.assert_not_called()
    models.clip.embed_image.assert_not_called()


def test_require_clip_exits_cleanly_with_install_guidance_when_runtime_is_missing(models, monkeypatch, capsys):
    monkeypatch.setattr(models.example, "_load_clip_dependencies", models.load_clip)
    monkeypatch.setitem(sys.modules, "inference.models.clip", None)

    with pytest.raises(SystemExit) as error:
        models.example.require_clip()

    assert error.value.code == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert models.example.CLIP_INSTALL_COMMAND in output.err
    assert "Traceback" not in output.err
    models.construct.assert_not_called()


def test_require_clip_checks_backend_without_loading_weights(models, monkeypatch, capsys):
    backend = MagicMock()
    monkeypatch.setattr(models.example, "_load_clip_dependencies", models.load_clip)
    monkeypatch.setitem(sys.modules, "inference.models.clip", SimpleNamespace(Clip=backend))

    models.example.require_clip()

    assert capsys.readouterr() == ("", "")
    backend.assert_not_called()
    models.construct.assert_not_called()


def test_require_clip_does_not_hide_non_import_failures(models):
    models.clip_check.side_effect = RuntimeError("broken native runtime")

    with pytest.raises(RuntimeError, match="broken native runtime"):
        models.example.require_clip()


def test_qwen_pipeline_runs_without_clip_backend(models, source_image, monkeypatch):
    monkeypatch.setattr(models.example, "_load_clip_dependencies", models.load_clip)
    monkeypatch.setitem(sys.modules, "inference.models.clip", None)

    _, detections = models.example.build_pipeline()(source_image[0])

    assert list(detections.data["label"]) == ["formal"]
    models.generate.assert_called_once()
    models.clip.embed_image.assert_not_called()


@pytest.mark.parametrize("value, expected", [
    ("casual", ("casual", "casual")),
    (" fashion style : overall style of the outfit ", ("fashion style", "overall style of the outfit")),
    ("formal: office attire: a suit, shirt, or tie", ("formal", "office attire: a suit, shirt, or tie")),
])
def test_named_text_splits_only_the_first_colon_and_trims_spaces(models, value, expected):
    assert models.example.parse_named_text(value, "candidate label") == expected


def test_descriptions_reach_qwen_but_only_label_names_are_annotated(models, source_image, monkeypatch):
    pipeline = models.example.build_pipeline(
        attribute=" fashion style : overall visual style of the outfit ",
        labels=(
            " casual : relaxed everyday clothing ",
            "formal: office attire: a suit, shirt, or tie",
            "sportswear",  # Plain labels still work alongside described labels.
        ),
    )
    operator = next(op for op in pipeline.operators if isinstance(op, LocalQwen3VL))
    annotator = next(op for op in pipeline.operators if isinstance(op, LabelAnnotator))
    label_spy = MagicMock(wraps=annotator.annotator.annotate)
    monkeypatch.setattr(annotator.annotator, "annotate", label_spy)
    pipeline.validate()

    _, detections = pipeline(source_image[0])

    assert '"name": "fashion style", "description": "overall visual style of the outfit"' in operator.prompt
    assert '"name": "casual", "description": "relaxed everyday clothing"' in operator.prompt
    assert '"name": "formal", "description": "office attire: a suit, shirt, or tie"' in operator.prompt
    assert '"name": "sportswear", "description": "sportswear"' in operator.prompt
    assert "copy it exactly onto the first line" in operator.prompt
    assert "Do not invent labels" in operator.prompt
    assert "visible evidence" in operator.prompt
    assert "hidden personal attributes" in operator.prompt
    assert list(detections.data["attribute"]) == ["fashion style"]
    assert list(detections.data["label"]) == ["formal"]
    assert label_spy.call_args.kwargs["labels"] == ["formal"]


def test_plain_attribute_uses_label_descriptions_when_supplied(models):
    operator = models.example.CLIPClassify(
        "style", ("casual: everyday clothing", "formal", "sportswear: athletic clothing"),
    )
    assert operator.attribute == "style"
    assert operator.labels == LABELS
    assert operator.prompts == (
        "A photo of a subject. style: everyday clothing.",
        "A photo of a subject. style: formal.",
        "A photo of a subject. style: athletic clothing.",
    )
    models.construct.assert_not_called()


def test_candidate_batches_preserve_label_order_and_cache_all_text_embeddings(models, monkeypatch):
    labels = tuple(f"label-{index}" for index in range(5))
    monkeypatch.setattr(models.example, "CLIP_MAX_BATCH_SIZE", 2)
    models.clip.embed_text.side_effect = lambda prompts: np.asarray([
        np.eye(5)[labels.index(prompt.rsplit(": ", 1)[1][:-1])] for prompt in prompts
    ])
    models.clip.embed_image.return_value = np.array([[0.0, 0.0, 0.0, 0.0, 2.0]])
    operator = models.example.CLIPClassify("condition", labels)
    image = np.zeros((20, 10, 3), dtype=np.uint8)

    first = operator(image)
    second = operator(image)

    assert first == second == models.example.Classification("label-4", 1.0)
    assert models.clip.embed_text.call_count == 3
    assert [len(call.args[0]) for call in models.clip.embed_text.call_args_list] == [2, 2, 1]
    assert [prompt for call in models.clip.embed_text.call_args_list for prompt in call.args[0]] == list(operator.prompts)
    assert models.clip.embed_image.call_count == 2
    models.construct.assert_called_once_with(model_id="clip/ViT-B-32")


@pytest.mark.parametrize("vector, expected", [
    ([1.0, 1.0, 0.0], "casual"),  # tied candidates retain supplied order
    ([-1.0, -2.0, -3.0], "casual"),  # no automatic abstention or invented label
])
def test_closed_set_selects_supplied_label_even_for_ties_or_negative_scores(models, vector, expected):
    models.clip.embed_image.return_value = np.asarray([vector])
    result = models.example.CLIPClassify("style", LABELS)(np.zeros((10, 10, 3), dtype=np.uint8))
    assert result.label == expected
    assert -1 <= result.similarity <= 1


def test_configuration_is_forwarded_and_qwen_receives_source_resolution_crops(models, source_image):
    pipeline = models.example.build_pipeline(
        attribute="  condition ", labels=(" casual ", "formal", "sportswear"),
        detection_model_id="custom-detector", qwen_model_id="/models/qwen", device="mps", max_tokens=80,
        api_key="test-key", confidence=0.7, padding=0,
    )
    operators = pipeline.operators
    index = next(i for i, op in enumerate(operators) if isinstance(op, Scatter))
    assert isinstance(operators[index + 1], ConvertColorSpace)
    assert isinstance(operators[index + 2], LocalQwen3VL)
    assert isinstance(operators[index + 3], Gather)

    _, detections = pipeline(source_image[0])

    models.construct.assert_called_once_with(model_id="custom-detector", api_key="test-key")
    models.qwen_factory.assert_called_once_with(
        prompt=models.example.build_attribute_prompt("condition", LABELS),
        model_id="/models/qwen", device="mps", max_tokens=80,
    )
    assert models.detector.infer.call_args.kwargs == {"confidence": 0.7}
    crop = models.generate.call_args.args[0]
    assert crop.color_space == "RGB"
    assert crop.array.shape == (40, 20, 3)
    np.testing.assert_array_equal(crop.array, source_image[1][20:60, 10:30, ::-1])
    np.testing.assert_array_equal(detections.xyxy, [[10, 20, 30, 60]])
    assert sum(isinstance(op, ImageToArray) for op in operators) == 1
    assert not any(isinstance(op, models.example.CLIPClassify) for op in operators)
    assert list(detections.data["attribute"]) == ["condition"]


@pytest.mark.parametrize("options, message", [
    ({"attribute": " "}, "non-empty attribute"),
    ({"attribute": ": overall outfit style"}, "non-empty attribute"),
    ({"attribute": "style: "}, "attribute description"),
    ({"labels": ()}, "candidate label"),
    ({"labels": "casual"}, "candidate label"),
    ({"labels": ("casual", " ")}, "non-empty"),
    ({"labels": ("casual", " casual ")}, "unique"),
    ({"labels": (": everyday clothing", "formal")}, "non-empty candidate label"),
    ({"labels": ("casual: ", "formal")}, "candidate label description"),
    ({"labels": ("casual: jeans and a T-shirt", "casual: everyday clothing")}, "unique"),
    ({"class_ids": ()}, "class ID"),
    ({"confidence": 1.1}, "between 0 and 1"),
    ({"padding": -0.1}, "finite and non-negative"),
    ({"padding": np.nan}, "finite and non-negative"),
    ({"max_tokens": 0}, "positive integer"),
    ({"max_tokens": 1.5}, "positive integer"),
    ({"max_tokens": True}, "positive integer"),
])
def test_invalid_options_fail_before_loading_models(models, options, message):
    kwargs = {"attribute": "style", "labels": LABELS, **options}
    with pytest.raises(ValueError, match=message):
        models.example.build_pipeline(**kwargs)
    models.construct.assert_not_called()
    models.qwen_factory.assert_not_called()


@pytest.mark.parametrize("image", [
    np.zeros((0, 10, 3), dtype=np.uint8), np.zeros((10, 10), dtype=np.uint8),
    np.zeros((10, 10, 4), dtype=np.uint8), np.zeros((10, 10, 3), dtype=np.float32),
])
def test_invalid_images_fail_before_loading_clip(models, image):
    with pytest.raises(ValueError, match="BGR HWC uint8"):
        models.example.CLIPClassify("style", LABELS)(image)
    models.construct.assert_not_called()


@pytest.mark.parametrize("stage, embeddings, message", [
    ("embed_text", np.ones((2, 3)), "one non-empty embedding"),
    ("embed_text", np.ones((3,)), "one non-empty embedding"),
    ("embed_text", np.ones((3, 0)), "one non-empty embedding"),
    ("embed_text", np.zeros((3, 3)), "finite and non-zero"),
    ("embed_text", np.full((3, 3), np.nan), "finite and non-zero"),
    ("embed_image", np.ones((2, 3)), "one non-empty embedding"),
    ("embed_image", np.full((1, 3), np.inf), "finite and non-zero"),
    ("embed_image", np.zeros((1, 3)), "finite and non-zero"),
    ("embed_image", np.ones((1, 4)), "same dimensions"),
])
def test_invalid_embeddings_are_rejected_instead_of_producing_a_label(models, stage, embeddings, message):
    getattr(models.clip, stage).return_value = embeddings
    with pytest.raises(ValueError, match=message):
        models.example.CLIPClassify("style", LABELS)(np.zeros((10, 10, 3), dtype=np.uint8))


def test_clip_classifications_use_public_data_attachment_without_mutation(models):
    detections = sv.Detections(
        xyxy=np.asarray([[0, 0, 10, 10]], dtype=float),
        class_id=np.array([1]), confidence=np.array([0.9]),
        data={"class_name": np.array(["person"])},
    )
    classifications = [models.example.Classification("casual", 0.3)]
    classified = Detections.AttachData()(detections, {
        "attribute": ["style"],
        "label": [result.label for result in classifications],
        "similarity": [result.similarity for result in classifications],
    })
    assert list(classified.data["class_name"]) == ["person"]
    assert list(classified.data["attribute"]) == ["style"]
    assert list(classified.data["label"]) == ["casual"]
    assert list(classified.data["similarity"]) == [0.3]
    assert classified.class_id.tolist() == [1]
    assert classified.confidence.tolist() == [0.9]
    assert "label" not in detections.data
    with pytest.raises(ValueError, match="exactly one"):
        Detections.AttachDataField("label")(detections, [])


def test_attaching_attribute_labels_preserves_metadata_and_full_text_without_mutating(models):
    detections = sv.Detections(
        xyxy=np.asarray([[0, 0, 10, 10]], dtype=float),
        class_id=np.array([1]), confidence=np.array([0.9]),
        data={"class_name": np.array(["person"])},
    )
    answer = "casual\nEveryday clothing, including a blue"

    responses = [answer]
    fields = models.example.ParseAttributeLabels("style: the outfit's visible style", LABELS)(
        responses,
    )
    result = Detections.AttachData()(detections, fields)

    assert list(result.data["class_name"]) == ["person"]
    assert list(result.data["attribute"]) == ["style"]
    assert list(result.data["label"]) == ["casual"]
    assert list(result.data["response"]) == [answer]
    assert result.class_id.tolist() == [1]
    assert result.confidence.tolist() == [0.9]
    np.testing.assert_array_equal(result.xyxy, detections.xyxy)
    assert "label" not in detections.data and "response" not in detections.data
    assert fields["response"] is responses


@pytest.mark.parametrize("answers", [[], ["casual", "formal"]])
def test_attribute_labels_require_aligned_responses(models, answers):
    detections = sv.Detections(xyxy=np.asarray([[0, 0, 10, 10]], dtype=float))
    fields = models.example.ParseAttributeLabels("style", LABELS)(answers)

    with pytest.raises(ValueError, match="exactly one"):
        Detections.AttachData()(detections, fields)


@pytest.mark.parametrize("responses, labels", [
    ([], []),
    (["casual", " formal \r\nOffice attire."], ["casual", "formal"]),
    (["", "   ", "for", "Formal", "formal: office attire"], ["unknown"] * 5),
    (["sportswear\nA partial explanation", "unknown\nNot visible."], ["sportswear", "unknown"]),
])
def test_attribute_parser_is_detection_independent_and_retains_raw_responses(models, responses, labels):
    original = responses.copy()
    fields = models.example.ParseAttributeLabels("style: visible outfit style", LABELS)(responses)

    assert fields == {
        "attribute": ["style"] * len(responses),
        "label": labels,
        "response": responses,
    }
    assert fields["response"] is responses
    assert responses == original


@pytest.mark.parametrize("answer, expected_label", [
    ("casual", "casual"),
    (" formal \r\nOffice attire.", "formal"),
    ("sportswear\nThe person wears a wind", "sportswear"),  # Partial explanation is retained.
    ("for", "unknown"),  # An incomplete label must not be guessed.
    ("streetwear\nA new label.", "unknown"),
    ("formal: office attire", "unknown"),
    ("Formal", "unknown"),
    ("unknown\nToo blurry to tell.", "unknown"),
])
def test_qwen_labels_are_validated_and_only_short_names_are_annotated(
    models, source_image, monkeypatch, answer, expected_label,
):
    models.generate.return_value = answer
    pipeline = models.example.build_pipeline(attribute="style", labels=LABELS)
    annotator = next(op for op in pipeline.operators if isinstance(op, LabelAnnotator))
    label_spy = MagicMock(wraps=annotator.annotator.annotate)
    monkeypatch.setattr(annotator.annotator, "annotate", label_spy)

    _, detections = pipeline(source_image[0])

    assert list(detections.data["label"]) == [expected_label]
    assert list(detections.data["response"]) == [answer]
    assert label_spy.call_args.kwargs["labels"] == [expected_label]
    models.clip.embed_image.assert_not_called()


def test_builder_defaults_to_described_fashion_style(models):
    pipeline = models.example.build_pipeline()
    operator = next(op for op in pipeline.operators if isinstance(op, LocalQwen3VL))

    assert operator.prompt == models.example.build_attribute_prompt(
        models.example.DEFAULT_ATTRIBUTE, models.example.DEFAULT_LABELS,
    )
    assert operator.model_id == "Qwen/Qwen3-VL-2B-Instruct"
    assert operator.device == "cpu" and operator.max_tokens == 256
    models.construct.assert_called_once_with(model_id=models.example.DEFAULT_DETECTION_MODEL_ID)
    assert operator._model is None


def test_cli_missing_qwen_prints_install_message_before_download_or_model_loading(models, monkeypatch, capsys):
    monkeypatch.setattr(models.example, "require_local_qwen", require_local_qwen)
    monkeypatch.setitem(sys.modules, "torch", None)
    download = MagicMock()
    monkeypatch.setattr(models.example, "download_assets", download)
    monkeypatch.setattr("sys.argv", ["run_detect_and_attribute.py"])

    with pytest.raises(SystemExit) as error:
        models.example.main()

    assert error.value.code == 1
    output = capsys.readouterr()
    assert INSTALL_COMMAND in output.err and "Traceback" not in output.err
    assert output.out == ""
    download.assert_not_called()
    models.construct.assert_not_called()
    models.qwen_factory.assert_not_called()
    models.generate.assert_not_called()


def test_cli_help_works_without_qwen(models, monkeypatch, capsys):
    monkeypatch.setattr(models.example, "require_local_qwen", require_local_qwen)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr("sys.argv", ["run_detect_and_attribute.py", "--help"])

    with pytest.raises(SystemExit) as error:
        models.example.main()

    assert error.value.code == 0
    assert "--qwen-model-id" in capsys.readouterr().out
    models.construct.assert_not_called()
    models.qwen_factory.assert_not_called()


def test_cli_forwards_local_qwen_options(models, source_image, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", [
        "run_detect_and_attribute.py", "--input", str(source_image[0]),
        "--qwen-model-id", "/models/qwen", "--device", "cuda:1", "--max-tokens", "80",
        "--api-key", "test-key",
    ])

    assert models.example.main() == 0

    models.qwen_factory.assert_called_once_with(
        prompt=models.example.build_attribute_prompt(models.example.DEFAULT_ATTRIBUTE, models.example.DEFAULT_LABELS),
        model_id="/models/qwen", device="cuda:1", max_tokens=80,
    )
    models.construct.assert_called_once_with(model_id=models.example.DEFAULT_DETECTION_MODEL_ID, api_key="test-key")
    image = models.generate.call_args.args[0]
    assert image.color_space == "RGB" and image.array.shape == (44, 22, 3)
    assert json.loads(capsys.readouterr().out)[0]["label"] == "formal"
    models.clip.embed_text.assert_not_called()
    models.clip.embed_image.assert_not_called()


@pytest.mark.parametrize("removed_option, value", [
    ("--clip-model-id", "clip/ViT-B-32"), ("--crop-size", "32"),
])
def test_cli_rejects_removed_options_before_loading_any_models(models, monkeypatch, removed_option, value):
    monkeypatch.setattr("sys.argv", ["run_detect_and_attribute.py", removed_option, value])

    with pytest.raises(SystemExit) as error:
        models.example.main()

    assert error.value.code == 2
    models.construct.assert_not_called()
    models.qwen_factory.assert_not_called()


def test_cli_defaults_to_people_walking_and_fashion_style_and_always_displays(
    models, source_image, tmp_path, monkeypatch, capsys,
):
    download = MagicMock(return_value=str(source_image[0]))
    monkeypatch.setattr(models.example, "download_assets", download)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["run_detect_and_attribute.py"])

    assert models.example.main() == 0

    download.assert_called_once_with(sv.assets.ImageAssets.PEOPLE_WALKING, directory=models.example.ASSETS_DIR)
    assert models.example.ASSETS_DIR.is_absolute()
    results = json.loads(capsys.readouterr().out)
    assert results[0]["attribute"] == "fashion style"
    assert results[0]["label"] == "formal"
    models.qwen_factory.assert_called_once_with(
        prompt=models.example.build_attribute_prompt(models.example.DEFAULT_ATTRIBUTE, models.example.DEFAULT_LABELS),
        model_id=models.example.DEFAULT_QWEN_MODEL_ID, device="cpu", max_tokens=256,
    )
    models.plot.assert_called_once()


@pytest.mark.parametrize("attribute, labels", [
    ("fashion style", LABELS), ("activity", ("walking", "standing", "sitting")),
    ("condition", ("new", "worn", "damaged")),
])
def test_cli_saves_classifications_and_annotated_image(
    models, source_image, tmp_path, monkeypatch, capsys, attribute, labels,
):
    download = MagicMock()
    monkeypatch.setattr(models.example, "download_assets", download)
    output = tmp_path / "results.json"
    output_image = tmp_path / "annotated.png"
    monkeypatch.setattr("sys.argv", [
        "run_detect_and_attribute.py", "--input", str(source_image[0]),
        "--attribute", attribute, "--labels", *labels, "--class-ids", "2", "--padding", "0",
        "--output", str(output), "--output-image", str(output_image), "--api-key", "test-key",
    ])
    models.detector.infer.return_value = response([prediction(class_id=2)])
    answer = f"{labels[1]}\nVisible evidence."
    models.generate.return_value = answer

    assert models.example.main() == 0

    download.assert_not_called()
    expected = [{
        "detection_index": 0, "xyxy": [10.0, 20.0, 30.0, 60.0],
        "class_id": 2, "confidence": pytest.approx(0.9),
        "attribute": attribute, "label": labels[1], "response": answer,
    }]
    assert json.loads(capsys.readouterr().out) == expected
    assert json.loads(output.read_text(encoding="utf-8")) == expected
    saved = cv2.imread(str(output_image))
    assert saved.shape == source_image[1].shape
    assert not np.array_equal(saved, source_image[1])
    np.testing.assert_array_equal(models.plot.call_args.args[0], saved)
    assert models.qwen_factory.call_args.kwargs["prompt"] == models.example.build_attribute_prompt(attribute, labels)
    models.construct.assert_called_once_with(model_id=models.example.DEFAULT_DETECTION_MODEL_ID, api_key="test-key")


def test_cli_reports_annotated_image_write_failure(models, source_image, tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "run_detect_and_attribute.py", "--input", str(source_image[0]),
        "--attribute", "style", "--labels", *LABELS,
        "--output-image", str(tmp_path / "annotated.png"),
    ])
    monkeypatch.setattr(cv2, "imwrite", MagicMock(return_value=False))
    with pytest.raises(OSError, match="Failed to save annotated image"):
        models.example.main()


def test_cli_descriptions_are_matching_text_while_json_keeps_short_names(
    models, source_image, tmp_path, monkeypatch, capsys,
):
    output = tmp_path / "results.json"
    monkeypatch.setattr("sys.argv", [
        "run_detect_and_attribute.py", "--input", str(source_image[0]),
        "--attribute", "fashion style: overall style of the visible outfit",
        "--labels", "casual: everyday clothing", "formal: office attire: a suit or tie",
        "sportswear: athletic clothing, including shorts and windbreakers",
        "--output", str(output),
    ])

    assert models.example.main() == 0

    results = json.loads(capsys.readouterr().out)
    assert results[0]["attribute"] == "fashion style"
    assert results[0]["label"] == "formal"
    assert json.loads(output.read_text(encoding="utf-8")) == results
    prompt = models.qwen_factory.call_args.kwargs["prompt"]
    assert "overall style of the visible outfit" in prompt
    assert "everyday clothing" in prompt
    assert "office attire: a suit or tie" in prompt
    assert "athletic clothing, including shorts and windbreakers" in prompt
