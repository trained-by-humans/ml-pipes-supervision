"""Two-stage pipeline filtering, crop alignment, annotation, and CLI behavior."""
from importlib import import_module
from importlib.util import find_spec
import json
import sys
from unittest.mock import MagicMock, create_autospec

import cv2
import numpy as np
import pytest
import supervision as sv

from ml_pipes.standard import Scatter
from ml_pipes.supervision import Detections, ImageToArray
from examples.qwen_local import INSTALL_COMMAND, LocalQwen3VL, require_local_qwen
from ml_pipes.vision import ConvertColorSpace, ImagePayload


@pytest.fixture
def plot_image(monkeypatch):
    plot = create_autospec(sv.plot_image)
    monkeypatch.setattr(sv, "plot_image", plot)
    return plot


@pytest.fixture
def example_models(monkeypatch, plot_image):
    if find_spec("inference") is None:
        pytest.skip("Optional Inference dependency is not installed.")
    from ml_pipes.supervision import inference as inference_ops

    example = import_module("examples.run_detect_and_describe")
    detector = create_autospec(inference_ops.Model, instance=True)
    factory = create_autospec(inference_ops.get_model, return_value=detector)
    monkeypatch.setattr(inference_ops, "get_model", factory)
    generate = MagicMock(side_effect=["walking", "standing"])

    def describe(self, image: ImagePayload) -> str:
        return generate(image)

    monkeypatch.setattr(LocalQwen3VL, "__call__", describe)
    qwen_factory = create_autospec(
        LocalQwen3VL,
        side_effect=lambda **kwargs: LocalQwen3VL(**kwargs),
    )
    monkeypatch.setattr(example, "LocalQwen3VL", qwen_factory)
    monkeypatch.setattr(example, "require_local_qwen", MagicMock())
    return example, detector, factory, generate, qwen_factory


@pytest.fixture
def source_image(tmp_path):
    image = np.arange(100 * 120 * 3, dtype=np.uint8).reshape(100, 120, 3)
    path = tmp_path / "people.png"
    assert cv2.imwrite(str(path), image)
    return path, image


def prediction(class_id=1, confidence=0.9, *, x=20, y=40, width=20, height=40):
    return {
        "x": x, "y": y, "width": width, "height": height,
        "class_id": class_id, "class": "person" if class_id == 1 else "bicycle",
        "confidence": confidence,
    }


def response(predictions):
    return [{"image": {"width": 120, "height": 100}, "predictions": predictions}]


def test_only_filtered_person_crops_reach_qwen_and_results_stay_aligned(
    example_models, source_image, plot_image,
):
    example, detector, factory, generate, _ = example_models
    path, source = source_image
    detector.infer.return_value = response([
        prediction(),
        prediction(confidence=0.7),  # duplicate: no extra model call
        prediction(class_id=2, x=50),
        prediction(confidence=0.4, x=70),
        prediction(confidence=0.8, x=100),
    ])
    pipeline = example.build_pipeline()
    pipeline.validate()

    annotated, detections = pipeline(path)

    factory.assert_called_once_with(model_id=example.DEFAULT_DETECTION_MODEL_ID)
    assert detector.infer.call_args.kwargs == {"confidence": 0.5}
    np.testing.assert_array_equal(detector.infer.call_args.args[0], source)
    assert detections.class_id.tolist() == [1, 1]
    assert list(detections.data["response"]) == ["walking", "standing"]
    assert annotated.shape == source.shape
    assert annotated.dtype == source.dtype
    assert not np.array_equal(annotated, source)
    plot_image.assert_called_once()
    assert plot_image.call_args.args[0] is annotated
    assert generate.call_count == 2
    expected_crops = [source[18:62, 9:31], source[18:62, 89:111]]
    for call, expected in zip(generate.call_args_list, expected_crops):
        payload = call.args[0]
        assert payload.color_space == "RGB" and payload.layout == "HWC"
        crop = payload.array
        np.testing.assert_array_equal(crop, expected[..., ::-1])


@pytest.mark.parametrize("predictions", [[], [prediction(class_id=2)], [prediction(confidence=0.1)]])
def test_empty_filtered_results_skip_local_model_initialization(
    example_models, source_image, predictions, monkeypatch, plot_image,
):
    example, detector, _, generate, _ = example_models
    detector.infer.return_value = response(predictions)
    pipeline = example.build_pipeline(prompt="style")
    qwen = next(op for op in pipeline.operators if isinstance(op, LocalQwen3VL))

    annotated, detections = pipeline(source_image[0])

    assert len(detections) == 0
    assert list(detections.data["response"]) == []
    np.testing.assert_array_equal(annotated, source_image[1])
    plot_image.assert_called_once()
    assert plot_image.call_args.args[0] is annotated
    assert qwen._model is qwen._processor is None
    generate.assert_not_called()


def test_alternate_class_filter_and_results_are_fresh_on_next_image(
    example_models, source_image, plot_image,
):
    example, detector, _, generate, _ = example_models
    pipeline = example.build_pipeline(class_ids=(2,), confidence=0.8)
    detector.infer.return_value = response([prediction(), prediction(class_id=2, confidence=0.8)])

    _, detections = pipeline(source_image[0])
    assert detections.class_id.tolist() == [2]
    assert list(detections.data["response"]) == ["walking"]
    detector.infer.return_value = response([])
    annotated, detections = pipeline(source_image[0])
    assert len(detections) == 0
    assert list(detections.data["response"]) == []
    np.testing.assert_array_equal(annotated, source_image[1])
    assert generate.call_count == 1
    assert plot_image.call_count == 2
    assert plot_image.call_args.args[0] is annotated


def test_annotations_use_only_the_first_response_line_and_preserve_full_text(
    example_models, source_image, monkeypatch,
):
    example, detector, _, generate, _ = example_models
    descriptions = ["casual\nBlue jeans and a white shirt.", "formal\r\nA dark suit."]
    generate.side_effect = descriptions
    detector.infer.return_value = response([prediction(), prediction(x=100, confidence=0.8)])
    pipeline = example.build_pipeline()
    box = next(op for op in pipeline.operators if isinstance(op, example.BoxAnnotator))
    label = next(op for op in pipeline.operators if isinstance(op, example.LabelAnnotator))
    box_spy = MagicMock(wraps=box.annotator.annotate)
    label_spy = MagicMock(wraps=label.annotator.annotate)
    monkeypatch.setattr(box.annotator, "annotate", box_spy)
    monkeypatch.setattr(label.annotator, "annotate", label_spy)

    annotated, detections = pipeline(source_image[0])

    box_spy.assert_called_once()
    assert box_spy.call_args.kwargs["detections"] is detections
    label_spy.assert_called_once()
    assert label_spy.call_args.kwargs["detections"] is detections
    assert label_spy.call_args.kwargs["labels"] == ["casual", "formal"]
    assert list(detections.data["response"]) == descriptions
    np.testing.assert_array_equal(detections.xyxy, [[9, 18, 31, 62], [89, 18, 111, 62]])
    np.testing.assert_array_equal(detector.infer.call_args.args[0], source_image[1])
    assert not np.shares_memory(annotated, detector.infer.call_args.args[0])
    assert not np.array_equal(annotated, source_image[1])
    assert generate.call_count == 2


def test_attaching_responses_preserves_existing_detection_data_without_mutating_it(example_models):
    example, *_ = example_models
    detections = sv.Detections(
        xyxy=np.asarray([[0, 0, 10, 10]], dtype=np.float32),
        data={"class_name": np.asarray(["person"])},
    )

    annotated = Detections.AttachDataField("response")(detections, ["walking\nOne foot raised."])

    assert list(annotated.data["class_name"]) == ["person"]
    assert list(annotated.data["response"]) == ["walking\nOne foot raised."]
    assert "response" not in detections.data


@pytest.mark.parametrize("responses", [[], ["walking", "standing"]])
def test_attaching_responses_rejects_misaligned_results(example_models, responses):
    example, *_ = example_models
    detections = sv.Detections(xyxy=np.asarray([[0, 0, 10, 10]], dtype=np.float32))

    with pytest.raises(ValueError, match="exactly one"):
        Detections.AttachDataField("response")(detections, responses)


def test_builder_constructs_qwen_inline_from_configuration(example_models):
    example, _, _, _, constructor = example_models

    pipeline = example.build_pipeline(
        prompt="Describe the colors.",
        qwen_model_id="custom-qwen-model",
        device="mps",
    )

    constructor.assert_called_once_with(
        prompt=example.build_prompt("Describe the colors."),
        model_id="custom-qwen-model",
        device="mps",
        max_tokens=256,
    )
    qwen = next(op for op in pipeline.operators if isinstance(op, LocalQwen3VL))
    assert qwen.prompt == example.build_prompt("Describe the colors.")
    assert qwen.model_id == "custom-qwen-model"
    assert qwen.device == "mps"


@pytest.mark.parametrize("api_key", [None, "test-key"])
def test_detector_credentials_are_forwarded_and_crops_keep_source_resolution(
    example_models, source_image, api_key,
):
    example, detector, detector_factory, generate, qwen_factory = example_models
    detector.infer.return_value = response([prediction()])
    pipeline = example.build_pipeline(
        detection_model_id="custom-detector", api_key=api_key,
        confidence=0.7, padding=0,
    )
    pipeline.validate()

    _, detections = pipeline(source_image[0])

    expected = {"model_id": "custom-detector"}
    if api_key is not None:
        expected["api_key"] = api_key
    detector_factory.assert_called_once_with(**expected)
    assert detector.infer.call_args.kwargs == {"confidence": 0.7}
    assert "api_key" not in qwen_factory.call_args.kwargs
    crop = generate.call_args.args[0]
    assert crop.color_space == "RGB"
    assert crop.array.shape == (40, 20, 3)
    np.testing.assert_array_equal(crop.array, source_image[1][20:60, 10:30, ::-1])
    np.testing.assert_array_equal(detections.xyxy, [[10, 20, 30, 60]])


def test_attribute_and_description_keep_matching_pipeline_stages(example_models):
    example, *_ = example_models
    classify = import_module("examples.run_detect_and_attribute")
    classification_steps = classify.build_pipeline().operators
    description_steps = example.build_pipeline().operators

    assert len(classification_steps) == len(description_steps) + 1
    parser = next(op for op in classification_steps if isinstance(op, classify.ParseAttributes))
    common_steps = [op for op in classification_steps if op is not parser]
    differences = [
        (type(attribute), type(description))
        for attribute, description in zip(common_steps, description_steps)
        if type(attribute) is not type(description)
    ]
    assert differences == [(Detections.AttachData, Detections.AttachDataField)]
    attachment = next(op for op in classification_steps if isinstance(op, Detections.AttachData))
    assert attachment.overwrite is False
    attachment_index = classification_steps.index(attachment)
    assert classification_steps.index(parser) == attachment_index - 2
    assert classification_steps[attachment_index - 1].name == "detections"
    field = next(op for op in description_steps if isinstance(op, Detections.AttachDataField))
    assert field.key == "response" and field.overwrite is False


def test_task_prompt_has_shared_grounding_and_annotation_format(example_models):
    example, *_ = example_models

    prompt = example.build_prompt("  Describe the visible activity.\n ")

    assert prompt.startswith("Task:\nDescribe the visible activity.\n\n")
    assert prompt.endswith(example.RESPONSE_INSTRUCTIONS)
    assert "first line" in prompt
    assert "'unknown'" in prompt
    assert "visible evidence" in prompt
    assert "hidden personal attributes" in prompt
    assert "fashion" not in example.DEFAULT_PROMPT.lower()


@pytest.mark.parametrize("options, message", [
    ({"prompt": ""}, "non-empty task prompt"),
    ({"prompt": " \n\t"}, "non-empty task prompt"),
    ({"padding": -0.1}, "finite and non-negative"),
    ({"padding": np.nan}, "finite and non-negative"),
    ({"padding": np.inf}, "finite and non-negative"),
    ({"class_ids": ()}, "At least one class ID"),
    ({"confidence": 1.1}, "between 0 and 1"),
    ({"max_tokens": 0}, "positive integer"),
])
def test_invalid_options_fail_before_loading_models(example_models, options, message):
    example, _, detector_factory, _, qwen_factory = example_models

    with pytest.raises(ValueError, match=message):
        example.build_pipeline(**options)

    detector_factory.assert_not_called()
    qwen_factory.assert_not_called()


def test_scattered_crops_go_directly_through_rgb_conversion_to_qwen(example_models):
    example, *_ = example_models
    pipeline = example.build_pipeline()
    operators = pipeline.operators
    index = next(i for i, op in enumerate(operators) if isinstance(op, Scatter))
    assert isinstance(operators[index + 1], ConvertColorSpace)
    assert operators[index + 1].output_color_space == "RGB"
    assert isinstance(operators[index + 2], LocalQwen3VL)
    # ImageToArray is used only before detection, not after RGB conversion.
    assert sum(isinstance(op, ImageToArray) for op in operators) == 1


@pytest.mark.parametrize("padding", [0, 0.05, 0.2])
def test_crop_preparation_uses_public_operators_and_stores_surviving_roi_boxes(example_models, padding):
    example, *_ = example_models
    pipeline = example.build_pipeline(padding=padding)
    operators = pipeline.operators
    index = next(i for i, op in enumerate(operators) if isinstance(op, Detections.ScaleBoxes))
    assert operators[index].factor_x == operators[index].factor_y == 1 + 2 * padding
    prepared = operators[index](sv.Detections(xyxy=np.asarray([[0., 0., 20., 40.]])))
    np.testing.assert_allclose(
        prepared.xyxy,
        [[-20 * padding, -40 * padding, 20 + 20 * padding, 40 + 40 * padding]],
    )
    # Recall supplies the original image only after padding has prepared boxes.
    assert operators[index + 1].name == "source_image"
    assert isinstance(operators[index + 2], Detections.ClipBoxes)
    assert operators[index + 2].rounding == "outward"
    assert operators[index + 2].invalid == "drop"
    assert isinstance(operators[index + 3], Detections.Crop)
    assert operators[index + 4].name == "detections"


@pytest.mark.parametrize("shape", [(2048, 1024, 3), (1024, 2048, 3)])
def test_large_crops_keep_source_resolution_before_qwen(
    example_models, tmp_path, shape,
):
    example, detector, _, generate, _ = example_models
    image = np.full(shape, [10, 20, 230], dtype=np.uint8)
    image_path = tmp_path / "large.png"
    assert cv2.imwrite(str(image_path), image)
    height, width = shape[:2]
    detector.infer.return_value = [{
        "image": {"width": width, "height": height},
        "predictions": [prediction(x=width / 2, y=height / 2, width=width, height=height)],
    }]

    _, detections = example.build_pipeline()(image_path)

    assert list(detections.data["response"]) == ["walking"]
    np.testing.assert_array_equal(detections.xyxy, [[0, 0, width, height]])
    prepared = generate.call_args.args[0].array
    assert prepared.shape == shape
    np.testing.assert_array_equal(prepared, image[..., ::-1])


@pytest.mark.parametrize("padding, expected", [
    (0.05, [[-1, -2, 21, 42]]),
    (0, [[0, 0, 20, 40]]),
])
def test_padding_preserves_detections_and_does_not_clip(example_models, padding, expected):
    example, *_ = example_models
    detections = sv.Detections(xyxy=np.asarray([[0, 0, 20, 40]], dtype=np.int32))

    prepared = Detections.ScaleBoxes(1 + 2 * padding)(detections)

    assert prepared is not detections
    np.testing.assert_array_equal(prepared.xyxy, expected)
    np.testing.assert_array_equal(detections.xyxy, [[0, 0, 20, 40]])
    assert not np.shares_memory(prepared.xyxy, detections.xyxy)
    assert "source_xyxy" not in detections.data


def test_clipping_works_without_padding_and_keeps_metadata_aligned(example_models):
    example, *_ = example_models
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    detections = sv.Detections(
        xyxy=np.asarray([
            [-5, -5, 10, 10], [20.25, 10.5, 40, 25], [31, 0, 40, 10],
            [0, 0, 0, 10], [10, 0, 5, 10], [5, 5, np.nan, 10],
            [-np.inf, 0, 10, 10],
        ], dtype=np.float64),
        class_id=np.arange(7),
        confidence=np.linspace(0.1, 0.7, 7),
        data={"name": np.asarray(["first", "second", "outside", "empty", "reversed", "nan", "inf"])},
    )
    original_boxes = detections.xyxy.copy()

    preserved_image, kept = Detections.ClipBoxes(invalid="drop")(image, detections)

    assert preserved_image is image
    assert kept.class_id.tolist() == [0, 1]
    assert list(kept.data["name"]) == ["first", "second"]
    np.testing.assert_array_equal(kept.confidence, detections.confidence[:2])
    np.testing.assert_array_equal(kept.xyxy, [[0, 0, 10, 10], [20.25, 10.5, 30, 20]])
    np.testing.assert_array_equal(detections.xyxy, original_boxes)
    assert not np.shares_memory(kept.xyxy, detections.xyxy)


def test_cropping_slices_only_prepared_boxes_without_adding_padding(example_models):
    example, *_ = example_models
    image = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
    detections = sv.Detections(xyxy=np.asarray([[2.25, 3.5, 10.75, 12.25]], dtype=np.float64))
    clipped_image, clipped = Detections.ClipBoxes(rounding="outward", invalid="drop")(image, detections)
    kept, crops = Detections.Crop()(clipped_image, clipped)

    assert kept is clipped
    np.testing.assert_array_equal(kept.xyxy, [[2, 3, 11, 13]])
    assert len(crops) == 1
    np.testing.assert_array_equal(crops[0].array, image[3:13, 2:11])
    assert not np.shares_memory(crops[0].array, image)
    np.testing.assert_array_equal(detections.xyxy, [[2.25, 3.5, 10.75, 12.25]])


def test_pad_clip_crop_preserves_alignment_and_returns_prepared_detection_boxes(example_models):
    example, *_ = example_models
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    detections = sv.Detections(
        xyxy=np.asarray([
            [-5, -5, 10, 10], [0, 0, 0, 10], [31, 0, 40, 10],
            [5, 5, np.nan, 10], [10, 10, 40, 25],
        ], dtype=np.float32),
        class_id=np.arange(5),
    )

    prepared = Detections.ScaleBoxes(1.1)(detections)
    clipped_image, clipped = Detections.ClipBoxes(rounding="outward", invalid="drop")(image, prepared)
    result, crops = Detections.Crop()(clipped_image, clipped)

    assert result.class_id.tolist() == [0, 4]
    assert [crop.shape for crop in crops] == [(11, 11, 3), (11, 22, 3)]
    assert all(isinstance(crop, ImagePayload) for crop in crops)
    assert all(crop.color_space == "BGR" and crop.layout == "HWC" for crop in crops)
    assert all(not np.shares_memory(crop.array, image) for crop in crops)
    np.testing.assert_array_equal(result.xyxy, [[0, 0, 11, 11], [8, 9, 30, 20]])
    assert result.xyxy.dtype == detections.xyxy.dtype
    assert "source_xyxy" not in result.data


def test_fully_outside_crop_boxes_skip_qwen_and_annotations(
    example_models, source_image, plot_image,
):
    example, detector, _, generate, _ = example_models
    detector.infer.return_value = response([prediction(x=200)])

    annotated, detections = example.build_pipeline()(source_image[0])

    assert len(detections) == 0
    assert list(detections.data["response"]) == []
    np.testing.assert_array_equal(annotated, source_image[1])
    generate.assert_not_called()
    plot_image.assert_called_once()


def test_invalid_regions_are_dropped_and_responses_annotate_surviving_roi_boxes(
    example_models, source_image,
):
    example, detector, _, generate, _ = example_models
    detector.infer.return_value = response([
        prediction(x=200, confidence=0.99), prediction(), prediction(x=116, confidence=0.8),
    ])

    _, detections = example.build_pipeline()(source_image[0])

    np.testing.assert_array_equal(detections.xyxy, [[9, 18, 31, 62], [105, 18, 120, 62]])
    assert list(detections.data["response"]) == ["walking", "standing"]
    assert "source_xyxy" not in detections.data
    assert generate.call_count == 2
    for call, expected in zip(generate.call_args_list, (
        source_image[1][18:62, 9:31], source_image[1][18:62, 105:120],
    )):
        np.testing.assert_array_equal(call.args[0].array, expected[..., ::-1])


def test_cli_defaults_to_people_walking_without_input(
    example_models, source_image, tmp_path, monkeypatch, capsys, plot_image,
):
    example, detector, _, generate, _ = example_models
    detector.infer.return_value = response([prediction()])
    download = MagicMock(return_value=str(source_image[0]))
    monkeypatch.setattr(example, "download_assets", download)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["run_detect_and_describe.py"])

    assert example.main() == 0

    download.assert_called_once_with(sv.assets.ImageAssets.PEOPLE_WALKING, directory=example.ASSETS_DIR)
    assert example.ASSETS_DIR.is_absolute()
    results = json.loads(capsys.readouterr().out)
    assert len(results) == 1
    assert results[0]["response"] == "walking"
    np.testing.assert_array_equal(detector.infer.call_args.args[0], source_image[1])
    generate.assert_called_once()
    plot_image.assert_called_once()


def test_cli_missing_qwen_prints_install_message_before_download_or_model_loading(
    example_models, monkeypatch, capsys,
):
    example, _, detector_factory, generate, qwen_factory = example_models
    monkeypatch.setattr(example, "require_local_qwen", require_local_qwen)
    monkeypatch.setitem(sys.modules, "torch", None)
    download = MagicMock()
    monkeypatch.setattr(example, "download_assets", download)
    monkeypatch.setattr("sys.argv", ["run_detect_and_describe.py"])

    with pytest.raises(SystemExit) as error:
        example.main()

    assert error.value.code == 1
    output = capsys.readouterr()
    assert INSTALL_COMMAND in output.err and "Traceback" not in output.err
    assert output.out == ""
    download.assert_not_called()
    detector_factory.assert_not_called()
    qwen_factory.assert_not_called()
    generate.assert_not_called()


def test_cli_help_works_without_qwen(example_models, monkeypatch, capsys):
    example, _, detector_factory, _, qwen_factory = example_models
    monkeypatch.setattr(example, "require_local_qwen", require_local_qwen)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr("sys.argv", ["run_detect_and_describe.py", "--help"])

    with pytest.raises(SystemExit) as error:
        example.main()

    assert error.value.code == 0
    assert "--qwen-model-id" in capsys.readouterr().out
    detector_factory.assert_not_called()
    qwen_factory.assert_not_called()


@pytest.mark.parametrize("task_prompt, answer, class_id", [
    (None, "walking\nOne foot is raised.", 1),
    ("Describe the fashion style.", "casual\nBlue jeans.", 1),
    ("Describe any visible occupational role.", "unknown\nNo visible work context.", 1),
    ("Is the person a tourist or resident?", "unknown\nResidency is not visible.", 1),
    ("Describe the bicycle's condition.", "worn\nVisible rust on the frame.", 2),
])
def test_cli_prints_and_saves_boxes_with_task_responses(
    example_models, source_image, tmp_path, monkeypatch, capsys, plot_image,
    task_prompt, answer, class_id,
):
    example, detector, _, generate, constructor = example_models
    download = MagicMock()
    monkeypatch.setattr(example, "download_assets", download)
    detector.infer.return_value = response([prediction(class_id=class_id)])
    generate.side_effect = [answer]
    output = tmp_path / "results.json"
    output_image = tmp_path / "annotated.png"
    arguments = [
        "run_detect_and_describe.py", "--input", str(source_image[0]),
        "--output", str(output),
        "--output-image", str(output_image), "--class-ids", str(class_id), "--padding", "0",
    ]
    if task_prompt is not None:
        arguments.extend(["--prompt", task_prompt])
    monkeypatch.setattr("sys.argv", arguments)

    assert example.main() == 0

    download.assert_not_called()
    constructor.assert_called_once_with(
        prompt=example.build_prompt(task_prompt or example.DEFAULT_PROMPT),
        model_id="Qwen/Qwen3-VL-2B-Instruct", device="cpu", max_tokens=256,
    )
    crop = generate.call_args.args[0].array
    # --padding 0 reaches crop preparation, not just argument parsing.
    np.testing.assert_array_equal(crop, source_image[1][20:60, 10:30, ::-1])
    result = json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == result
    assert result == [{
        "detection_index": 0, "xyxy": [10.0, 20.0, 30.0, 60.0],
        "class_id": class_id, "confidence": pytest.approx(0.9), "response": answer,
    }]
    annotated = cv2.imread(str(output_image))
    assert annotated.shape == source_image[1].shape
    assert not np.array_equal(annotated, source_image[1])
    plot_image.assert_called_once()
    np.testing.assert_array_equal(plot_image.call_args.args[0], annotated)


def test_cli_empty_detections_prints_only_json(
    example_models, source_image, monkeypatch, capsys,
):
    example, detector, _, generate, _ = example_models
    detector.infer.return_value = response([])
    monkeypatch.setattr("sys.argv", [
        "run_detect_and_describe.py", "--input", str(source_image[0]),
    ])

    assert example.main() == 0

    assert json.loads(capsys.readouterr().out) == []
    generate.assert_not_called()


def test_cli_reports_annotated_image_write_failure(
    example_models, source_image, tmp_path, monkeypatch,
):
    example, detector, *_ = example_models
    detector.infer.return_value = response([])
    monkeypatch.setattr("sys.argv", [
        "run_detect_and_describe.py", "--input", str(source_image[0]),
        "--output-image", str(tmp_path / "annotated.png"),
    ])
    monkeypatch.setattr(cv2, "imwrite", MagicMock(return_value=False))

    with pytest.raises(OSError, match="Failed to save annotated image"):
        example.main()


def test_local_branch_explicitly_converts_crops_to_rgb_and_annotates_roi_boxes(
    example_models, source_image, monkeypatch, plot_image,
):
    example, detector, _, generate, constructor = example_models
    detector.infer.return_value = response([
        prediction(), prediction(class_id=2, x=50), prediction(x=100, confidence=0.8),
    ])
    calls = []
    answers = ["casual\nBlue jeans.", "formal\nA dark suit."]

    def describe(self, image: ImagePayload) -> str:
        calls.append((self, image))
        return answers[len(calls) - 1]

    monkeypatch.setattr(LocalQwen3VL, "__call__", describe)
    pipeline = example.build_pipeline(device="cpu", max_tokens=100)
    pipeline.validate()
    ops = pipeline.operators
    index = next(i for i, op in enumerate(ops) if isinstance(op, Scatter))
    assert isinstance(ops[index + 1], ConvertColorSpace)
    assert ops[index + 1].output_color_space == "RGB"
    assert isinstance(ops[index + 2], LocalQwen3VL)
    label = next(op for op in ops if isinstance(op, example.LabelAnnotator))
    label_spy = MagicMock(wraps=label.annotator.annotate)
    monkeypatch.setattr(label.annotator, "annotate", label_spy)

    _, detections = pipeline(source_image[0])

    assert list(detections.data["response"]) == answers
    assert label_spy.call_args.kwargs["labels"] == ["casual", "formal"]
    np.testing.assert_array_equal(detections.xyxy, [[9, 18, 31, 62], [89, 18, 111, 62]])
    assert len(calls) == 2
    assert calls[0][0] is calls[1][0]
    operator = calls[0][0]
    assert operator.model_id == example.DEFAULT_QWEN_MODEL_ID
    assert operator.prompt == example.build_prompt(example.DEFAULT_PROMPT)
    assert operator.device == "cpu" and operator.max_tokens == 100
    for (_, crop), expected in zip(calls, (source_image[1][18:62, 9:31], source_image[1][18:62, 89:111])):
        assert crop.color_space == "RGB" and crop.layout == "HWC"
        np.testing.assert_array_equal(crop.array, expected[..., ::-1])
    # Detection still receives the original BGR pixels; only local Qwen is imported.
    np.testing.assert_array_equal(detector.infer.call_args.args[0], source_image[1])
    constructor.assert_called_once()
    assert not hasattr(example, "Qwen3VL")
    generate.assert_not_called()
    plot_image.assert_called_once()


def test_empty_local_pipeline_does_not_load_qwen(example_models, source_image):
    example, detector, _, _, constructor = example_models
    detector.infer.return_value = response([])
    pipeline = example.build_pipeline()
    qwen = next(op for op in pipeline.operators if isinstance(op, LocalQwen3VL))

    _, detections = pipeline(source_image[0])

    assert len(detections) == 0
    assert qwen._model is qwen._processor is None
    constructor.assert_called_once()


@pytest.mark.parametrize("override", [[], ["--qwen-model-id", "/models/qwen", "--device", "mps"]])
def test_cli_uses_local_qwen_with_default_or_custom_model(
    example_models, source_image, monkeypatch, capsys, override,
):
    example, detector, detector_factory, _, constructor = example_models
    detector.infer.return_value = response([prediction()])
    calls = []

    def describe(self, image: ImagePayload) -> str:
        calls.append(self)
        assert image.color_space == "RGB"
        assert image.array.shape == (44, 22, 3)
        return "casual\nBlue jeans."

    monkeypatch.setattr(LocalQwen3VL, "__call__", describe)
    monkeypatch.setattr("sys.argv", [
        "run_detect_and_describe.py", "--input", str(source_image[0]),
        "--max-tokens", "80", "--api-key", "test-key", *override,
    ])

    assert example.main() == 0

    assert calls[0].model_id == ("/models/qwen" if override else example.DEFAULT_QWEN_MODEL_ID)
    assert calls[0].device == ("mps" if override else "cpu")
    assert calls[0].max_tokens == 80
    detector_factory.assert_called_once_with(model_id=example.DEFAULT_DETECTION_MODEL_ID, api_key="test-key")
    assert json.loads(capsys.readouterr().out)[0]["response"] == "casual\nBlue jeans."
    constructor.assert_called_once()


@pytest.mark.parametrize("removed_option, value", [
    ("--qwen-backend", "bedrock"), ("--region", "us-east-1"), ("--crop-size", "32"),
])
def test_cli_rejects_removed_options_before_loading_models(example_models, monkeypatch, removed_option, value):
    example, _, detector_factory, _, constructor = example_models
    monkeypatch.setattr("sys.argv", ["run_detect_and_describe.py", removed_option, value])

    with pytest.raises(SystemExit) as error:
        example.main()

    assert error.value.code == 2
    detector_factory.assert_not_called()
    constructor.assert_not_called()
