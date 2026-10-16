"""Two-stage pipeline filtering, crop alignment, annotation, and CLI behavior."""
from importlib import import_module
from importlib.util import find_spec
import json
from unittest.mock import MagicMock, create_autospec

import cv2
import numpy as np
import pytest
import supervision as sv

from ml_pipes.standard import Pick, Scatter
from ml_pipes.supervision import ImageToArray
from ml_pipes.supervision.bedrock import Qwen3VL
from ml_pipes.vision import ImagePayload, Resize


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
    client = MagicMock(spec=["converse"])
    client.converse.side_effect = [
        {"output": {"message": {"content": [{"text": label}]}}, "stopReason": "end_turn"}
        for label in ("walking", "standing")
    ]
    qwen_factory = create_autospec(
        Qwen3VL,
        side_effect=lambda **kwargs: Qwen3VL(**kwargs, client=client),
    )
    monkeypatch.setattr(example, "Qwen3VL", qwen_factory)
    return example, detector, factory, client, qwen_factory


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
    example, detector, factory, client, _ = example_models
    path, source = source_image
    detector.infer.return_value = response([
        prediction(),
        prediction(confidence=0.7),  # duplicate: no extra Bedrock request
        prediction(class_id=2, x=50),
        prediction(confidence=0.4, x=70),
        prediction(confidence=0.8, x=100),
    ])
    pipeline = example.build_pipeline()
    pipeline.validate()

    annotated, detections = pipeline(path)

    factory.assert_called_once_with(model_id="rfdetr-small")
    assert detector.infer.call_args.kwargs == {"confidence": 0.5}
    np.testing.assert_array_equal(detector.infer.call_args.args[0], source)
    assert detections.class_id.tolist() == [1, 1]
    assert detections.data["response"].tolist() == ["walking", "standing"]
    assert annotated.shape == source.shape
    assert annotated.dtype == source.dtype
    assert not np.array_equal(annotated, source)
    plot_image.assert_called_once()
    assert plot_image.call_args.args[0] is annotated
    assert client.converse.call_count == 2
    expected_crops = [source[18:62, 9:31], source[18:62, 89:111]]
    for call, expected in zip(client.converse.call_args_list, expected_crops):
        content = call.kwargs["messages"][0]["content"]
        encoded = content[0]["image"]["source"]["bytes"]
        crop = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert crop.shape == (1024, 1024, 3)
        height, width = expected.shape[:2]
        top, left = (1024 - height) // 2, (1024 - width) // 2
        np.testing.assert_array_equal(crop[top:top + height, left:left + width], expected)
        np.testing.assert_array_equal(crop[0, 0], [114, 114, 114])


@pytest.mark.parametrize("predictions", [[], [prediction(class_id=2)], [prediction(confidence=0.1)]])
def test_empty_filtered_results_skip_bedrock_initialization(
    example_models, source_image, predictions, monkeypatch, plot_image,
):
    example, detector, _, client, _ = example_models
    detector.infer.return_value = response(predictions)
    monkeypatch.setattr(example, "Qwen3VL", Qwen3VL)
    pipeline = example.build_pipeline(prompt="style")
    qwen = next(op for op in pipeline.operators if isinstance(op, Qwen3VL))

    annotated, detections = pipeline(source_image[0])

    assert len(detections) == 0
    assert detections.data["response"].tolist() == []
    np.testing.assert_array_equal(annotated, source_image[1])
    plot_image.assert_called_once()
    assert plot_image.call_args.args[0] is annotated
    assert qwen._client is None
    client.converse.assert_not_called()


def test_alternate_class_filter_and_results_are_fresh_on_next_image(
    example_models, source_image, plot_image,
):
    example, detector, _, client, _ = example_models
    pipeline = example.build_pipeline(class_ids=(2,), confidence=0.8)
    detector.infer.return_value = response([prediction(), prediction(class_id=2, confidence=0.8)])

    _, detections = pipeline(source_image[0])
    assert detections.class_id.tolist() == [2]
    assert detections.data["response"].tolist() == ["walking"]
    detector.infer.return_value = response([])
    annotated, detections = pipeline(source_image[0])
    assert len(detections) == 0
    assert detections.data["response"].tolist() == []
    np.testing.assert_array_equal(annotated, source_image[1])
    assert client.converse.call_count == 1
    assert plot_image.call_count == 2
    assert plot_image.call_args.args[0] is annotated


def test_annotations_use_only_the_first_response_line_and_preserve_full_text(
    example_models, source_image, monkeypatch,
):
    example, detector, _, client, _ = example_models
    descriptions = ["casual\nBlue jeans and a white shirt.", "formal\r\nA dark suit."]
    client.converse.side_effect = [
        {"output": {"message": {"content": [{"text": description}]}}, "stopReason": "end_turn"}
        for description in descriptions
    ]
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
    assert detections.data["response"].tolist() == descriptions
    np.testing.assert_array_equal(detections.xyxy, [[10, 20, 30, 60], [90, 20, 110, 60]])
    np.testing.assert_array_equal(detector.infer.call_args.args[0], source_image[1])
    assert not np.shares_memory(annotated, detector.infer.call_args.args[0])
    assert not np.array_equal(annotated, source_image[1])
    assert client.converse.call_count == 2


def test_attaching_responses_preserves_existing_detection_data_without_mutating_it(example_models):
    example, *_ = example_models
    detections = sv.Detections(
        xyxy=np.asarray([[0, 0, 10, 10]], dtype=np.float32),
        data={"class_name": np.asarray(["person"])},
    )

    annotated = example.AttachResponses()(detections, ["walking\nOne foot raised."])

    assert annotated.data["class_name"].tolist() == ["person"]
    assert annotated.data["response"].tolist() == ["walking\nOne foot raised."]
    assert "response" not in detections.data


@pytest.mark.parametrize("responses", [[], ["walking", "standing"]])
def test_attaching_responses_rejects_misaligned_results(example_models, responses):
    example, *_ = example_models
    detections = sv.Detections(xyxy=np.asarray([[0, 0, 10, 10]], dtype=np.float32))

    with pytest.raises(ValueError, match="exactly one"):
        example.AttachResponses()(detections, responses)


def test_builder_constructs_qwen_inline_from_configuration(example_models):
    example, _, _, _, constructor = example_models

    pipeline = example.build_pipeline(
        prompt="Describe the colors.",
        qwen_model_id="custom-qwen-model",
        region_name="us-west-2",
    )

    constructor.assert_called_once_with(
        prompt=example.build_prompt("Describe the colors."),
        model_id="custom-qwen-model",
        region_name="us-west-2",
    )
    qwen = next(op for op in pipeline.operators if isinstance(op, Qwen3VL))
    assert qwen.prompt == example.build_prompt("Describe the colors.")
    assert qwen.model_id == "custom-qwen-model"
    assert qwen.region_name == "us-west-2"


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
    ({"class_ids": ()}, "At least one class ID"),
    ({"confidence": 1.1}, "between 0 and 1"),
])
def test_invalid_options_fail_before_loading_models(example_models, options, message):
    example, _, detector_factory, _, qwen_factory = example_models

    with pytest.raises(ValueError, match=message):
        example.build_pipeline(**options)

    detector_factory.assert_not_called()
    qwen_factory.assert_not_called()


def test_resize_is_an_explicit_step_between_scatter_and_qwen(example_models):
    example, *_ = example_models
    pipeline = example.build_pipeline()
    operators = pipeline.operators
    index = next(i for i, op in enumerate(operators) if isinstance(op, Resize))
    resize = operators[index]

    assert isinstance(operators[index - 1], Scatter)
    assert resize.size == (1024, 1024)
    assert resize.mode == "letterbox"
    assert resize.allow_scale_up is False
    assert resize.interpolation == "area"
    assert isinstance(operators[index + 1], Pick)
    assert operators[index + 1].indices == (0,)
    assert isinstance(operators[index + 2], ImageToArray)
    assert isinstance(operators[index + 3], Qwen3VL)


@pytest.mark.parametrize("padding", [0, 0.05, 0.2])
def test_crop_preparation_is_three_explicit_steps(example_models, padding):
    example, *_ = example_models
    pipeline = example.build_pipeline(padding=padding)
    operators = pipeline.operators
    index = next(i for i, op in enumerate(operators) if isinstance(op, example.PadBoxes))

    assert operators[index].padding == padding
    # Recall supplies the original image only after padding has prepared boxes.
    assert operators[index + 1].name == "source_image"
    assert isinstance(operators[index + 2], example.ClipBoxes)
    assert isinstance(operators[index + 3], example.CropDetections)


@pytest.mark.parametrize("shape, content_slice", [
    ((2048, 1024, 3), (slice(None), slice(256, 768))),
    ((1024, 2048, 3), (slice(256, 768), slice(None))),
])
def test_large_crops_are_letterboxed_before_bedrock_without_changing_source_boxes(
    example_models, tmp_path, shape, content_slice,
):
    example, detector, _, client, _ = example_models
    image = np.full(shape, [10, 20, 230], dtype=np.uint8)
    image_path = tmp_path / "large.png"
    assert cv2.imwrite(str(image_path), image)
    height, width = shape[:2]
    detector.infer.return_value = [{
        "image": {"width": width, "height": height},
        "predictions": [prediction(x=width / 2, y=height / 2, width=width, height=height)],
    }]

    _, detections = example.build_pipeline()(image_path)

    assert detections.data["response"].tolist() == ["walking"]
    np.testing.assert_array_equal(detections.xyxy, [[0, 0, width, height]])
    content = client.converse.call_args.kwargs["messages"][0]["content"]
    encoded = content[0]["image"]["source"]["bytes"]
    prepared = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert prepared.shape == (1024, 1024, 3)
    assert np.all(prepared[content_slice] == [10, 20, 230])
    np.testing.assert_array_equal(prepared[0, 0], [114, 114, 114])


@pytest.mark.parametrize("padding, expected", [
    (0.05, [[-1, -2, 21, 42]]),
    (0, [[0, 0, 20, 40]]),
])
def test_padding_preserves_detections_and_does_not_clip(example_models, padding, expected):
    example, *_ = example_models
    detections = sv.Detections(xyxy=np.asarray([[0, 0, 20, 40]], dtype=np.int32))

    preserved, crop_boxes = example.PadBoxes(padding=padding)(detections)

    assert preserved is detections
    np.testing.assert_array_equal(crop_boxes, expected)
    np.testing.assert_array_equal(detections.xyxy, [[0, 0, 20, 40]])
    assert not np.shares_memory(crop_boxes, detections.xyxy)


@pytest.mark.parametrize("padding", [-0.1, np.nan, np.inf, -np.inf])
def test_padding_requires_a_finite_non_negative_fraction(example_models, padding):
    example, *_ = example_models

    with pytest.raises(ValueError, match="finite and non-negative"):
        example.PadBoxes(padding=padding)


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

    preserved_image, kept, crop_boxes = example.ClipBoxes()(image, detections, detections.xyxy)

    assert preserved_image is image
    assert kept.class_id.tolist() == [0, 1]
    assert kept.data["name"].tolist() == ["first", "second"]
    np.testing.assert_array_equal(kept.confidence, detections.confidence[:2])
    np.testing.assert_array_equal(crop_boxes, [[0, 0, 10, 10], [20.25, 10.5, 30, 20]])
    np.testing.assert_array_equal(kept.xyxy, original_boxes[:2])
    np.testing.assert_array_equal(detections.xyxy, original_boxes)
    assert not np.shares_memory(crop_boxes, detections.xyxy)


def test_cropping_slices_only_prepared_boxes_without_adding_padding(example_models):
    example, *_ = example_models
    image = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
    detections = sv.Detections(xyxy=np.asarray([[8, 8, 12, 12]], dtype=np.float64))
    crop_boxes = np.asarray([[2.25, 3.5, 10.75, 12.25]], dtype=np.float64)

    kept, crops = example.CropDetections()(image, detections, crop_boxes)

    assert kept is detections
    assert len(crops) == 1
    np.testing.assert_array_equal(crops[0].array, image[3:13, 2:11])
    assert not np.shares_memory(crops[0].array, image)
    np.testing.assert_array_equal(crop_boxes, [[2.25, 3.5, 10.75, 12.25]])


@pytest.mark.parametrize("operator_name", ["ClipBoxes", "CropDetections"])
def test_crop_boxes_must_stay_aligned_with_detections(example_models, operator_name):
    example, *_ = example_models
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    detections = sv.Detections(xyxy=np.asarray([[0, 0, 10, 10]], dtype=np.float64))

    with pytest.raises(ValueError, match="exactly one crop box"):
        getattr(example, operator_name)()(image, detections, np.empty((0, 4)))


def test_pad_clip_crop_preserves_alignment_and_original_detection_boxes(example_models):
    example, *_ = example_models
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    detections = sv.Detections(
        xyxy=np.asarray([
            [-5, -5, 10, 10], [0, 0, 0, 10], [31, 0, 40, 10],
            [5, 5, np.nan, 10], [10, 10, 40, 25],
        ], dtype=np.float32),
        class_id=np.arange(5),
    )

    preserved, padded_boxes = example.PadBoxes()(detections)
    clipped_image, kept, clipped_boxes = example.ClipBoxes()(image, preserved, padded_boxes)
    result, crops = example.CropDetections()(clipped_image, kept, clipped_boxes)

    assert result is kept
    assert kept.class_id.tolist() == [0, 4]
    assert [crop.shape for crop in crops] == [(11, 11, 3), (11, 22, 3)]
    assert all(isinstance(crop, ImagePayload) for crop in crops)
    assert all(crop.color_space == "BGR" and crop.layout == "HWC" for crop in crops)
    assert all(not np.shares_memory(crop.array, image) for crop in crops)
    np.testing.assert_array_equal(kept.xyxy, detections.xyxy[[0, 4]])


def test_fully_outside_crop_boxes_skip_bedrock_and_annotations(
    example_models, source_image, plot_image,
):
    example, detector, _, client, _ = example_models
    detector.infer.return_value = response([prediction(x=200)])

    annotated, detections = example.build_pipeline()(source_image[0])

    assert len(detections) == 0
    assert detections.data["response"].tolist() == []
    np.testing.assert_array_equal(annotated, source_image[1])
    client.converse.assert_not_called()
    plot_image.assert_called_once()


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
    example, detector, _, client, constructor = example_models
    detector.infer.return_value = response([prediction(class_id=class_id)])
    client.converse.side_effect = [{
        "output": {"message": {"content": [{"text": answer}]}},
        "stopReason": "end_turn",
    }]
    output = tmp_path / "results.json"
    output_image = tmp_path / "annotated.png"
    arguments = [
        "run_detect_and_describe.py", "--input", str(source_image[0]),
        "--output", str(output), "--region", "us-west-2",
        "--output-image", str(output_image), "--class-ids", str(class_id), "--padding", "0",
    ]
    if task_prompt is not None:
        arguments.extend(["--prompt", task_prompt])
    monkeypatch.setattr("sys.argv", arguments)

    assert example.main() == 0

    constructor.assert_called_once_with(
        prompt=example.build_prompt(task_prompt or example.DEFAULT_PROMPT),
        model_id="qwen.qwen3-vl-235b-a22b", region_name="us-west-2",
    )
    content = client.converse.call_args.kwargs["messages"][0]["content"]
    assert content[1]["text"] == example.build_prompt(task_prompt or example.DEFAULT_PROMPT)
    encoded = content[0]["image"]["source"]["bytes"]
    crop = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    # --padding 0 reaches crop preparation, not just argument parsing.
    np.testing.assert_array_equal(crop[492:532, 502:522], source_image[1][20:60, 10:30])
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
