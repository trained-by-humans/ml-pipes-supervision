"""Tracked video descriptions are requested once per ID and reused on every frame."""
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

from ml_pipes.supervision import LabelAnnotator
from ml_pipes.supervision import trackers as tracker_ops
from ml_pipes.standard import Pick, Scatter
from examples.qwen_local import INSTALL_COMMAND, LocalQwen3VL, require_local_qwen
from ml_pipes.vision import ConvertColorSpace, ImagePayload


def tracked(ids, boxes=None):
    count = 2 if ids is None else len(ids)
    if boxes is None:
        boxes = [[10 + index * 40, 20, 30 + index * 40, 60] for index in range(count)]
    return sv.Detections(
        xyxy=np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
        class_id=np.ones(count, dtype=int),
        confidence=np.full(count, 0.9),
        tracker_id=None if ids is None else np.asarray(ids, dtype=int),
    )


def prediction(class_id=1, confidence=0.9, x=20):
    return {
        "x": x, "y": 40, "width": 20, "height": 40,
        "class_id": class_id, "class": "person" if class_id == 1 else "bicycle",
        "confidence": confidence,
    }


@pytest.fixture
def models(monkeypatch):
    if find_spec("inference") is None:
        pytest.skip("Optional Inference dependency is not installed.")
    from ml_pipes.supervision import inference as inference_ops

    example = import_module("examples.run_track_and_describe")
    detector = create_autospec(inference_ops.Model, instance=True)
    detector.infer.return_value = [{
        "image": {"width": 120, "height": 100}, "predictions": [prediction()],
    }]
    detector_factory = create_autospec(inference_ops.get_model, return_value=detector)
    monkeypatch.setattr(inference_ops, "get_model", detector_factory)
    tracker_class = tracker_ops.ByteTrackTracker
    tracker = create_autospec(tracker_class, instance=True)
    tracker_factory = create_autospec(tracker_class, return_value=tracker)
    monkeypatch.setattr(tracker_ops, "ByteTrackTracker", tracker_factory)
    generate = MagicMock(side_effect=[
        "walking\nOne foot raised.", "standing\nBoth feet planted.", "unknown\nToo blurry to judge.",
    ])

    def describe(self, image: ImagePayload) -> str:
        return generate(image)

    monkeypatch.setattr(LocalQwen3VL, "__call__", describe)
    qwen_factory = create_autospec(
        LocalQwen3VL, side_effect=lambda **kwargs: LocalQwen3VL(**kwargs),
    )
    monkeypatch.setattr(example, "LocalQwen3VL", qwen_factory)
    monkeypatch.setattr(example, "require_local_qwen", MagicMock())
    show = MagicMock()
    close = MagicMock()
    monkeypatch.setattr(sv.ImageWindow, "show", show)
    monkeypatch.setattr(sv.ImageWindow, "close", close)
    monkeypatch.setattr(sv.ImageWindow, "is_open", property(lambda self: True))
    return SimpleNamespace(
        example=example, detector=detector, detector_factory=detector_factory,
        tracker=tracker, tracker_factory=tracker_factory, tracker_class=tracker_class,
        generate=generate, qwen_factory=qwen_factory, show=show, close=close,
    )


@pytest.fixture
def frame():
    return np.arange(100 * 120 * 3, dtype=np.uint8).reshape(100, 120, 3)


def test_each_id_is_described_once_and_all_current_tracks_keep_their_labels(models, frame, monkeypatch):
    models.tracker.update.side_effect = [
        tracked([7, 9]), tracked([9, 7]), tracked([9, 7, 11]), tracked([]), tracked([7, 11]),
    ]
    pipeline = models.example.build_frame_pipeline()
    pipeline.validate()
    label = next(op for op in pipeline.operators if isinstance(op, LabelAnnotator))
    label_spy = MagicMock(wraps=label.annotator.annotate)
    monkeypatch.setattr(label.annotator, "annotate", label_spy)

    outputs = [pipeline(frame) for _ in range(5)]

    assert models.generate.call_count == 3
    assert models.detector.infer.call_count == models.tracker.update.call_count == 5
    assert models.show.call_count == 5
    assert [d.tracker_id.tolist() for _, d in outputs] == [[7, 9], [9, 7], [9, 7, 11], [], [7, 11]]
    assert outputs[0][1].data["response"].tolist() == [
        "walking\nOne foot raised.", "standing\nBoth feet planted.",
    ]
    assert outputs[1][1].data["response"].tolist() == [
        "standing\nBoth feet planted.", "walking\nOne foot raised.",
    ]
    assert outputs[4][1].data["response"].tolist() == [
        "walking\nOne foot raised.", "unknown\nToo blurry to judge.",
    ]
    assert label_spy.call_args.kwargs["labels"] == ["#7 walking", "#11 unknown"]
    assert outputs[0][0].shape == frame.shape
    assert not np.array_equal(outputs[0][0], frame)
    np.testing.assert_array_equal(outputs[3][0], frame)
    np.testing.assert_array_equal(models.detector.infer.call_args.args[0], frame)
    # Small crops retain their natural size; only original, unannotated pixels reach Qwen.
    payload = models.generate.call_args_list[0].args[0]
    assert payload.color_space == "RGB" and payload.layout == "HWC"
    crop = payload.array
    np.testing.assert_array_equal(crop, frame[18:62, 9:31, ::-1])


@pytest.mark.parametrize("ids, expected", [
    ([-1, 0, 7], [0, 7]), ([-1, -1], []), (None, []), ([], []),
])
def test_pending_or_missing_ids_never_reach_qwen(models, frame, ids, expected):
    models.tracker.update.return_value = tracked(ids)

    _, detections = models.example.build_frame_pipeline()(frame)

    assert detections.tracker_id is None or detections.tracker_id.tolist() == expected
    assert len(detections) == len(expected)
    assert models.generate.call_count == len(expected)


@pytest.mark.parametrize("box, expected_shape", [
    ([10, 20, 30, 60], (16, 8, 3)),
    ([10, 20, 50, 40], (8, 16, 3)),
])
def test_oversized_track_crops_are_downscaled_without_borders_and_cached(
    models, frame, box, expected_shape,
):
    models.tracker.update.return_value = tracked([7], [box])
    pipeline = models.example.build_frame_pipeline(crop_size=16, padding=0)
    pipeline.validate()
    ops = pipeline.operators
    index = next(i for i, op in enumerate(ops) if isinstance(op, models.example.LimitCropSize))
    assert ops[index].max_size == 16
    assert isinstance(ops[index - 1], Scatter)
    assert isinstance(ops[index + 1], Pick) and ops[index + 1].indices == (0,)
    assert isinstance(ops[index + 2], ConvertColorSpace)
    assert isinstance(ops[index + 3], LocalQwen3VL)

    _, first = pipeline(frame)
    _, repeated = pipeline(frame)

    models.generate.assert_called_once()
    crop = models.generate.call_args.args[0]
    assert crop.color_space == "RGB" and crop.layout == "HWC"
    assert crop.array.shape == expected_shape
    left, top, right, bottom = box
    expected = cv2.resize(
        frame[top:bottom, left:right], expected_shape[1::-1], interpolation=cv2.INTER_AREA,
    )
    np.testing.assert_array_equal(crop.array, expected[..., ::-1])
    for detections in (first, repeated):
        np.testing.assert_array_equal(detections.xyxy, [box])
        assert detections.tracker_id.tolist() == [7]
        assert detections.data["response"].tolist() == ["walking\nOne foot raised."]
    np.testing.assert_array_equal(models.detector.infer.call_args.args[0], frame)


def test_no_describable_tracks_leaves_local_model_uninitialized(models, frame):
    models.tracker.update.return_value = tracked([])
    pipeline = models.example.build_frame_pipeline()
    qwen = next(op for op in pipeline.operators if isinstance(op, LocalQwen3VL))

    _, detections = pipeline(frame)

    assert detections.data["response"].tolist() == []
    assert qwen._model is qwen._processor is None
    models.generate.assert_not_called()


def test_invalid_crops_remain_pending_until_a_valid_crop_is_available(models, frame):
    models.tracker.update.side_effect = [tracked([7], [[130, 10, 150, 30]]), tracked([7])]
    pipeline = models.example.build_frame_pipeline()

    _, pending = pipeline(frame)

    assert pending.tracker_id.tolist() == [7]
    assert pending.data["response"].tolist() == [""]
    models.generate.assert_not_called()

    _, described = pipeline(frame)

    assert described.data["response"].tolist() == ["walking\nOne foot raised."]
    models.generate.assert_called_once()


def test_dropped_new_crop_does_not_shift_responses_onto_other_tracks(models, frame):
    models.tracker.update.side_effect = [
        tracked([7]),
        tracked([7, 9, 11], [[10, 20, 30, 60], [130, 10, 150, 30], [80, 20, 100, 60]]),
        tracked([9]),
    ]
    pipeline = models.example.build_frame_pipeline()

    pipeline(frame)
    _, mixed = pipeline(frame)

    assert mixed.tracker_id.tolist() == [7, 9, 11]
    np.testing.assert_array_equal(mixed.xyxy, [[10, 20, 30, 60], [130, 10, 150, 30], [80, 20, 100, 60]])
    assert mixed.data["response"].tolist() == [
        "walking\nOne foot raised.", "", "standing\nBoth feet planted.",
    ]
    assert models.generate.call_count == 2

    _, retried = pipeline(frame)

    assert retried.tracker_id.tolist() == [9]
    assert retried.data["response"].tolist() == ["unknown\nToo blurry to judge."]
    assert models.generate.call_count == 3


def test_errors_propagate_and_do_not_mark_a_track_as_described(models, frame):
    models.tracker.update.return_value = tracked([7])
    models.generate.side_effect = [RuntimeError("out of memory"), "walking"]
    pipeline = models.example.build_frame_pipeline()

    with pytest.raises(RuntimeError, match="out of memory"):
        pipeline(frame)

    _, detections = pipeline(frame)

    assert detections.data["response"].tolist() == ["walking"]
    assert models.generate.call_count == 2


def test_unknown_is_cached_and_new_pipeline_has_an_independent_cache(models, frame):
    models.tracker.update.return_value = tracked([7])
    models.generate.side_effect = ["unknown\nNot visible.", "standing"]
    pipeline = models.example.build_frame_pipeline()

    pipeline(frame)
    _, repeated = pipeline(frame)
    _, fresh = models.example.build_frame_pipeline()(frame)

    assert repeated.data["response"].tolist() == ["unknown\nNot visible."]
    assert fresh.data["response"].tolist() == ["standing"]
    assert models.generate.call_count == 2


def test_inline_filter_describes_duplicate_ids_only_once(models, frame):
    models.tracker.update.side_effect = [tracked([9]), tracked([7, 7, 9, -1])]
    pipeline = models.example.build_frame_pipeline()

    pipeline(frame)
    _, detections = pipeline(frame)

    assert models.generate.call_count == 2
    assert detections.tracker_id.tolist() == [7, 7, 9]
    assert detections.data["response"].tolist() == [
        "standing\nBoth feet planted.", "standing\nBoth feet planted.",
        "walking\nOne foot raised.",
    ]


def test_track_attributes_restore_multiple_fields_by_id_without_replacing_current_detections(models):
    attributes = models.example.TrackAttributes({"response": "", "score": None, "visible": None})
    described = tracked([0, 7])
    described.data.update({
        "response": np.array(["a longer response\nFull text.", "standing"]),
        "score": np.array([0.0, 0.9]),
        "visible": np.array([False, True]),
    })
    attributes(described)
    current = tracked([7, 9, 0])
    current.data["scene"] = np.array(["street", "street", "street"])

    restored = attributes(current)

    assert restored.tracker_id.tolist() == [7, 9, 0]
    assert restored.data["response"].tolist() == ["standing", "", "a longer response\nFull text."]
    assert restored.data["score"].tolist() == [0.9, None, 0.0]
    assert restored.data["visible"].tolist() == [True, None, False]
    np.testing.assert_array_equal(restored.xyxy, current.xyxy)
    np.testing.assert_array_equal(restored.confidence, current.confidence)
    np.testing.assert_array_equal(restored.class_id, current.class_id)
    np.testing.assert_array_equal(restored.data["scene"], current.data["scene"])
    assert "response" not in current.data


@pytest.mark.parametrize("missing", ["", None])
def test_track_attributes_missing_values_do_not_overwrite_saved_values(models, missing):
    attributes = models.example.TrackAttributes({"response": ""})
    stale_frame = attributes(tracked([7]))
    described = models.example.AttachResponses()(tracked([7]), ["unknown\nToo blurry."])
    attributes(described)
    stale_frame.data["response"][0] = missing

    restored = attributes(stale_frame)

    assert restored.data["response"].tolist() == ["unknown\nToo blurry."]


def test_track_attributes_can_update_one_field_without_losing_another(models):
    attributes = models.example.TrackAttributes({"style": "", "activity": ""})
    original = tracked([7])
    original.data.update({"style": np.array(["casual"]), "activity": np.array(["walking"])})
    attributes(original)
    changed = tracked([7])
    changed.data["activity"] = np.array(["standing"])
    attributes(changed)

    restored = attributes(tracked([7]))

    assert restored.data["style"].tolist() == ["casual"]
    assert restored.data["activity"].tolist() == ["standing"]


@pytest.mark.parametrize("ids", [None, [-1, -1]])
def test_track_attributes_never_store_or_restore_unconfirmed_ids(models, ids):
    attributes = models.example.TrackAttributes({"response": ""})
    described = models.example.AttachResponses()(tracked(ids), ["walking", "standing"])

    unchanged = attributes(described)
    restored = attributes(tracked(ids))

    assert len(unchanged) == 2
    assert unchanged.data["response"].tolist() == ["walking", "standing"]
    assert restored.data["response"].tolist() == ["", ""]
    assert attributes.attributes == {}


def test_track_attributes_survive_empty_frames_and_have_instance_local_state(models):
    attributes = models.example.TrackAttributes({"response": ""})
    attributes(models.example.AttachResponses()(tracked([7]), ["walking"]))

    empty = attributes(tracked([]))
    restored = attributes(tracked([7]))
    separate = models.example.TrackAttributes({"response": ""})(tracked([7]))

    assert empty.data["response"].shape == (0,)
    assert restored.data["response"].tolist() == ["walking"]
    assert separate.data["response"].tolist() == [""]


def test_track_attributes_do_not_alias_mutable_input_or_output_values(models):
    attributes = models.example.TrackAttributes({"embedding": None})
    described = tracked([7])
    described.data["embedding"] = np.array([[1.0, 2.0]])
    initial = attributes(described)
    described.data["embedding"][0, 0] = 99.0
    initial.data["embedding"][0, 1] = 88.0

    restored = attributes(tracked([7]))
    np.testing.assert_array_equal(restored.data["embedding"], [[1.0, 2.0]])
    restored.data["embedding"][0, 0] = 77.0
    np.testing.assert_array_equal(attributes(tracked([7])).data["embedding"], [[1.0, 2.0]])


def test_pipeline_reuses_one_stateful_attribute_operator_and_an_inline_response_filter(models):
    pipeline = models.example.build_frame_pipeline()
    instances = [op for op in pipeline.operators if isinstance(op, models.example.TrackAttributes)]

    assert len(instances) == 3
    assert all(op is instances[0] for op in instances)
    assert instances[0].defaults == {"response": ""}


def test_filters_run_before_tracking_and_configuration_reaches_both_models(models, frame):
    models.detector.infer.return_value[0]["predictions"] = [
        prediction(), prediction(confidence=0.4), prediction(class_id=2, x=70),
        prediction(confidence=0.2, x=100),
    ]
    models.tracker.update.return_value = tracked([])
    pipeline = models.example.build_frame_pipeline(
        prompt="Describe the activity.", detection_model_id="custom-detector",
        confidence=0.4, frame_rate=25.0, qwen_model_id="custom-qwen", device="mps",
        api_key="test-key",
    )

    pipeline(frame)

    models.detector_factory.assert_called_once_with(model_id="custom-detector", api_key="test-key")
    assert models.detector.infer.call_args.kwargs == {"confidence": 0.4}
    assert models.tracker.update.call_args.args[0].class_id.tolist() == [1]
    kwargs = models.tracker_factory.call_args.kwargs
    assert kwargs["frame_rate"] == 25.0
    assert kwargs["track_activation_threshold"] == kwargs["high_conf_det_threshold"] == 0.4
    assert kwargs["minimum_consecutive_frames"] == 2
    models.qwen_factory.assert_called_once_with(
        prompt=models.example.describe_image.build_prompt("Describe the activity."),
        model_id="custom-qwen", device="mps", max_tokens=256,
    )


@pytest.mark.parametrize("options", [
    {"frame_rate": 0}, {"frame_rate": np.nan}, {"frame_rate": np.inf},
    {"class_ids": ()}, {"confidence": 1.1}, {"padding": -0.1}, {"prompt": " "},
    {"max_tokens": 0}, {"padding": np.nan}, {"padding": np.inf},
    {"crop_size": 0}, {"crop_size": -1}, {"crop_size": 1.5}, {"crop_size": True},
])
def test_invalid_options_fail_before_models_are_loaded(models, options):
    with pytest.raises(ValueError):
        models.example.build_frame_pipeline(**options)
    models.detector_factory.assert_not_called()
    models.qwen_factory.assert_not_called()


def test_real_bytetrack_flow_only_describes_after_confirmation(models, frame, monkeypatch):
    monkeypatch.setattr(tracker_ops, "ByteTrackTracker", models.tracker_class)
    pipeline = models.example.build_frame_pipeline()
    tracker = next(op for op in pipeline.operators if isinstance(op, tracker_ops.ByteTrack))
    tracker.reset()

    _, pending = pipeline(frame)
    assert len(pending) == 0
    models.generate.assert_not_called()
    _, confirmed = pipeline(frame)
    _, repeated = pipeline(frame)

    assert len(confirmed) == 1
    np.testing.assert_array_equal(confirmed.tracker_id, repeated.tracker_id)
    assert confirmed.data["response"].tolist() == repeated.data["response"].tolist()
    models.generate.assert_called_once()


@pytest.fixture
def video(tmp_path, frame):
    path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (120, 100))
    assert writer.isOpened()
    try:
        for _ in range(3):
            writer.write(frame)
    finally:
        writer.release()
    return path


@pytest.mark.parametrize("quit_early", [False, True])
def test_cli_saves_a_real_video_and_one_json_entry_per_track(
    models, video, tmp_path, monkeypatch, capsys, quit_early,
):
    models.tracker.update.side_effect = [tracked([7]), tracked([7]), tracked([7, 9])]
    output = tmp_path / "tracks.json"
    output_video = tmp_path / "annotated.mp4"
    if quit_early:
        monkeypatch.setattr(sv.ImageWindow, "wait_key", lambda *_: "q")
    monkeypatch.setattr("sys.argv", [
        "run_track_and_describe.py", "--input", str(video),
        "--output", str(output), "--output-video", str(output_video),
        "--prompt", "Describe the visible activity.", "--padding", "0.2",
    ])

    assert models.example.main() == 0

    expected = [{"tracker_id": 7, "response": "walking\nOne foot raised."}]
    if not quit_early:
        expected.append({"tracker_id": 9, "response": "standing\nBoth feet planted."})
    assert json.loads(capsys.readouterr().out) == expected
    assert json.loads(output.read_text(encoding="utf-8")) == expected
    assert models.generate.call_count == len(expected)
    saved = sv.VideoInfo.from_video_path(str(output_video))
    assert saved.total_frames == (1 if quit_early else 3)
    assert saved.resolution_wh == (120, 100)
    assert saved.fps == pytest.approx(25)
    assert models.show.call_count == saved.total_frames
    models.close.assert_called_once()


def test_cli_displays_without_requiring_output_files(models, video, monkeypatch, capsys):
    models.tracker.update.return_value = tracked([7])
    download = MagicMock()
    monkeypatch.setattr(models.example, "download_assets", download)
    monkeypatch.setattr("sys.argv", ["run_track_and_describe.py", "--input", str(video)])

    assert models.example.main() == 0

    assert json.loads(capsys.readouterr().out) == [
        {"tracker_id": 7, "response": "walking\nOne foot raised."},
    ]
    assert models.show.call_count == 3
    models.generate.assert_called_once()
    download.assert_not_called()


def test_cli_defaults_to_people_walking_without_input(models, video, tmp_path, monkeypatch, capsys):
    models.tracker.update.return_value = tracked([7])
    download = MagicMock(return_value=str(video))
    monkeypatch.setattr(models.example, "download_assets", download)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["run_track_and_describe.py"])

    assert models.example.main() == 0

    download.assert_called_once_with(
        sv.assets.VideoAssets.PEOPLE_WALKING, directory=models.example.ASSETS_DIR,
    )
    assert models.example.ASSETS_DIR.is_absolute()
    assert json.loads(capsys.readouterr().out) == [
        {"tracker_id": 7, "response": "walking\nOne foot raised."},
    ]
    assert models.show.call_count == 3
    models.generate.assert_called_once()


@pytest.mark.parametrize("output_option", ["--output", "--output-video"])
def test_cli_rejects_overwriting_input_before_loading_models(models, video, monkeypatch, output_option):
    original = video.read_bytes()
    monkeypatch.setattr("sys.argv", [
        "run_track_and_describe.py", "--input", str(video), output_option, str(video),
    ])

    with pytest.raises(SystemExit) as error:
        models.example.main()

    assert error.value.code == 2
    assert video.read_bytes() == original
    models.detector_factory.assert_not_called()


def test_local_backend_converts_only_new_track_crops_and_reuses_cached_answers(models, frame, monkeypatch):
    models.tracker.update.side_effect = [tracked([-1, 7]), tracked([7]), tracked([9, 7])]
    calls = []
    answers = ["casual\nBlue jeans.", "unknown\nToo blurry."]

    def describe(self, image: ImagePayload) -> str:
        calls.append((self, image))
        return answers[len(calls) - 1]

    monkeypatch.setattr(LocalQwen3VL, "__call__", describe)
    pipeline = models.example.build_frame_pipeline()
    pipeline.validate()
    assert any(isinstance(op, ConvertColorSpace) for op in pipeline.operators)

    outputs = [pipeline(frame) for _ in range(3)]

    assert len(calls) == 2
    assert calls[0][0] is calls[1][0]
    assert calls[0][0].model_id == models.example.describe_image.DEFAULT_QWEN_MODEL_ID
    assert outputs[0][1].tracker_id.tolist() == [7]
    assert outputs[1][1].data["response"].tolist() == [answers[0]]
    assert outputs[2][1].data["response"].tolist() == [answers[1], answers[0]]
    for _, crop in calls:
        assert crop.color_space == "RGB" and crop.layout == "HWC"
        assert crop.array.shape == (44, 22, 3)
    np.testing.assert_array_equal(calls[0][1].array, frame[18:62, 49:71, ::-1])
    np.testing.assert_array_equal(models.detector.infer.call_args.args[0], frame)
    models.qwen_factory.assert_called_once()
    assert not hasattr(models.example, "Qwen3VL")
    models.generate.assert_not_called()


def test_local_generation_failure_is_not_cached_and_track_can_retry(models, frame, monkeypatch):
    models.tracker.update.return_value = tracked([7])
    answers = iter([RuntimeError("out of memory"), "casual\nBlue jeans."])

    def describe(self, image: ImagePayload) -> str:
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(LocalQwen3VL, "__call__", describe)
    pipeline = models.example.build_frame_pipeline()
    with pytest.raises(RuntimeError, match="out of memory"):
        pipeline(frame)

    _, retried = pipeline(frame)
    _, repeated = pipeline(frame)

    assert retried.data["response"].tolist() == repeated.data["response"].tolist() == ["casual\nBlue jeans."]
    models.qwen_factory.assert_called_once()


@pytest.mark.parametrize("partial", ["casual, wearing a blue", "casual\nThe subject wears a blue"])
def test_partial_responses_are_shown_and_cached_while_new_tracks_continue(
    models, frame, monkeypatch, partial,
):
    models.tracker.update.side_effect = [tracked([7]), tracked([7]), tracked([7, 9])]
    models.generate.side_effect = [partial, "standing\nBoth feet planted."]
    pipeline = models.example.build_frame_pipeline()
    label = next(op for op in pipeline.operators if isinstance(op, LabelAnnotator))
    label_spy = MagicMock(wraps=label.annotator.annotate)
    monkeypatch.setattr(label.annotator, "annotate", label_spy)

    outputs = [pipeline(frame) for _ in range(3)]

    assert outputs[0][1].data["response"].tolist() == [partial]
    assert outputs[1][1].data["response"].tolist() == [partial]
    assert outputs[2][1].data["response"].tolist() == [partial, "standing\nBoth feet planted."]
    assert label_spy.call_args.kwargs["labels"] == [f"#7 {partial.splitlines()[0]}", "#9 standing"]
    assert models.generate.call_count == 2
    assert models.show.call_count == 3


def test_no_confirmed_tracks_leaves_local_model_uninitialized(models, frame):
    models.tracker.update.return_value = tracked([-1])
    pipeline = models.example.build_frame_pipeline()
    qwen = next(op for op in pipeline.operators if isinstance(op, LocalQwen3VL))

    _, detections = pipeline(frame)

    assert len(detections) == 0
    assert qwen._model is qwen._processor is None
    models.qwen_factory.assert_called_once()


def test_cli_missing_qwen_prints_install_message_before_download_or_model_loading(models, monkeypatch, capsys):
    monkeypatch.setattr(models.example, "require_local_qwen", require_local_qwen)
    monkeypatch.setitem(sys.modules, "torch", None)
    download = MagicMock()
    monkeypatch.setattr(models.example, "download_assets", download)
    monkeypatch.setattr("sys.argv", ["run_track_and_describe.py"])

    with pytest.raises(SystemExit) as error:
        models.example.main()

    assert error.value.code == 1
    output = capsys.readouterr()
    assert INSTALL_COMMAND in output.err and "Traceback" not in output.err
    assert output.out == ""
    download.assert_not_called()
    models.detector_factory.assert_not_called()
    models.qwen_factory.assert_not_called()
    models.generate.assert_not_called()


def test_cli_help_works_without_qwen(models, monkeypatch, capsys):
    monkeypatch.setattr(models.example, "require_local_qwen", require_local_qwen)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr("sys.argv", ["run_track_and_describe.py", "--help"])

    with pytest.raises(SystemExit) as error:
        models.example.main()

    assert error.value.code == 0
    assert "--qwen-model-id" in capsys.readouterr().out
    models.detector_factory.assert_not_called()
    models.qwen_factory.assert_not_called()


def test_cli_video_accepts_local_model_options(models, video, monkeypatch, capsys):
    models.tracker.update.return_value = tracked([7])
    calls = []

    def describe(self, image: ImagePayload) -> str:
        calls.append(self)
        assert image.color_space == "RGB"
        assert image.array.shape == (32, 16, 3)
        return "casual\nBlue jeans."

    monkeypatch.setattr(LocalQwen3VL, "__call__", describe)
    monkeypatch.setattr("sys.argv", [
        "run_track_and_describe.py", "--input", str(video),
        "--qwen-model-id", "/models/qwen", "--device", "cuda:1", "--max-tokens", "80",
        "--crop-size", "32", "--api-key", "test-key",
    ])

    assert models.example.main() == 0

    assert len(calls) == 1
    assert calls[0].model_id == "/models/qwen" and calls[0].device == "cuda:1"
    assert calls[0].max_tokens == 80
    models.detector_factory.assert_called_once_with(
        model_id=models.example.describe_image.DEFAULT_DETECTION_MODEL_ID, api_key="test-key",
    )
    assert json.loads(capsys.readouterr().out) == [{"tracker_id": 7, "response": "casual\nBlue jeans."}]
    assert models.show.call_count == 3
    models.qwen_factory.assert_called_once()


@pytest.mark.parametrize("removed_option, value", [("--qwen-backend", "bedrock"), ("--region", "us-east-1")])
def test_cli_rejects_remote_options_before_loading_models(models, monkeypatch, removed_option, value):
    monkeypatch.setattr("sys.argv", ["run_track_and_describe.py", removed_option, value])

    with pytest.raises(SystemExit) as error:
        models.example.main()

    assert error.value.code == 2
    models.detector_factory.assert_not_called()
    models.qwen_factory.assert_not_called()
