"""Track video objects and ask Qwen3-VL about each confirmed track once.

Run from the repository root:
    python examples/run_track_and_describe.py
    python examples/run_track_and_describe.py --input video.mp4
    python examples/run_track_and_describe.py --device cpu
    python examples/run_track_and_describe.py --input video.mp4 --prompt "Describe the outfit." --output-video annotated.mp4 --output tracks.json

Build a fresh frame pipeline for each video. Responses, including 'unknown',
are cached by tracker ID for that pipeline's lifetime, not by person identity.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import supervision as sv
from supervision.assets import VideoAssets, download_assets

if __package__:
    from . import run_detect_and_describe as describe_image
    from .common import ASSETS_DIR
    from .qwen_local import LocalQwen3VL, require_local_qwen
else:
    import run_detect_and_describe as describe_image
    from common import ASSETS_DIR
    from qwen_local import LocalQwen3VL, require_local_qwen

from ml_pipes.core import Pipeline
from ml_pipes.operator import Operator
from ml_pipes.standard import Gather, Pick, Recall, Scatter, Select, Store
from ml_pipes.supervision import (
    BoxAnnotator,
    Detections,
    ImageWindow,
    LabelAnnotator,
)
from ml_pipes.supervision.inference import RoboflowInference
from ml_pipes.supervision.trackers import ByteTrack
from ml_pipes.vision import ConvertColorSpace

DEFAULT_VIDEO_ASSET = VideoAssets.PEOPLE_WALKING


def confirmed_tracks(detections: sv.Detections) -> npt.NDArray[np.bool_]:
    """Pending IDs (-1) and detections without IDs must never reach the VLM."""
    if detections.tracker_id is None:
        return np.zeros(len(detections), dtype=bool)
    return detections.tracker_id >= 0


@Operator
class TrackAttributes:
    """Store and restore selected detections.data fields by confirmed tracker ID.

    Each configured default represents a missing value. Neither defaults nor
    None overwrite cached attributes. Other supplied values update the cache.
    Reuse the same instance before and after enriching detections; it never
    filters detections or replaces their current boxes and confidence scores.
    """

    def __init__(self, defaults: dict[str, Any]) -> None:
        self.defaults = deepcopy(defaults)
        self.attributes: dict[int, dict[str, Any]] = {}

    def __call__(self, detections: sv.Detections) -> sv.Detections:
        current = {
            key: detections.data.get(key, [default] * len(detections))
            for key, default in self.defaults.items()
        }
        ids = detections.tracker_id
        if ids is not None:
            for index, track_id in enumerate(ids):
                if track_id < 0:
                    continue
                for key, values in current.items():
                    value = values[index]
                    if value is not None and not np.array_equal(value, self.defaults[key]):
                        self.attributes.setdefault(int(track_id), {})[key] = deepcopy(value)

        restored = {
            key: np.asarray([
                deepcopy(self.attributes.get(int(ids[index]), {}).get(key, value))
                if ids is not None and ids[index] >= 0 else deepcopy(value)
                for index, value in enumerate(values)
            ], dtype=object)
            for key, values in current.items()
        }
        return replace(detections, data={**detections.data, **restored})


def build_frame_pipeline(
    *,
    detection_model_id: str = describe_image.DEFAULT_DETECTION_MODEL_ID,
    class_ids: tuple[int, ...] = describe_image.DEFAULT_CLASS_IDS,
    confidence: float = 0.5,
    padding: float = 0.05,
    prompt: str = describe_image.DEFAULT_PROMPT,
    qwen_model_id: str = describe_image.DEFAULT_QWEN_MODEL_ID,
    device: str = "cpu",
    max_tokens: int = 256,
    api_key: str | None = None,
    frame_rate: float = 30.0,
) -> Pipeline[npt.NDArray[np.uint8], tuple[npt.NDArray[np.uint8], sv.Detections]]:
    """Track every frame; describe only uncached IDs; annotate every current track."""
    if not class_ids:
        raise ValueError("At least one class ID is required.")
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1.")
    if not np.isfinite(padding) or padding < 0:
        raise ValueError("padding must be finite and non-negative.")
    if not np.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError("frame_rate must be finite and positive.")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer.")
    prepared_prompt = describe_image.build_prompt(prompt)
    track_attributes = TrackAttributes({"response": ""})
    return Pipeline(
        [
            Store("source_frame"),
            RoboflowInference(model_id=detection_model_id, api_key=api_key, confidence=confidence),
            Select(0),
            Detections.FromInference(),
            Detections.Filter(lambda detections: np.isin(detections.class_id, class_ids)),
            Detections.Filter(lambda detections: detections.confidence >= confidence),
            Detections.NMS(threshold=0.5),
            ByteTrack(
                frame_rate=frame_rate,
                track_activation_threshold=confidence,
                high_conf_det_threshold=confidence,
                minimum_consecutive_frames=2,
            ),
            Detections.Filter(confirmed_tracks),
            track_attributes,
            Store("tracked_detections"),
            Detections.Filter(lambda detections: (
                (detections.data["response"] == "")
                & np.isin(
                    np.arange(len(detections)),
                    np.unique(detections.tracker_id, return_index=True)[1],
                )
            )),
            Detections.ScaleBoxes(1 + 2 * padding),
            Recall("source_frame", prepend=True),
            Detections.ClipBoxes(rounding="outward", invalid="drop"),
            Detections.Crop(),
            Store("new_detections", source=0),
            Pick(1),
            Scatter(max_concurrency=1),
            ConvertColorSpace(output_color_space="RGB"),
            LocalQwen3VL(
                prompt=prepared_prompt, model_id=qwen_model_id,
                device=device, max_tokens=max_tokens,
            ),
            Gather(),
            Recall("new_detections", prepend=True),
            Detections.AttachDataField("response", overwrite=True),  # Replace the pending-response placeholder.
            track_attributes,
            Recall("tracked_detections"),
            Pick(1),
            track_attributes,
            Recall("source_frame", prepend=True),
            BoxAnnotator(color_lookup=sv.ColorLookup.TRACK),
            LabelAnnotator(
                color_lookup=sv.ColorLookup.TRACK,
                label_formatter=lambda detection: (
                    f"#{detection.tracker_id} "
                    + (
                        detection.data["response"].splitlines()[0]
                        if detection.data["response"] else "pending"
                    )
                ),
            ),
            ImageWindow("Track and Describe", at=0),
        ],
        auto_validate=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Source video path. Defaults to Supervision's PEOPLE_WALKING sample (downloaded if missing).",
    )
    parser.add_argument("--output", type=Path, help="Optional per-track JSON result path.")
    parser.add_argument("--output-video", type=Path, help="Optional annotated video path (no audio).")
    parser.add_argument("--detection-model-id", default=describe_image.DEFAULT_DETECTION_MODEL_ID)
    parser.add_argument("--class-ids", type=int, nargs="+", default=describe_image.DEFAULT_CLASS_IDS)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--padding", type=float, default=0.05)
    parser.add_argument(
        "--qwen-model-id", default=describe_image.DEFAULT_QWEN_MODEL_ID,
        help=f"Local model ID or path (default: {describe_image.DEFAULT_QWEN_MODEL_ID}).",
    )
    parser.add_argument("--device", default="cpu", help="Local Qwen device: cpu (default), mps, cuda, or cuda:N.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum generated answer tokens (default: 256).")
    parser.add_argument("--api-key", help="Optional Roboflow key for detection; ROBOFLOW_API_KEY also works.")
    parser.add_argument("--prompt", default=describe_image.DEFAULT_PROMPT)
    args = parser.parse_args()
    require_local_qwen()
    input_path = args.input or Path(download_assets(DEFAULT_VIDEO_ASSET, directory=ASSETS_DIR))
    if not input_path.is_file():
        parser.error(f"input video not found: {input_path}")
    outputs = [path.resolve() for path in (args.output, args.output_video) if path is not None]
    if input_path.resolve() in outputs or len(outputs) != len(set(outputs)):
        parser.error("input, JSON output, and video output must use different paths")

    video_info = sv.VideoInfo.from_video_path(str(input_path))
    pipeline = build_frame_pipeline(
        detection_model_id=args.detection_model_id,
        class_ids=tuple(args.class_ids),
        confidence=args.confidence,
        padding=args.padding,
        prompt=args.prompt,
        qwen_model_id=args.qwen_model_id,
        device=args.device,
        max_tokens=args.max_tokens,
        api_key=args.api_key,
        frame_rate=video_info.fps,
    )
    results: dict[int, str] = {}
    with ExitStack() as stack:
        window = next(op.window for op in pipeline.operators if isinstance(op, ImageWindow))
        stack.enter_context(window)
        sink = (
            stack.enter_context(sv.VideoSink(str(args.output_video), video_info))
            if args.output_video is not None else None
        )
        frames = sv.get_video_frames_generator(str(input_path))
        stack.callback(frames.close)
        for frame in frames:
            annotated, detections = pipeline(frame)
            if sink is not None:
                sink.write_frame(annotated)
            if detections.tracker_id is not None:
                results.update(
                    (int(track_id), response)
                    for track_id, response in zip(detections.tracker_id, detections.data["response"])
                    if response
                )
            if window.wait_key(1) in ("q", "Escape") or not window.is_open:
                break

    output = json.dumps(
        [{"tracker_id": track_id, "response": response} for track_id, response in sorted(results.items())],
        indent=2, ensure_ascii=False,
    )
    print(output)
    if args.output is not None:
        args.output.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
