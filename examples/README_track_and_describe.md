# Track and describe

Pipeline source: [run_track_and_describe.py](./run_track_and_describe.py)

Run local RF-DETR Small on each video frame, track detections with ByteTrack,
and ask local Qwen3-VL a question about each confirmed track once. Later frames
reuse the stored answer rather than sending the same track to Qwen again.
People are selected by default.

For the detection, crop, and description steps, see the
[detect-and-describe recipe](../docs/recipes/detect_and_describe.md).

## Setup and run

From the repository root:

```bash
python -m pip install -e '.[inference]'
python -m pip install "transformers>=4.57.0,<6" "torch>=2.6" "Pillow>=10"
python examples/run_track_and_describe.py
```

Without `--input`, the example uses Supervision's people-walking video in
`examples/.example_assets`, downloading it if missing. Qwen weights are cached
automatically on first use, and the loaded model is reused across frames.

```bash
python examples/run_track_and_describe.py --input video.mp4 --prompt "Describe the outfit's fashion style." --output-video annotated.mp4 --output tracks.json
python examples/run_track_and_describe.py --help
```

Use `--prompt` to change the question and `--class-ids` to select detector
classes. `--device mps` or `--device cuda` changes Qwen's device only; CPU is
the default. Run `--help` for all options.

## How tracking changes the pipeline

RF-DETR runs on every frame, then ByteTrack associates the filtered detections.
New tracks need two consecutive detections for confirmation; pending IDs (`-1`)
are excluded.

`TrackAttributes({"response": ""})` stores and restores responses by tracker ID.
It does not filter detections. A separate `Detections.Filter` selects confirmed
IDs without a response and removes duplicate IDs before description.

For those tracks, the pipeline pads and clips the crop boxes, crops the frame,
converts the source-resolution crops to RGB, and sends them to Qwen. The model
processor handles resizing, normalization, and patch alignment.
It uses public `Detections.ScaleBoxes(1 + 2 * padding)`,
`Detections.ClipBoxes(rounding="outward", invalid="drop")`, and `Detections.Crop`.
`Store("tracked_detections")` retains original tracked boxes for annotation;
responses from surviving crops are cached by tracker ID before those original
detections are recalled. No coordinate-preservation callbacks are needed.
Unusable crops remain pending and can be described when a valid crop is
available later.

`Detections.AttachDataField("response", overwrite=True)` attaches new answers,
explicitly replacing the empty response placeholders populated by the cache.
Attachment is positional within the surviving crop batch; `TrackAttributes`
handles the separate tracker-ID join when restoring all current tracks.

The same `TrackAttributes` instance runs before filtering, after attaching new
answers, and when restoring all current tracks for annotation. Only successfully
described tracks update the cache. Defaults and `None` never overwrite cached
values; configure other field defaults to reuse this operator for other
attributes.

Cached answers are attached to all current confirmed tracks, including those
that were not sent to Qwen in the current frame. Each label shows `#ID` followed
by the first response line.

## Results and Python usage

The video window is always displayed; press `q` or Escape, or close it, to stop.
`--output-video` saves processed frames at the source frame rate without audio.
`--output` saves one JSON entry per described track, containing `tracker_id`
and the full `response`. The same JSON is printed when processing ends.

When used from Python, build `build_frame_pipeline(prompt="...", frame_rate=...)`
once and call it for consecutive BGR frames. It returns
`(annotated_frame, detections)`, with full answers in
`detections.data["response"]`. Build a fresh pipeline for each video: tracker
state and cached responses belong to that pipeline.

## Limitations

- Responses describe the first usable observation. Changing activities or
  clothing are not checked again automatically.
- `unknown` and non-empty cut-off answers are cached too. They are not retried
  or completed on later frames. Empty answers and inference failures stop
  processing.
- The cache uses track IDs, not identity. An ID switch can attach an answer to
  the wrong object; a new ID for the same object triggers another request.
- Responses stay in memory for the video's lifetime, including temporarily
  lost tracks. Nothing is saved across videos unless exported as JSON.
- Qwen calls are sequential. Describing a new track pauses frame processing,
  so this is not a real-time or asynchronous pipeline.
