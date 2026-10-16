# Detect and describe: prompt-driven crop analysis

Run local object detection with RF-DETR Small, filter to the requested classes,
then ask Qwen3-VL on Amazon Bedrock a question about each retained crop. The
same pipeline supports clothing descriptions, visible activity or work-role
cues, object condition, and other visually grounded questions. People are
selected by default; choose other detector class IDs for other subjects.

The annotated image is always displayed. Each label is the first line of the
model's response; complete responses are retained in the results. Requests are
sequential, one per retained detection. Images without retained detections do
not initialize the Bedrock client or make Bedrock requests.

## Setup

From the repository root:

```bash
python -m pip install -e '.[inference,bedrock]'

# Set this securely in your shell/IDE; never commit the real key.
export AWS_BEARER_TOKEN_BEDROCK="<your-bedrock-api-key>"
export AWS_DEFAULT_REGION="us-east-1"

python examples/run_detect_and_describe.py
python examples/run_detect_and_describe.py --input path/to/people.jpg --output-image annotated.jpg --output results.json
```

Without `--input`, the image example uses Supervision's people-walking image in
`examples/.example_assets`, downloading it if missing. Pass `--input` to use
your own image.

Standard AWS SDK credentials/profiles also work. Your key or IAM credentials
must permit model invocation, and the selected region/account must support the
model. Override the region with `--region`.

The default Bedrock model ID is `qwen.qwen3-vl-235b-a22b`, not the locally hosted
2B/8B variants. The operator uses the Bedrock **Converse API**, not a separate
OpenAI-compatible endpoint. See the [AWS model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-qwen-qwen3-vl-235b-a22b.html)
and [API-key setup](https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys-use.html).

## Change the task

Use `--prompt` to ask a different question. Shared instructions are appended to
every task: use visible evidence, put a short answer on the first line, provide
a brief explanation afterward, and answer `unknown` when the attribute cannot
be established from the crop. The default task describes appearance/activity.

```bash
# Fashion: discover labels without supplying a taxonomy.
python examples/run_detect_and_describe.py --input people.jpg --prompt "Describe the outfit's fashion style using one or two short labels."

# Visible occupation cues, not a claim about the person's actual employment.
python examples/run_detect_and_describe.py --input people.jpg --padding 0.2 --prompt "Describe the visible work role or activity, if supported by tools, actions, or context. Do not infer actual occupation from clothing alone."

# A supplied label set; unknown remains a valid outcome.
python examples/run_detect_and_describe.py --input people.jpg --prompt "Is the person carrying a bag? Choose carrying-bag, no-visible-bag, or unknown."

# A non-person task: RF-DETR category 2 is bicycle.
python examples/run_detect_and_describe.py --input street.jpg --class-ids 2 --prompt "Describe the bicycle's visible condition."

# Residency usually cannot be established visually: expect unknown.
python examples/run_detect_and_describe.py --input people.jpg --prompt "Can you establish whether the person is a tourist or resident from visible evidence? Do not equate a camera or luggage with residency status."
```

These are prompt-driven answers, not validated classifications: a supplied
label set is not enforced by a parser. No task-specific modes or taxonomies are
hard-coded. Prompts encourage abstention but cannot guarantee correctness.
Actual occupation and tourist/resident status generally require information
outside a single crop. Visible activity or occupational cues are different
from verified personal attributes. Evaluate against labeled data before using
these outputs for decisions about people.

## Results and configuration

Results are printed as JSON and optionally saved with `--output`. Each result
contains `detection_index`, source-image `xyxy`, detector `class_id` and
`confidence`, and the complete text in `response`. Use `--output-image` to
also save the annotated image. Detector confidence is **not** confidence in
the VLM's answer.

When used from Python, `build_pipeline(prompt="...")` returns a pipeline whose
output is `(annotated_image, detections)`. Responses live in
`detections.data["response"]`; there is no duplicate response-list output.

Crop preparation is explicit: `PadBoxes` expands each box by 5% per side by
default (`--padding 0.05`), `ClipBoxes` constrains those crop boxes to the source
image and drops unusable ones, and `CropDetections` slices the prepared boxes.
Only clipping needs image dimensions. Original detection boxes remain unchanged
for annotation and JSON results. These operators can be used independently:
padding does not clip, and clipping can take unpadded detection boxes.

Only crops, not the entire source image, are sent to Bedrock. An explicit
ml-pipes `Resize` step letterboxes each crop to 1024 x 1024 while preserving
its aspect ratio. Large crops are downscaled; small crops are padded without
upscaling. `Qwen3VL` only encodes the prepared image and runs inference; it
does not resize images itself. Remote inference incurs Bedrock usage charges.

Increase `--padding` when nearby tools or context matter, or use `--padding 0`
for exact detection crops. This still does not provide full-scene context.
Use `--class-ids 1` to select classes (person is ID 1 for this RF-DETR model),
`--confidence 0.5` to change the detector threshold, and `--detection-model-id`
or `--qwen-model-id` to override models. Class mappings may differ for other
detectors. Full-scene reasoning and structured-output validation are outside
these simple examples.

## Video with tracking

Use `run_track_and_describe.py` with the same prompt, model, region, class,
confidence, and padding options:

```bash
python examples/run_track_and_describe.py
python examples/run_track_and_describe.py --input video.mp4 --prompt "Describe the outfit's fashion style." --output-video annotated.mp4 --output tracks.json
```

By default it uses Supervision's people-walking video in
`examples/.example_assets`, downloading it if missing. Use `--input` to select
another video.

RF-DETR runs on every frame, then ByteTrack associates the filtered detections.
Pending tracker IDs (`-1`) are excluded. For each confirmed ID without a cached
response, the pipeline pads, clips, crops, resizes, and sends that crop to Qwen.
`TrackAttributes({"response": ""})` stores and restores detection data by tracker
ID without filtering. An inline `Detections.Filter` selects missing responses
and deduplicates IDs. The same stateful instance runs before filtering, after
attaching new answers, and when restoring all current tracks for annotation.
Configure other data-field defaults to reuse it for attributes besides text;
defaults and `None` are treated as missing and never overwrite cached values.
Only the successfully described subset updates the cache. Cached responses are
then attached to **all current confirmed tracks**, so already-described tracks
remain visible without more Bedrock requests. Each label shows `#ID` followed
by the first response line. Unusable crops remain pending and can be described
when a valid crop is available later.

The video window is always displayed; press `q` or Escape, or close it, to stop.
`--output-video` optionally saves the processed frames at the source frame rate
(without audio). `--output` optionally saves one JSON entry per described track,
containing `tracker_id` and the full `response`; the same JSON is printed.

When used from Python, build `build_frame_pipeline(prompt="...", frame_rate=...)`
**once** and call it for consecutive BGR frames. It returns
`(annotated_frame, detections)` with answers in `detections.data["response"]`.
Build a fresh pipeline for each video; tracker state and the response cache
belong to that pipeline, not to a global store.

This is intentionally a simple, synchronous demo:

- New tracks wait for confirmation (two consecutive detections) before analysis.
- `unknown` is cached as an answer, avoiding repeated charges for unclear crops.
- Responses are first-observation snapshots; changing activities or clothing
  are not re-evaluated automatically.
- The cache is by **track ID, not identity**. An ID switch can misapply a label;
  the same person receiving a new ID can trigger another request.
- Responses remain in memory for the video's lifetime, including temporarily
  lost tracks. There is no cross-video identity matching or persistent cache.
- Calls are sequential. A new track can pause processing while Bedrock responds;
  this is not a real-time/asynchronous pipeline. API failures stop processing
  rather than being cached as descriptions.

This replaces `run_fashion_style.py`: the script is now
`run_detect_and_describe.py`, and the former `fashion_style` result field is
now the task-neutral `response` field.
