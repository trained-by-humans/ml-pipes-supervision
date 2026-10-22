---
title: Detect and Describe
description: >-
  Build a detection-to-VLM pipeline that answers a question about each detected object and annotates it with the response.
---

# Detect and Describe

Pipeline source: [run_detect_and_describe.py](https://github.com/trained-by-humans/ml-pipes-supervision/blob/main/examples/run_detect_and_describe.py)

Detect objects, then ask a vision-language model (VLM) a question about each
one. The detector finds the boxes; the VLM generates free-form text from their
image crops. Use this to describe clothing or activity, describe a vehicle's
visible condition, answer “What is this person carrying?”, or attach searchable
descriptions to objects. Unlike classification, answers need not come from a
predefined label set.

Build the pipeline in five stages: detection and filtering, crop preparation,
description, response attachment, and annotation. This example uses local
RF-DETR Small and Qwen3-VL, but the same structure applies to other detectors
and VLMs.

## Install dependencies

```bash
python -m pip install "ml-pipes-supervision[inference]"
python -m pip install "transformers>=4.57.0,<6" "torch>=2.6" "Pillow>=10"
```

The VLM runtime is an explicit install, not a common package dependency.
Save [qwen_local.py](https://raw.githubusercontent.com/trained-by-humans/ml-pipes-supervision/main/examples/qwen_local.py)
beside your own Python script for the local VLM adapter imported below.
Scaling, clipping, cropping, and data attachment use the public `Detections`
operators.

> [!TIP]
> CPU inference can be slow: the default 2B VLM's float32 weights alone need roughly 8 GB,
> plus runtime memory. Use `device="mps"` or `"cuda"` if available.

## Start with the source image

Download Supervision's people-walking image, or use your own image path.

```python
from supervision.assets import ImageAssets, download_assets

source_path = download_assets(ImageAssets.PEOPLE_WALKING, directory=".")
```

## Detect and filter subjects

Every retained detection becomes a VLM request, which is typically much more
expensive than detection. Filter to relevant classes, discard low-confidence
predictions, and suppress duplicate boxes before generating descriptions to
avoid spending inference time and memory on irrelevant or repeated subjects.

```python
import numpy as np

from ml_pipes.core import Pipeline
from ml_pipes.standard import Select, Store
from ml_pipes.supervision import Detections, ImageToArray
from ml_pipes.supervision.inference import RoboflowInference
from ml_pipes.vision import Decode, LoadFile

pipeline = Pipeline(
    [
        LoadFile(),
        Decode(),
        ImageToArray(),
        Store("source_image"),
        RoboflowInference(model_id="rfdetr-small", confidence=0.5),
        Select(0),
        Detections.FromInference(),
        Detections.Filter(lambda detections: np.isin(detections.class_id, (1,))),  # RF-DETR: 1 = person.
        Detections.Filter(lambda detections: detections.confidence >= 0.5),
        Detections.NMS(threshold=0.5),
    ],
    auto_validate=True,
)
```

## Prepare a crop for each detection

The VLM needs an image, not just detection metadata. Cropping each subject
from the source image gives it a focused view and helps preserve details.

Padding each box by a fraction of its width and height adds nearby context,
such as an object the person is carrying. Expanded boxes can extend beyond
the image, so clip them before slicing to avoid negative-index wrapping and
discard unusable crops.

```{ .py hl_lines="7-10" }
from ml_pipes.standard import Recall

pipeline = Pipeline(
    [
        ...,  # Earlier operators unchanged.
        Detections.NMS(threshold=0.5),
        Detections.ScaleBoxes(1.1),  # 5% padding per side.
        Recall("source_image", prepend=True),
        Detections.ClipBoxes(rounding="outward", invalid="drop"),
        Detections.Crop(),
    ],
    auto_validate=True,
)
```

The stage returns `(detections, crops)`, with one BGR `ImagePayload` per
retained detection, in matching order. These same padded, image-clipped boxes
are used for annotation; original detector coordinates are not restored.

For a padding fraction per side, use `Detections.ScaleBoxes(1 + 2 * padding)`.
`Detections.PadBoxes(px, py=None)` instead takes pixels per side. Clip drops
invalid detections with all their row data before the survivors are stored for
response attachment. No coordinate-preservation callbacks or temporary fields
are needed.

The recipe chooses outward rounding and drops invalid regions. Public Clip
defaults instead match `sv.clip_boxes`: no rounding and no row removal.

## Describe each detection

Each crop is an independent VLM request, so
[Scatter/Gather](https://ml-pipes.com/regions/#scatter-gather) can process crops
concurrently and collect answers in input order. This keeps answers aligned
with their detections even when requests finish out of order.

Prepare each crop according to the model adapter's requirements. Here, the
adapter expects RGB images. Crops retain their source resolution; the model's
processor handles resizing, normalization, and patch alignment internally,
so no additional model-specific preprocessing is needed in the pipeline.

```{ .py hl_lines="10-20" }
from ml_pipes.standard import Gather, Pick, Scatter
from ml_pipes.vision import ConvertColorSpace

from qwen_local import LocalQwen3VL

pipeline = Pipeline(
    [
        ...,  # Earlier operators unchanged.
        Detections.Crop(),
        Store("detections", source=0),
        Pick(1),  # Select crops from (detections, crops); detections are stored.
        Scatter(max_concurrency=1),
        ConvertColorSpace(output_color_space="RGB"),
        LocalQwen3VL(
            prompt=prompt,
            model_id="Qwen/Qwen3-VL-2B-Instruct",
            device="cpu",
            max_tokens=256,
        ),
        Gather(),
    ],
    auto_validate=True,
)
```

Here is the full `prompt` used in this guide:

```python
prompt = (
    "Task:\nDescribe the visible subject's appearance and activity.\n\n"
    "Use only visible evidence in this image crop. "
    "Put a short answer or label on the first line, followed by a brief explanation "
    "on subsequent lines. "
    "If the requested attribute cannot be determined visually, or the crop is too "
    "small, blurry, or obscured, put 'unknown' on the first line and explain why. "
    "Do not guess hidden personal attributes from appearance."
)
```

The first line will become the annotation label; the explanation remains
available in the full response. Asking for `unknown` helps express uncertainty,
but does not guarantee correctness. Keep questions grounded in visible evidence,
not hidden personal traits or facts requiring wider scene or temporal context.

!!! warning "Only enable concurrency when the model supports it"

    `Scatter(max_concurrency=N)` can process crops concurrently using threads,
    not separate processes; `Gather` preserves their input order. Increase
    `N` only if the model adapter safely handles simultaneous requests and has
    enough memory. This local Qwen wrapper shares one lazily loaded model and
    requires sequential calls, so keep `max_concurrency=1`. A concurrency-safe
    inference server is an option for parallel requests.

## Attach responses to detections

Supervision's `Detections` already supports extra per-detection fields through
its `data` dictionary. Add the answers as `detections.data["response"]` rather
than introducing a separate result object. `Detections.AttachDataField("response")`
preserves the supplied padded/clipped boxes, classes, confidence scores,
tracking IDs, and other data while returning new detections with the responses
attached. It validates one value per detection, never silently drops rows, and
does not modify collection-level `metadata`. Existing data keys require explicit
`overwrite=True`; this recipe adds a new key.

Recall the detections stored before `Scatter` and pair them with the ordered
responses from `Gather`. Each response occupies the same index as its detection:

```{ .py hl_lines="5-6" }
pipeline = Pipeline(
    [
        ...,  # Earlier operators unchanged.
        Gather(),
        Recall("detections", prepend=True),
        Detections.AttachDataField("response"),
    ],
    auto_validate=True,
)
```

## Annotate the result

With responses attached, annotate the original image with the VLM's answers.
`LabelAnnotator` accepts a `label_formatter` callback that builds each label
from the detection itself, including its extra `data` fields.

```{ .py hl_lines="7-12" }
from ml_pipes.supervision import BoxAnnotator, LabelAnnotator, PlotImage

pipeline = Pipeline(
    [
        ...,  # Earlier operators unchanged.
        Detections.AttachDataField("response"),
        Recall("source_image", prepend=True),
        BoxAnnotator(),
        LabelAnnotator(
            label_formatter=lambda detection: detection.data["response"].splitlines()[0],   # Use the first line as a compact label
        ),
        PlotImage(at=0),
    ],
    auto_validate=True,
)
```

## Running the pipeline

Run the completed pipeline to display the annotated image and read the answers:

??? note "Full pipeline"

    This uses the imports and `prompt` defined above.

    ```python
    pipeline = Pipeline(
        [
            LoadFile(),
            Decode(),
            ImageToArray(),
            Store("source_image"),
            RoboflowInference(model_id="rfdetr-small", confidence=0.5),
            Select(0),
            Detections.FromInference(),
            Detections.Filter(lambda detections: np.isin(detections.class_id, (1,))),  # RF-DETR: 1 = person.
            Detections.Filter(lambda detections: detections.confidence >= 0.5),
            Detections.NMS(threshold=0.5),
            Detections.ScaleBoxes(1.1),
            Recall("source_image", prepend=True),
            Detections.ClipBoxes(rounding="outward", invalid="drop"),
            Detections.Crop(),
            Store("detections", source=0),
            Pick(1),  # Select crops from (detections, crops); detections are stored.
            Scatter(max_concurrency=1),
            ConvertColorSpace(output_color_space="RGB"),
            LocalQwen3VL(
                prompt=prompt,
                model_id="Qwen/Qwen3-VL-2B-Instruct",
                device="cpu",
                max_tokens=256,
            ),
            Gather(),
            Recall("detections", prepend=True),
            Detections.AttachDataField("response"),
            Recall("source_image", prepend=True),
            BoxAnnotator(),
            LabelAnnotator(
                label_formatter=lambda detection: detection.data["response"].splitlines()[0],
            ),
            PlotImage(at=0),
        ],
        auto_validate=True,
    )
    ```

```python
annotated_image, detections = pipeline(source_path)

for box, response in zip(detections.xyxy, detections.data["response"]):
    print(box.tolist(), response)
```

The result of the run on `ImageAssets.PEOPLE_WALKING`:

![People-walking detections annotated with VLM responses](../assets/detect_and_describe/annotated.jpg)

The first lines of its first three responses:

```text
[1136.0, 945.0, 1249.0, 1080.0] Label: Person wearing a light-colored shirt and dark pants with a strap over the shoulder.
[1404.0, 693.0, 1527.0, 898.0] Subject: A person wearing a backpack, dark shorts, and athletic shoes, holding a bottle. The person is standing on a tiled floor, facing away from the camera, and appears to be looking up or toward the right. The image is blurry and low resolution, making it difficult to determine exact details.
[662.0, 640.0, 749.0, 866.0] Subject: A person walking away from the camera, wearing a dark vest over a plaid shirt, dark pants, and a dark beanie, carrying a bag in their left hand.
```

Labels use the first response line without changes. Some responses still include
prefixes such as `Label:` or `Subject:` and long descriptions, so annotations
can be wide or clipped at the image edge. See the [full responses](../assets/detect_and_describe/responses.json)
for all 11 detections, including their boxes and scores.

The model can make mistakes, so review its descriptions. Detection scores don't
measure how reliable the descriptions are. `max_tokens` limits answer length;
cut-off answers are still shown.

## Run the example

The downloaded example contains the completed pipeline and defaults to the
people-walking image:

```bash
python run_detect_and_describe.py
python run_detect_and_describe.py --input people.jpg --prompt "What is this person carrying?"
python run_detect_and_describe.py --help
```

Use `--prompt` to change the question and `--class-ids` to select detector
classes. Run `--help` for all options, including device selection and saving
the results.

## Inspect the Pipeline

Use `Pipeline.inspect()` to capture each step's output, including the crops
and VLM responses. This runs the models again, so it takes about as long as a
normal run. Save the result as an HTML report:

```python
from ml_pipes.inspection import PipelineInspector

inspection = pipeline.inspect(source_path)
PipelineInspector().save(inspection, "inspection.html")
```

The report below was captured from the current example on 2026-10-22 using
`ImageAssets.PEOPLE_WALKING`. It completed with 11 detections, 11 aligned
responses, and no failed steps. The report, annotated image, responses, and
trace all come from the same inspection invocation.

Compared with the earlier capture, detector predictions and all 11 crop images
are unchanged. Annotation boxes now use the padded, clipped crop regions, and
responses remain lists after `Detections.AttachDataField("response")`.
`LimitCropSize` and its extra `Pick` are gone; the report now has 25 cards
instead of 27. Response wording differs; the old recipe prompt used different
line breaks from the example. See the [comparison record](../assets/detect_and_describe/comparison.json)
and [run configuration](../assets/detect_and_describe/run.json).

[![Detect-and-describe pipeline inspection](../assets/detect_and_describe/inspection.png)](../assets/detect_and_describe/inspection.html)

*Click the image to open the interactive inspection report.*

## Trace the runs

For a traced run, attach a collector before calling the pipeline. Use this
in place of the untraced call above:

```python
from ml_pipes.collectors import CaptureCollector

collector = CaptureCollector()
pipeline.set_tracing(collector)
annotated_image, detections = pipeline(source_path)

print(collector.last_trace)
```

The outer `Scatter` timing covers processing all crops; its nested trace shows
average per-crop timings. The saved trace below comes from the same
`pipeline.inspect()` invocation as the HTML report, so its total includes
inspection snapshot overhead:

```text
  0:LoadFile                        0.41ms  ( 0.0%)
  1:Decode                          5.64ms  ( 0.0%)
  2:ImageToArray                    0.56ms  ( 0.0%)
  3:Store                           0.01ms  ( 0.0%)
  4:RoboflowInference             356.56ms  ( 0.2%)
  5:Select                          0.22ms  ( 0.0%)
  6:DetectionsFromInference         0.21ms  ( 0.0%)
  7:DetectionsFilter                0.21ms  ( 0.0%)
  8:DetectionsFilter                0.05ms  ( 0.0%)
  9:DetectionsNMS                   0.19ms  ( 0.0%)
  10:DetectionsScaleBoxes           0.11ms  ( 0.0%)
  11:Recall                         0.00ms  ( 0.0%)
  12:DetectionsClipBoxes            0.11ms  ( 0.0%)
  13:DetectionsCrop                 0.43ms  ( 0.0%)
  14:Store                          0.01ms  ( 0.0%)
  15:Pick                           0.02ms  ( 0.0%)
  16:Scatter                     183668.28ms  (99.8%)
    ↳ child trace [n_items=11, concurrency=1]:
      17:ConvertColorSpace              0.17ms  ( 0.0%)
      18:LocalQwen3VL                16696.63ms  (100.0%)
      total                          16697.07ms
  20:Recall                         0.01ms  ( 0.0%)
  21:DetectionsAttachDataField      0.20ms  ( 0.0%)
  22:Recall                         0.00ms  ( 0.0%)
  23:BoxAnnotator                   0.96ms  ( 0.0%)
  24:LabelAnnotator                 2.00ms  ( 0.0%)
  25:PlotImage                     37.54ms  ( 0.0%)
  total                          184081.03ms
```

With local Qwen on CPU (float32, one PyTorch thread), this capture took
**184.08 seconds**, with 99.8% spent in the crop-description region.
The first crop includes lazy Qwen model loading. Qwen weights were already
cached, and detector construction happened before timing. Matplotlib used the
non-interactive Agg backend. This is not a like-for-like speed comparison with
the previous capture: inspection snapshots and generated answer lengths affect
the total.

Download the [text trace](../assets/detect_and_describe/trace.txt) or
[structured trace](../assets/detect_and_describe/trace.json) for this run.
