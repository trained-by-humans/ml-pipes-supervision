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
Save these example support files beside your own Python script:
[run_detect_and_describe.py](https://raw.githubusercontent.com/trained-by-humans/ml-pipes-supervision/main/examples/run_detect_and_describe.py),
[qwen_local.py](https://raw.githubusercontent.com/trained-by-humans/ml-pipes-supervision/main/examples/qwen_local.py),
and [common.py](https://raw.githubusercontent.com/trained-by-humans/ml-pipes-supervision/main/examples/common.py).
They provide the crop helpers and local VLM adapter imported below;
these are example code, not public package operators.

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

```{ .py hl_lines="9-12" }
from ml_pipes.standard import Recall

from run_detect_and_describe import ClipBoxes, CropDetections, PadBoxes

pipeline = Pipeline(
    [
        ...,  # Earlier operators unchanged.
        Detections.NMS(threshold=0.5),
        PadBoxes(padding=0.05),
        Recall("source_image", prepend=True),
        ClipBoxes(),
        CropDetections(),
    ],
    auto_validate=True,
)
```

The stage returns `(detections, crops)`, with one BGR `ImagePayload` per
retained detection, in matching order.

## Describe each detection

Each crop is an independent VLM request, so
[Scatter/Gather](https://ml-pipes.com/regions/#scatter-gather) can process crops
concurrently and collect answers in input order. This keeps answers aligned
with their detections even when requests finish out of order.

Prepare each crop according to the model adapter's requirements. Here, the
adapter expects RGB images. The 1024-pixel maximum edge is a pipeline choice
to control inference cost, not a Qwen requirement. The model's processor
handles resizing, normalization, and patch alignment internally, so no
additional model-specific preprocessing is needed in the pipeline.

```{ .py hl_lines="11-23" }
from ml_pipes.standard import Gather, Pick, Scatter
from ml_pipes.vision import ConvertColorSpace

from qwen_local import LocalQwen3VL
from run_detect_and_describe import LimitCropSize

pipeline = Pipeline(
    [
        ...,  # Earlier operators unchanged.
        CropDetections(),
        Store("detections", source=0),
        Pick(1),  # Select crops from (detections, crops); detections are stored.
        Scatter(max_concurrency=1),
        LimitCropSize(max_size=1024),
        Pick(0),  # Discard the resize transform; source boxes stay unchanged.
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
prompt = """Task:
Describe the visible subject's appearance and activity.

Use only visible evidence in this image crop.
Put a short answer or label on the first line, followed by a brief explanation
on subsequent lines.
If the requested attribute cannot be determined visually, or the crop is too
small, blurry, or obscured, put 'unknown' on the first line and explain why.
Do not guess hidden personal attributes from appearance."""
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
than introducing a separate result object. This operator preserves the original
boxes, classes, confidence scores, tracking IDs, and other data while returning
a copy with the responses attached:

```python
from dataclasses import replace

import numpy as np
import supervision as sv

from ml_pipes.operator import Operator


@Operator
class AttachResponses:
    def __call__(self, detections: sv.Detections, responses: list[str]) -> sv.Detections:
        if len(detections) != len(responses):
            raise ValueError("Each detection must have exactly one model response.")
        return replace(
            detections,
            data={**detections.data, "response": np.asarray(responses, dtype=str)},
        )
```

Recall the detections stored before `Scatter` and pair them with the ordered
responses from `Gather`. Each response occupies the same index as its detection:

```{ .py hl_lines="5-6" }
pipeline = Pipeline(
    [
        ...,  # Earlier operators unchanged.
        Gather(),
        Recall("detections", prepend=True),
        AttachResponses(),
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
        AttachResponses(),
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

    This uses the imports, `prompt`, and helper operators defined above.

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
            PadBoxes(padding=0.05),
            Recall("source_image", prepend=True),
            ClipBoxes(),
            CropDetections(),
            Store("detections", source=0),
            Pick(1),  # Select crops from (detections, crops); detections are stored.
            Scatter(max_concurrency=1),
            LimitCropSize(max_size=1024),
            Pick(0),  # Discard the resize transform; source boxes stay unchanged.
            ConvertColorSpace(output_color_space="RGB"),
            LocalQwen3VL(
                prompt=prompt,
                model_id="Qwen/Qwen3-VL-2B-Instruct",
                device="cpu",
                max_tokens=256,
            ),
            Gather(),
            Recall("detections", prepend=True),
            AttachResponses(),
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

Its first three responses:

```text
[1142.0, 952.0, 1243.0, 1080.0] Back view of a person wearing a light-colored shirt and dark pants, with a strap over the shoulder, possibly from a bag or backpack. The person is facing away from the camera and appears to be walking or moving forward.
[1410.0, 703.0, 1521.0, 888.0] Subject: A person wearing a dark shirt, dark shorts, and a backpack, holding a phone to their ear.
[666.0, 651.0, 745.0, 855.0] Subject: A person viewed from behind, wearing a dark vest over a plaid shirt, dark pants, and a dark beanie, walking forward while carrying a bag in their left hand.
```

Labels use the first response line without changes. In this run, the model returned
single-line descriptions rather than short labels, so some annotations are
wide or clipped at the image edge.

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

The report below uses `ImageAssets.PEOPLE_WALKING`.

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
average per-crop timings. This helps identify expensive steps:

```text
  0:LoadFile                        0.32ms  ( 0.0%)
  1:Decode                          6.11ms  ( 0.0%)
  2:ImageToArray                    0.57ms  ( 0.0%)
  3:Store                           0.01ms  ( 0.0%)
  4:RoboflowInference             391.46ms  ( 0.3%)
  5:Select                          0.12ms  ( 0.0%)
  6:DetectionsFromInference         0.18ms  ( 0.0%)
  7:DetectionsFilter                0.21ms  ( 0.0%)
  8:DetectionsFilter                0.05ms  ( 0.0%)
  9:DetectionsNMS                   0.20ms  ( 0.0%)
  10:PadBoxes                       0.03ms  ( 0.0%)
  11:Recall                         0.00ms  ( 0.0%)
  12:ClipBoxes                      0.07ms  ( 0.0%)
  13:CropDetections                 0.20ms  ( 0.0%)
  14:Store                          0.01ms  ( 0.0%)
  15:Pick                           0.02ms  ( 0.0%)
  16:Scatter                     125614.65ms  (99.6%)
    ↳ child trace [n_items=11, concurrency=1]:
      17:LimitCropSize                  0.08ms  ( 0.0%)
      18:Pick                           0.02ms  ( 0.0%)
      19:ConvertColorSpace              0.12ms  ( 0.0%)
      20:LocalQwen3VL                11419.22ms  (100.0%)
      total                          11419.47ms
  22:Recall                         0.02ms  ( 0.0%)
  23:AttachResponses                0.22ms  ( 0.0%)
  24:Recall                         0.00ms  ( 0.0%)
  25:BoxAnnotator                   1.61ms  ( 0.0%)
  26:LabelAnnotator                 1.21ms  ( 0.0%)
  27:PlotImage                     39.63ms  ( 0.0%)
  total                          126057.56ms
```

With local Qwen on CPU, this run took **126.06 seconds**, with 99.6% spent in
the crop-description region. The **first** crop includes lazy Qwen model loading;
weight downloads and detector construction happened **before** timing.
