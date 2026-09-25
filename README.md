![python-version](https://img.shields.io/pypi/pyversions/supervision)
[![Supervision Coverage](https://img.shields.io/badge/supervision-0.30.0-purple)](./docs/coverage.md)

# ml-pipes-supervision

> [!IMPORTANT]
> `ml-pipes-supervision` remains a community-maintained operator package
> within the [ml-pipes](https://github.com/trained-by-humans) ecosystem. The
> project has its own maintainers and development roadmap, while benefiting
> from ml-pipes' verified publishing and distribution process.
>
> For contributions, issues, and project decisions, use this repository's
> maintainers and issue tracker.

## Hello

[Supervision](https://github.com/roboflow/supervision) is your essential toolkit for computer vision. From data loading to real-time zone counting, it provides the building blocks so you can focus on building applications around your models.
`ml-pipes-supervision` provides `Supervision`
capabilities as composable operators in [ml-pipes](https://github.com/trained-by-humans/ml-pipes).

## Coverage

| Task                        | Status      |
|-----------------------------|-------------|
| Classification              | Not covered |
| Detection                   | Covered     |
| Segmentation                | Covered     |
| Keypoints                   | Not covered |
| Tracking                    | Covered     |
| Tools (Zones, Slicer, etc.) | Covered     |
| Dataset                     | Not covered |
| Evaluation                  | Not covered |
| Vision-language models      | Not covered |

See [coverage](./docs/coverage.md) for role definitions and the detailed API
compatibility matrix.

## Install

Install directly from this repository in a [Python >=3.10](https://www.python.org/)
environment:

```bash
python -m pip install "ml-pipes-supervision @ git+https://github.com/trained-by-humans/ml-pipes-supervision.git"
```

This also installs the required `ml-pipes` packages, including
`ml-pipes-core` and `ml-pipes-vision`, plus the Supervision and Roboflow
Inference and tracker runtime dependencies.

The public operators are available from `ml_pipes.supervision`. Roboflow
Inference and external tracker boundaries are available from
`ml_pipes.supervision.inference` and `ml_pipes.supervision.trackers`.

To use the integration from another project's `pyproject.toml`, add the same
Git dependency:

```toml
dependencies = [
    "ml-pipes-supervision @ git+https://github.com/trained-by-humans/ml-pipes-supervision.git",
]
```

## Quickstart

Build the usual detection-and-annotation flow as one pipeline. The operators
below are the same thin boundaries used by the runnable examples.

```python
from ml_pipes.core import Pipeline
from ml_pipes.standard import Recall, Select, Store
from ml_pipes.vision import Decode, LoadFile
from ml_pipes.supervision import BoxAnnotator, Detections, ImageToArray, LabelAnnotator, PlotImage
from ml_pipes.supervision.inference import RoboflowInference

pipeline = Pipeline(
    [
        LoadFile(),
        Decode(),
        ImageToArray(),
        Store("source_image"),
        RoboflowInference(model_id="rfdetr-small"),
        Select(0),
        Detections.FromInference(),
        Recall("source_image", prepend=True),
        BoxAnnotator(),
        LabelAnnotator(),
        PlotImage(),
    ]
)
```

For model integrations that produce an `ml_pipes.tensor.TensorRegistry`, use
`Detections.FromTensorRegistry()` before Supervision annotators, trackers,
zones, or sinks. This is the explicit boundary from tensor post-processing to
Supervision data.

## Why Supervision with ml-pipes?

Supervision provides the computer-vision building blocks; ml-pipes makes the
boundaries between those blocks explicit, composable, and inspectable. A
pipeline can validate its contracts before execution and capture the value at
each operator boundary with `Pipeline.inspect()`. That makes complex flows
easier to understand and debug without adding ad-hoc logging to every step.

The Detect Small Objects pipeline is a good example: it tiles the input image,
runs inference on each tile, gathers the results, stitches detections back into
the source coordinate system, merges overlaps, and annotates the final image.
The inspection report shows every boundary in that flow.

[![Detect Small Objects pipeline inspection](docs/assets/detect_small_objects/inspection.png)](docs/assets/detect_small_objects/inspection.html)

*Click the image to open the interactive inspection report.*

## Built with Supervision x ml-pipes

<details>
<summary>View supported Supervision example pipelines</summary>

| Example | Upstream Source | Section | Note |
|---|---|---|---|
| [`run_detect_and_annotate.py`](./examples/run_detect_and_annotate.py) | [`Detect and Annotate` (`0.30.0`)](https://supervision.roboflow.com/0.30.0/how_to/detect_and_annotate/) | `Run Detection`, `Annotate Image with Detections`, `Display Custom Labels` | Runs object detection, then draws bounding boxes and available class labels on the image. |
| [`run_filter_detections.py`](./examples/run_filter_detections.py) | [`Filter Detections` (`0.30.0`)](https://supervision.roboflow.com/0.30.0/how_to/filter_detections/) | `Filter Detections` | Keeps detections by class, confidence, and relative bounding-box area before annotation. |
| [`run_segment_and_annotate.py`](./examples/run_segment_and_annotate.py) | [`Detect and Annotate` (`0.30.0`)](https://supervision.roboflow.com/0.30.0/how_to/detect_and_annotate/) | `Run Detection`, `Annotate Image with Segmentations` | Runs instance segmentation and draws masks and labels on the image. |
| [`run_detection_video.py`](./examples/run_detection_video.py) | [`Annotate Video with Detections`](https://supervision.roboflow.com/0.30.0/notebooks/annotate-video-with-detections/) | `Run Detection` | Detects and annotates objects on each video frame, with an FPS overlay. |
| [`run_save_detections.py`](./examples/run_save_detections.py) | [`Save Detections`](https://supervision.roboflow.com/latest/how_to/save_detections/) | `Save Detections` | Runs detection on each video frame and writes the results to CSV. |
| [`run_track_objects.py`](./examples/run_track_objects.py) | [`Track Objects` (`0.30.0`)](https://supervision.roboflow.com/0.30.0/how_to/track_objects/) | `Track Objects`, `Annotate Tracking IDs`, `Annotate Traces`, `Smooth Tracked Detections` | Assigns persistent IDs, smooths tracked boxes, and draws IDs, classes, and motion paths. |
| [`run_count_in_zone.py`](./examples/run_count_in_zone.py) | [`Count Objects in Zone` (`0.30.0`)](https://supervision.roboflow.com/0.30.0/how_to/count_in_zone/) | `Count Objects in Zone` | Counts and annotates detections inside each configured polygon zone. |
| [`run_traffic_analysis.py`](./examples/run_traffic_analysis.py) | [`Traffic Analysis`](https://github.com/roboflow/supervision/tree/develop/examples/traffic_analysis) | `Track Zone Visits` | Records ordered vehicle visits between configured zones and displays unique origin-to-destination totals. |
| [`run_time_in_zone.py`](./examples/run_time_in_zone.py) | [`Time in Zone`](https://github.com/roboflow/supervision/tree/develop/examples/time_in_zone) | `Process Video` | Tracks each object and displays its continuous dwell time within each configured polygon zone. |
| [`run_count_objects_crossing_line.py`](./examples/run_count_objects_crossing_line.py) | [`Count Objects Crossing the Line`](https://supervision.roboflow.com/latest/notebooks/count-objects-crossing-the-line/#process-video) | `Process Video` | Tracks objects and counts their crossings in each direction over a line. |
| [`run_detect_small_objects.py`](./examples/run_detect_small_objects.py) | [`Detect Small Objects` (`0.30.0`)](https://supervision.roboflow.com/0.30.0/how_to/detect_small_objects/) | `Use InferenceSlicer` | Splits an image into overlapping tiles, detects objects per tile, and merges the results. |
| [`run_zero_shot_object_detection.py`](./examples/run_zero_shot_object_detection.py) | [`Zero-Shot Object Detection with YOLO-World` (`0.30.0`)](https://supervision.roboflow.com/0.30.0/notebooks/zero-shot-object-detection-with-yolo-world/) | `Process Video` | Detects objects matching a supplied text prompt and filters duplicate or oversized predictions. |
| [`run_oriented_bounding_boxes.py`](./examples/run_oriented_bounding_boxes.py) | [`Oriented Bounding Boxes` (`0.30.0`)](https://supervision.roboflow.com/0.30.0/notebooks/oriented-bounding-boxes/) | `Oriented Box Annotation` | Detects ships and draws their rotated bounding boxes. |
| [`run_blur_faces.py`](./examples/run_blur_faces.py) | [`Blurring Faces`](https://github.com/roboflow/supervision/blob/develop/docs/notebooks/blurring_faces.ipynb) | `Detecting Faces`, `Blurring the Face` | Detects faces in MediaPipe's sample image locally, then blurs them. |

</details>

## Tutorials

Want to learn how to use Supervision with `ml-pipes`? Explore our
[how-to guides](https://requiem4machines.github.io/ml-pipes-supervision/tutorials/detect_and_annotate/)
and [end-to-end examples](./examples/)!

The GitHub Pages tutorials preserve the corresponding Supervision guides and
add their `ml-pipes` counterparts, so you can compare both approaches side by
side.
