---
title: Supervision Operator Reference
description: >-
  Reference for ml_pipes.supervision operators, including detection conversion, annotation, tracking, zones, timers, image views, and Roboflow inference.
---

# ml-pipes-supervision Index

This page catalogs the Supervision compatibility surface in
`ml_pipes.supervision`. For installation and quickstart, see
[Home](index.md). For task and upstream API coverage,
see [`coverage.md`](./coverage.md).

For framework-wide operator concepts, see
[`ml-pipes operators`](https://github.com/trained-by-humans/ml-pipes/blob/main/docs/operators.md).
For the cross-package catalog, see
[`ml-pipes packages`](https://github.com/trained-by-humans/ml-pipes/blob/main/docs/packages.md).

## Public Modules

| Module | Scope |
|---|---|
| `ml_pipes.supervision` | Supervision-backed detection conversion, annotation, viewing, and zones. |
| `ml_pipes.supervision.inference` | Roboflow Inference model boundary. |
| `ml_pipes.supervision.trackers` | Adapters for the external Roboflow `trackers` package. |

## Detection Boundaries

| Operator | Input -> Output | Notes |
|---|---|---|
| `ImageToArray()` | `ImagePayload` -> `NDArray[uint8]` | Converts an HWC `ImagePayload` to a BGR image for Supervision and model APIs. |
| `Detections.FromInference(compact_masks=False)` | Inference result -> `sv.Detections` | Calls `sv.Detections.from_inference(...)`. Opt into cropped encoded masks with `compact_masks=True`; see [compact masks](tutorials/detect_and_annotate.md#compact-masks). |
| `Detections.FromUltralytics()` | Ultralytics result -> `sv.Detections` | Calls `sv.Detections.from_ultralytics(...)`. |
| `Detections.FromTensorRegistry(...)` | `TensorRegistry` -> `sv.Detections` | Converts configured `boxes`, `scores`, `classes`, and optional `masks` tensors. |
| `Detections.Filter(filter_fn)` | `sv.Detections` -> `sv.Detections` | Applies a custom callable that returns detections or a boolean selection mask, while retaining the detections contract. |
| `Detections.NMS(...)` | `sv.Detections` -> `sv.Detections` | Applies [upstream non-maximum suppression](https://supervision.roboflow.com/0.30.9/detection/core/#supervision.detection.core.Detections.with_nms): masks first, otherwise oriented boxes, otherwise `xyxy` boxes. |
| `Detections.NMM(...)` | `sv.Detections` -> `sv.Detections` | Applies Supervision non-maximum merge. |
| `Detections.Stitch()` | `(list[sv.Detections], list[TileRect])` -> `sv.Detections` | Moves tiled detections into source-image coordinates and merges them. |
| `DetectionsSmoother(length=5)` | `sv.Detections` -> `sv.Detections` | Smooths tracked boxes over `length` frames; requires tracker IDs and does not support segmentation. See [smoothing and reset](tutorials/track_objects.md#smooth-tracked-detections). |

## Detection Transformations

These operators accept `sv.Detections`, not raw box arrays. Geometry operators
return new detections without mutating their inputs: they change only `xyxy`,
preserving masks, oriented boxes, keypoints, confidence, classes, tracking IDs,
data, and metadata. Preserved fields may share storage with the input; this is
not a deep-copy guarantee. For raw boxes, use a typed `Map` callback with the
corresponding Supervision helper.

| Operator | Input -> Output | Semantics |
|---|---|---|
| `Detections.PadBoxes(px, py=None)` | detections -> detections | `sv.pad_boxes`: pixels added on **each side**; omitted `py` equals `px`. No automatic clipping. |
| `Detections.ScaleBoxes(factor_x, factor_y=None)` | detections -> detections | Scale width/height around each box's center. Omitted y equals x. Factors must be finite and positive; factors below 1 shrink. Independent factors extend Supervision's scalar API. This does not resize the image or scale coordinates about the origin. |
| `Detections.MoveBoxes(offset)` | detections -> detections | `sv.move_boxes`: translate boxes by `(dx, dy)`. Other geometry is not translated. |
| `Detections.ClipBoxes(*, rounding="none", invalid="keep")` | `(image, detections)` -> `(image, detections)` | Clip `xyxy` to image width/height; pass the image through unchanged. Optional rounding and invalid-row handling below. |
| `Detections.Crop(*, copy=True)` | `(image, detections)` -> `(detections, list[ImagePayload])` | One image region per detection, in matching order. Supervision-style nearest rounding and bounds clamping; copies by default. No crop-local coordinate conversion. |
| `Detections.Merge()` | `list[sv.Detections]` -> detections | Delegate to `sv.Detections.merge`: concatenate compatible prediction sets. Not the box-union operation called Detections Merge in Inference Workflows. |

Clip and Crop accept non-empty three-channel HWC `uint8` images: bare arrays
are BGR; `ImagePayload` inputs may be RGB or BGR. Crops preserve the input color
space and use HWC layout. `Crop(copy=False)` permits views of the source image.

Clip policies are predefined `Literal` values and validated at runtime:

| `rounding` | Minimum edges | Maximum edges |
|---|---|---|
| `"none"` (default) | unchanged | unchanged |
| `"nearest"` | round, ties to even | round, ties to even |
| `"outward"` | floor | ceil |
| `"inward"` | ceil | floor |

Rounding happens after bounds clipping. `invalid="keep"` (default) retains all
rows, matching `sv.clip_boxes`. `"drop"` removes rows and all their corresponding
fields; `"raise"` reports a `ValueError`. Both reject nonfinite/nonpositive input
boxes and boxes that become empty after clipping or rounding. Outward rounding
does not rescue invalid input boxes.

Crop never silently drops a detection. Invalid or pixel-empty regions raise;
use Clip with an explicit rounding policy and `invalid="drop"` when filtering
is needed. Its returned detections retain their supplied source-image geometry,
not coordinates local to each crop. No masks or other annotations are cropped.

```python
import supervision as sv

from ml_pipes.core import Pipeline
from ml_pipes.supervision import Detections
from ml_pipes.vision import ImagePayload

crop_pipeline = Pipeline(
    [
        Detections.ClipBoxes(rounding="outward", invalid="drop"),
        Detections.Crop(),
    ],
)
# Declare the pair at a multi-argument pipeline entry. Here image is an ImagePayload.
crop_pipeline.validate(pipeline_input_type=tuple[ImagePayload, sv.Detections])
# kept_detections, crops = crop_pipeline((image, detections))
```

## Annotation

All annotators preserve the detection handoff: `(scene, detections)` ->
`(scene, detections)`. They copy the input scene before delegating to
Supervision, so annotation never mutates the source image. Constructor values
configure the underlying Supervision annotator.

`TraceAnnotator` and `HeatMapAnnotator` retain state across frames. Call the
operator's `reset()` between independent videos to clear accumulated trajectories
or heat without replacing the configured annotator.

`LabelAnnotator` and `RichLabelAnnotator` can both compose class,
confidence, and tracker-ID labels with the `show_*` options, or create labels
from any detection data with `label_formatter=...`. The callback receives a
`Detection` with its box, confidence, class ID, tracker ID, and per-detection
data, and must return the label text.

Annotators backed by Supervision's per-detection color lookup also accept
`custom_color_lookup=...`. This callback receives the same `Detection` and
returns an integer index into the configured `color` palette (or
`border_color` palette for `CropAnnotator`), overriding `color_lookup` with
color based on tracking IDs or custom detection data.

| Operator group | Operators |
|---|---|
| Detection | `BoxAnnotator`, `BoxCornerAnnotator`, `CircleAnnotator`, `ColorAnnotator`, `DotAnnotator`, `EllipseAnnotator`, `HaloAnnotator`, `LabelAnnotator`, `OrientedBoxAnnotator`, `RichLabelAnnotator`, `RoundBoxAnnotator`, `TriangleAnnotator` |
| Segmentation and region | `MaskAnnotator`, `PolygonAnnotator`, `PolygonZoneAnnotator`, `BlurAnnotator`, `CropAnnotator`, `HeatMapAnnotator`, `PixelateAnnotator` |
| Tracking and overlays | `TraceAnnotator`, `FPSAnnotator`, `LineZoneAnnotator`, `BackgroundOverlayAnnotator`, `ComparisonAnnotator`, `IconAnnotator`, `PercentageBarAnnotator` |

## Zones And Views

| Operator | Input -> Output | Notes |
|---|---|---|
| `TriggerZone(zone)` | `sv.Detections` -> `sv.Detections` | Keeps detections for which `sv.PolygonZone.trigger(...)` is true. |
| `TrackingTimer(fps, field="tracking_time", reset_missing_tracks=True)` | tracked `sv.Detections` -> `sv.Detections` | Adds elapsed time for tracks present in the incoming stream. Unconfirmed negative IDs receive `0.0`; set `reset_missing_tracks=False` to retain a track's entry time across gaps. Filtering stages define membership. |
| `TriggerLineZone(line_zone)` | `sv.Detections` -> `sv.Detections` | Updates the line-zone counters and retains detections. |
| `PlotImage(at=None)` | payload -> payload | Displays one image through `sv.plot_image(...)`. |
| `ImageWindow(title="supervision", at=None)` | payload -> payload | Updates a [Tkinter/Pillow image window](https://supervision.roboflow.com/0.30.9/utils/image_window/#supervision.utils.image_window.ImageWindow); requires Tkinter and a desktop display. |
| `FPSMonitor(sample_size=30)` | payload -> payload | Writes the current FPS to the console. |

## Roboflow Inference

| Operator | Input -> Output | Notes |
|---|---|---|
| `RoboflowInference(model_id, api_key=None, ...)` | image -> model result | Resolves and owns a Roboflow Inference model, then calls `model.infer(...)`. |
| `RoboflowInference(model, ...)` | image -> model result | Uses a caller-initialized `inference.Model`; supports models that require custom construction, such as YOLO-World. |

## Trackers

| Operator | Input -> Output | Notes |
|---|---|---|
| `UpdateTrackedObjects(tracker)` | `sv.Detections` or `(sv.Detections, frame)` -> `sv.Detections` | Updates an external `trackers.BaseTracker`. |
| `ReadTrackedObjects(tracker)` | no input -> `sv.Detections` | Returns the tracker's current detections. |
| `ByteTrack(...)` | `sv.Detections` -> `sv.Detections` | Configures `trackers.ByteTrackTracker`. |
| `BoTSORT(enable_cmc=True, ...)` | `(sv.Detections, frame)` -> `sv.Detections` | Configures `trackers.BoTSORTTracker`; the frame is required when CMC is enabled. |
| `OCSORT(...)` | `sv.Detections` -> `sv.Detections` | Configures `trackers.OCSORTTracker`. |
| `SORT(...)` | `sv.Detections` -> `sv.Detections` | Configures `trackers.SORTTracker`. |
