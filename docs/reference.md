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
| `Detections.AttachData(*, overwrite=False)` | `(sv.Detections, Mapping[str, list or ndarray])` -> `sv.Detections` | Adds multiple aligned data fields atomically; validates all fields before attachment. Existing keys require `overwrite=True`. |
| `Detections.AttachDataField(key, *, overwrite=False)` | `(sv.Detections, list or ndarray)` -> `sv.Detections` | Adds one aligned column to `data[key]`. Existing keys require `overwrite=True`. |
| `Detections.Filter(filter_fn)` | `sv.Detections` -> `sv.Detections` | Applies a custom callable that returns detections or a boolean selection mask, while retaining the detections contract. |
| `Detections.NMS(...)` | `sv.Detections` -> `sv.Detections` | Applies [upstream non-maximum suppression](https://supervision.roboflow.com/0.30.9/detection/core/#supervision.detection.core.Detections.with_nms): masks first, otherwise oriented boxes, otherwise `xyxy` boxes. |
| `Detections.NMM(...)` | `sv.Detections` -> `sv.Detections` | Applies Supervision non-maximum merge. |
| `Detections.Stitch()` | `(list[sv.Detections], list[TileRect])` -> `sv.Detections` | Moves tiled detections into source-image coordinates and merges them. |
| `DetectionsSmoother(length=5)` | `sv.Detections` -> `sv.Detections` | Smooths tracked boxes over `length` frames; requires tracker IDs and does not support segmentation. See [smoothing and reset](tutorials/track_objects.md#smooth-tracked-detections). |

## Detection Transformations

Transform detection boxes while retaining aligned detection fields. Box edits
return new detections and affect only `xyxy`; other geometry is left unchanged.

| Operator | Input -> Output | Notes |
|---|---|---|
| `Detections.PadBoxes(px, py=None)` | `sv.Detections` -> `sv.Detections` | Calls `sv.pad_boxes`: adds pixels on each side; `py` defaults to `px`. Does not clip to image bounds. |
| `Detections.ScaleBoxes(factor_x, factor_y=None)` | `sv.Detections` -> `sv.Detections` | Scales width/height around each box's center; `factor_y` defaults to `factor_x`. Factors must be finite and positive; values below `1` shrink. Does not resize the image. |
| `Detections.MoveBoxes(offset)` | `sv.Detections` -> `sv.Detections` | Calls `sv.move_boxes` to translate boxes by `(dx, dy)`. |
| `Detections.ClipBoxes(*, rounding="none", invalid="keep")` | `(image, sv.Detections)` -> `(image, sv.Detections)` | Clips to image bounds, then rounds; passes the image through unchanged. `rounding`: `"none"`, `"nearest"` (ties to even), `"outward"`, or `"inward"`. `invalid`: `"keep"`, `"drop"` (removes aligned rows), or `"raise"` (`ValueError`). |
| `Detections.Crop(*, copy=True)` | `(image, sv.Detections)` -> `(sv.Detections, list[ImagePayload])` | One crop per detection, in order, with nearest rounding and bounds clamping. Invalid or empty regions raise; use `ClipBoxes` to filter first. Copies by default; `copy=False` permits views. Returned detections retain source-image coordinates. |
| `Detections.Merge()` | `list[sv.Detections]` -> `sv.Detections` | Calls `sv.Detections.merge(...)` to concatenate compatible prediction sets. |

## Annotation

All annotators preserve the detection handoff: `(scene, detections)` ->
`(scene, detections)`. They copy the input scene before delegating to
Supervision, so annotation never mutates the source image. Constructor values
configure the underlying Supervision annotator.

Annotators backed by Supervision's per-detection color lookup also accept
`custom_color_lookup=...`. This callback receives a `Detection` and
returns an integer index into the configured `color` palette (or
`border_color` palette for `CropAnnotator`), overriding `color_lookup` with
color based on tracking IDs or custom detection data.

| Operator group | Operators | Notes |
|---|---|---|
| Detection | `BoxAnnotator`, `BoxCornerAnnotator`, `CircleAnnotator`, `ColorAnnotator`, `DotAnnotator`, `EllipseAnnotator`, `HaloAnnotator`, `LabelAnnotator`, `OrientedBoxAnnotator`, `RichLabelAnnotator`, `RoundBoxAnnotator`, `TriangleAnnotator` | `LabelAnnotator` and `RichLabelAnnotator` compose class, confidence, and tracker-ID labels with `show_*`, or use `label_formatter=...` for custom labels. The callback receives a `Detection` (box, confidence, class ID, tracker ID, and per-detection data) and returns label text. Data fields are row values from lists or arrays; nested collections are not wrapped or flattened. |
| Segmentation and region | `MaskAnnotator`, `PolygonAnnotator`, `PolygonZoneAnnotator`, `BlurAnnotator`, `CropAnnotator`, `HeatMapAnnotator`, `PixelateAnnotator` | `HeatMapAnnotator` retains heat across frames. Call `reset()` between independent videos to clear accumulated heat without replacing the configured annotator. |
| Tracking and overlays | `TraceAnnotator`, `FPSAnnotator`, `LineZoneAnnotator`, `BackgroundOverlayAnnotator`, `ComparisonAnnotator`, `IconAnnotator`, `PercentageBarAnnotator` | `TraceAnnotator` retains trajectories across frames. Call `reset()` between independent videos to clear trajectories without replacing the configured annotator. |

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
