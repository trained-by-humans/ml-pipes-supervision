# ml-pipes-supervision Index

This page catalogs the Supervision compatibility surface in
`ml_pipes.supervision`. For installation and quickstart, see
[Home](index.md). For task and upstream API coverage,
see [`coverage.md`](./coverage.md).

For framework-wide operator concepts, see
[`ml-pipes operators`](https://github.com/trained-by-humans/ml-pipes/blob/main/docs/OPERATORS.md).
For the cross-package catalog, see
[`ml-pipes packages`](https://github.com/trained-by-humans/ml-pipes/blob/main/docs/PACKAGES.md).

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
| `Detections.FromInference(compact_masks=False)` | Inference result -> `sv.Detections` | Calls `sv.Detections.from_inference(...)`. |
| `Detections.FromUltralytics()` | Ultralytics result -> `sv.Detections` | Calls `sv.Detections.from_ultralytics(...)`. |
| `Detections.FromTensorRegistry(...)` | `TensorRegistry` -> `sv.Detections` | Converts configured `boxes`, `scores`, `classes`, and optional `masks` tensors. |
| `Detections.Filter(filter_fn)` | `sv.Detections` -> `sv.Detections` | Applies a custom callable that returns detections or a boolean selection mask, while retaining the detections contract. |
| `Detections.NMS(...)` | `sv.Detections` -> `sv.Detections` | Applies Supervision non-maximum suppression. |
| `Detections.NMM(...)` | `sv.Detections` -> `sv.Detections` | Applies Supervision non-maximum merge. |
| `Detections.Stitch()` | `(list[sv.Detections], list[TileRect])` -> `sv.Detections` | Moves tiled detections into source-image coordinates and merges them. |
| `DetectionsSmoother(length=5)` | `sv.Detections` -> `sv.Detections` | Applies `sv.DetectionsSmoother`. |

## Annotation

All annotators preserve the detection handoff: `(scene, detections)` ->
`(scene, detections)`. They copy the input scene before delegating to
Supervision, so annotation never mutates the source image. Constructor values
configure the underlying Supervision annotator.

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
| `ImageWindow(title="supervision", at=None)` | payload -> payload | Updates an OpenCV-backed Supervision image window. |
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
