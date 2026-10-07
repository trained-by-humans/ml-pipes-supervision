---
title: Supervision and Tracker API Coverage
description: >-
  Compare Supervision and Roboflow tracker APIs with ml-pipes operators, distinguishing pipeline steps, native configuration and data, and unsupported APIs.
---

# Supervision Coverage

## Roles

Not every upstream API needs a port. Configuration and data values are used
AS-IS; only operators and utilities with value as standalone pipeline steps
need an ml-pipes counterpart.

| Role         | Description           | Examples                                          |
|--------------|-----------------------|---------------------------------------------------|
| `Operator`   | Pipeline step         | `BoxAnnotator`, `TriggerZone`, `Detections.Filter()`, `Detections.NMS()` |
| `Config`     | Constructor value     | `Color`, `Point`, `Position`                      |
| `Data`       | Pipeline value        | `Detections`, `KeyPoints`, `CompactMask`          |
| `Utility`    | Direct callable       | `plot_image`, `box_iou`, `resize_image`           |
| `Deprecated` | Avoid new integration | `ByteTrack`                                       |

## Supervision API Coverage

| Source directory (`src/supervision`) | Kind | Supervision surface | ml-pipes correspondence |
|---|---|---|---|
| `annotators` | Component | `BackgroundOverlayAnnotator` | `BackgroundOverlayAnnotator` |
| `annotators` | Component | `BlurAnnotator` | `BlurAnnotator` |
| `annotators` | Component | `BoxAnnotator` | `BoxAnnotator` |
| `annotators` | Component | `BoxCornerAnnotator` | `BoxCornerAnnotator` |
| `annotators` | Component | `CircleAnnotator` | `CircleAnnotator` |
| `annotators` | Component | `ColorAnnotator` | `ColorAnnotator` |
| `annotators` | Component | `ColorLookup` | AS-IS (`Config`) |
| `annotators` | Component | `ComparisonAnnotator` | `ComparisonAnnotator` |
| `annotators` | Component | `CropAnnotator` | `CropAnnotator` |
| `annotators` | Component | `DotAnnotator` | `DotAnnotator` |
| `annotators` | Component | `EllipseAnnotator` | `EllipseAnnotator` |
| `annotators` | Component | `HaloAnnotator` | `HaloAnnotator` |
| `annotators` | Component | `HeatMapAnnotator` | `HeatMapAnnotator` |
| `annotators` | Component | `IconAnnotator` | `IconAnnotator` |
| `annotators` | Component | `LabelAnnotator` | `LabelAnnotator` |
| `annotators` | Component | `MaskAnnotator` | `MaskAnnotator` |
| `annotators` | Component | `OrientedBoxAnnotator` | `OrientedBoxAnnotator` |
| `annotators` | Component | `PercentageBarAnnotator` | `PercentageBarAnnotator` |
| `annotators` | Component | `PixelateAnnotator` | `PixelateAnnotator` |
| `annotators` | Component | `PolygonAnnotator` | `PolygonAnnotator` |
| `annotators` | Component | `RichLabelAnnotator` | `RichLabelAnnotator` |
| `annotators` | Component | `RoundBoxAnnotator` | `RoundBoxAnnotator` |
| `annotators` | Component | `TraceAnnotator` | `TraceAnnotator` |
| `annotators` | Component | `TriangleAnnotator` | `TriangleAnnotator` |
| `annotators` | Function | `hex_to_rgba` | None |
| `annotators` | Function | `is_valid_hex` | None |
| `annotators` | Function | `rgba_to_hex` | None |
| `classification` | Component | `Classifications` | None |
| `dataset` | Component | `BaseDataset` | None |
| `dataset` | Component | `ClassificationDataset` | None |
| `dataset` | Component | `DetectionDataset` | None |
| `dataset` | Function | `get_coco_class_index_mapping` | None |
| `detection` | Component | `CompactMask` | AS-IS (`Data`); `Detections.FromInference()` |
| `detection` | Component | `CSVSink` | AS-IS (`Utility`) |
| `detection` | Component | `Detections` | AS-IS (`Data`); `Detections.FromInference()`, `Detections.FromUltralytics()`, `Detections.Filter()`, `Detections.NMS()`, `Detections.NMM()` |
| `detection` | Component | `DetectionsSmoother` | `DetectionsSmoother` |
| `detection` | Component | `InferenceSlicer` | `ml_pipes.vision.Tile()` + `Detections.Stitch()` |
| `detection` | Component | `JSONSink` | AS-IS (`Utility`) |
| `detection` | Component | `LineZone` | AS-IS (`Config`); `TriggerLineZone` |
| `detection` | Component | `LineZoneAnnotator` | `LineZoneAnnotator` |
| `detection` | Component | `LineZoneAnnotatorMulticlass` | None |
| `detection` | Component | `LMM` | None |
| `detection` | Component | `OverlapFilter` | None |
| `detection` | Component | `OverlapMetric` | AS-IS (`Config`); `Detections.NMS()`, `Detections.NMM()` |
| `detection` | Component | `PolygonZone` | AS-IS (`Config`); `TriggerZone` |
| `detection` | Component | `PolygonZoneAnnotator` | `PolygonZoneAnnotator` |
| `detection` | Component | `VLM` | None |
| `detection` | Component | `WindowedRasterDataset` | None |
| `detection` | Function | `approximate_polygon` | None |
| `detection` | Function | `box_iou_batch_with_jaccard` | None |
| `detection` | Function | `box_iou_batch` | None |
| `detection` | Function | `box_iou` | None |
| `detection` | Function | `box_non_max_merge` | None |
| `detection` | Function | `box_non_max_suppression` | None |
| `detection` | Function | `box_soft_non_max_suppression` | None |
| `detection` | Function | `calculate_masks_centroids` | None |
| `detection` | Function | `clip_boxes` | None |
| `detection` | Function | `contains_holes` | None |
| `detection` | Function | `contains_multiple_segments` | None |
| `detection` | Function | `denormalize_boxes` | None |
| `detection` | Function | `edit_distance` | None |
| `detection` | Function | `filter_polygons_by_area` | None |
| `detection` | Function | `filter_segments_by_distance` | None |
| `detection` | Function | `fuzzy_match_index` | None |
| `detection` | Function | `is_compressed_rle` | None |
| `detection` | Function | `mask_iou_batch` | None |
| `detection` | Function | `mask_non_max_merge` | None |
| `detection` | Function | `mask_non_max_suppression` | None |
| `detection` | Function | `mask_soft_non_max_suppression` | None |
| `detection` | Function | `mask_to_polygons` | None |
| `detection` | Function | `mask_to_rle` | None |
| `detection` | Function | `mask_to_roi` | None |
| `detection` | Function | `mask_to_xyxy` | None |
| `detection` | Function | `move_boxes` | None |
| `detection` | Function | `move_masks` | None |
| `detection` | Function | `oriented_box_iou_batch` | None |
| `detection` | Function | `oriented_box_non_max_merge` | None |
| `detection` | Function | `oriented_box_non_max_suppression` | None |
| `detection` | Function | `pad_boxes` | None |
| `detection` | Function | `polygon_to_mask` | None |
| `detection` | Function | `polygon_to_xyxy` | None |
| `detection` | Function | `rle_to_mask` | None |
| `detection` | Function | `scale_boxes` | None |
| `detection` | Function | `xcycwh_to_xyxy` | None |
| `detection` | Function | `xywh_to_xyxy` | None |
| `detection` | Function | `xyxy_to_mask` | None |
| `detection` | Function | `xyxy_to_polygons` | None |
| `detection` | Function | `xyxy_to_xcycarh` | None |
| `detection` | Function | `xyxy_to_xywh` | None |
| `detection` | Function | `xyxyxyxy_to_xyxy` | None |
| `draw` | Component | `Color` | AS-IS (`Config`) |
| `draw` | Component | `ColorPalette` | AS-IS (`Config`) |
| `draw` | Function | `calculate_optimal_line_thickness` | None |
| `draw` | Function | `calculate_optimal_text_scale` | None |
| `draw` | Function | `draw_filled_polygon` | None |
| `draw` | Function | `draw_filled_rectangle` | None |
| `draw` | Function | `draw_image` | None |
| `draw` | Function | `draw_line` | None |
| `draw` | Function | `draw_polygon` | None |
| `draw` | Function | `draw_rectangle` | None |
| `draw` | Function | `draw_text` | None |
| `geometry` | Component | `Point` | AS-IS (`Config`) |
| `geometry` | Component | `Position` | AS-IS (`Config`) |
| `geometry` | Component | `Rect` | AS-IS (`Config`) |
| `geometry` | Function | `get_polygon_center` | None |
| `key_points` | Component | `EdgeAnnotator` | None |
| `key_points` | Component | `KeyPoints` | AS-IS (`Data`) |
| `key_points` | Component | `VertexAnnotator` | None |
| `key_points` | Component | `VertexEllipseAnnotator` | None |
| `key_points` | Component | `VertexEllipseAreaAnnotator` | None |
| `key_points` | Component | `VertexEllipseHaloAnnotator` | None |
| `key_points` | Component | `VertexEllipseOutlineAnnotator` | None |
| `key_points` | Component | `VertexLabelAnnotator` | None |
| `metrics` | Component | `ConfusionMatrix` | None |
| `metrics` | Component | `MeanAveragePrecision` | None |
| `tracker` | Component | `ByteTrack` | Deprecated; [Tracker API Coverage](#tracker-api-coverage) |
| `utils` | Component | `FPSMonitor` | `FPSAnnotator`, `FPSMonitor` |
| `utils` | Component | `ImageSink` | AS-IS (`Utility`) |
| `utils` | Component | `ImageWindow` | `ImageWindow` |
| `utils` | Component | `VideoInfo` | AS-IS (`Data`) |
| `utils` | Component | `VideoSink` | AS-IS (`Utility`) |
| `utils` | Function | `crop_image` | None |
| `utils` | Function | `cv2_to_pillow` | None |
| `utils` | Function | `get_image_resolution_wh` | None |
| `utils` | Function | `get_video_frames_generator` | AS-IS (`Utility`) |
| `utils` | Function | `grayscale_image` | None |
| `utils` | Function | `letterbox_image` | None |
| `utils` | Function | `list_files_with_extensions` | None |
| `utils` | Function | `load_image_from_url` | None |
| `utils` | Function | `overlay_image` | None |
| `utils` | Function | `pillow_to_cv2` | None |
| `utils` | Function | `plot_image` | `PlotImage` |
| `utils` | Function | `plot_images_grid` | None |
| `utils` | Function | `process_video` | AS-IS (`Utility`) |
| `utils` | Function | `resize_image` | None |
| `utils` | Function | `scale_image` | None |
| `utils` | Function | `tint_image` | None |

Source: [roboflow/supervision](https://github.com/roboflow/supervision/blob/d2aa06b13dc7a56ddce283aa8f8bc04fbe5a5a1d/src/supervision/__init__.py)

## Tracker API Coverage

| Group   | Tracking surface            | ml-pipes correspondence                      |
|---------|-----------------------------|----------------------------------------------|
| Tracker | `BaseTracker`               | `UpdateTrackedObjects`, `ReadTrackedObjects` |
| Tracker | `BoTSORTTracker`            | `BoTSORT`                                    |
| Tracker | `ByteTrackTracker`          | `ByteTrack`                                  |
| Tracker | `OCSORTTracker`             | `OCSORT`                                     |
| Tracker | `SORTTracker`               | `SORT`                                       |
| Utils   | `CoordinatesTransformation` | None                                         |
| Utils   | `Dataset`                   | None                                         |
| Utils   | `DatasetAsset`              | None                                         |
| Utils   | `DatasetSplit`              | None                                         |
| Utils   | `HomographyTransformation`  | None                                         |
| Utils   | `IdentityTransformation`    | None                                         |
| Utils   | `MotionAwareTraceAnnotator` | None                                         |
| Utils   | `MotionEstimator`           | None                                         |
| Utils   | `download_dataset`          | None                                         |
| Utils   | `frames_from_source`        | None                                         |
| Utils   | `load_mot_file`             | None                                         |
| Utils   | `xcycsr_to_xyxy`            | None                                         |
| Utils   | `xyxy_to_xcycsr`            | None                                         |

Source: [roboflow/trackers](https://github.com/roboflow/trackers)
