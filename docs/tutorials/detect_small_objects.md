---
title: Detect Small Objects with Supervision
description: >-
  Detect small objects with tiled inference using RF-DETR, YOLO, or your preferred model in Supervision and ml-pipes.
---

# Detect Small Objects

Detect small objects with RF-DETR, YOLO, or your preferred model using tiled
inference in Supervision and `ml-pipes`. The examples use
[RF-DETR](https://github.com/roboflow/rf-detr) through
[Inference](https://github.com/roboflow/inference), comparing full-image detection
with [`InferenceSlicer`](https://supervision.roboflow.com/latest/detection/tools/inference_slicer/#supervision.detection.tools.inference_slicer.InferenceSlicer).

<video controls>
    <source src="https://media.roboflow.com/supervision_detect_small_objects_example.mp4" type="video/mp4">
</video>

## Baseline Detection

Small object detection in high-resolution images presents challenges due to the objects' size relative to the image resolution.

Running a standard detection model on the full image establishes a baseline for comparison. Load your chosen model, pass the image through it, and convert the results into a `Detections` object. This baseline reveals how many small objects the model misses at native resolution, motivating the sliced inference approach shown later.

=== "ml-pipes"

    ```python
    from ml_pipes.core import Pipeline
    from ml_pipes.standard import Recall, Select, Store
    from ml_pipes.supervision import BoxAnnotator, Detections, ImageToArray, LabelAnnotator, PlotImage
    from ml_pipes.supervision.inference import RoboflowInference
    from ml_pipes.vision import Decode, LoadFile

    pipeline = Pipeline(
        [
            LoadFile(),
            Decode(),
            ImageToArray(),
            Store("source_image"),
            RoboflowInference(model_id="rfdetr-medium"),
            Select(0),
            Detections.FromInference(),
            Recall("source_image", prepend=True),
            BoxAnnotator(),
            LabelAnnotator(show_class=True, show_confidence=True),
            PlotImage(at=0),
        ]
    )

    annotated_image, detections = pipeline("<SOURCE_IMAGE_PATH>")
    ```

=== "Supervision"

    ```python
    import cv2
    import supervision as sv
    from inference import get_model

    model = get_model(model_id="rfdetr-medium")
    image = cv2.imread("<SOURCE_IMAGE_PATH>")
    results = model.infer(image)[0]
    detections = sv.Detections.from_inference(results)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = box_annotator.annotate(
        scene=image,
        detections=detections,
    )
    annotated_image = label_annotator.annotate(
        scene=annotated_image,
        detections=detections,
    )
    ```


![basic-detection](https://media.roboflow.com/supervision_detect_small_objects_example_1.png)

## Input Resolution

Higher model input resolution can improve small-object detection at a speed
and memory cost. This is model-specific tuning; see the
[upstream input-resolution comparison](https://supervision.roboflow.com/0.30.9/how_to/detect_small_objects/#input-resolution).

## Inference Slicer

[`InferenceSlicer`](https://supervision.roboflow.com/0.30.9/detection/tools/inference_slicer/#supervision.detection.tools.inference_slicer.InferenceSlicer) processes high-resolution images by dividing them into smaller segments, detecting objects within each, and aggregating the results.

The Supervision examples use the slicer's defaults: 640×640-pixel tiles,
100-pixel overlap, and non-maximum suppression (NMS) at an IoU threshold of 0.5.
The `ml-pipes` examples set those values explicitly.

`overlap_wh` is measured in pixels, not percentages. Increase it when objects
span tile boundaries, at the cost of more inference. The upstream reference
also describes alternative overlap filters.

<video controls>
    <source src="https://media.roboflow.com/supervision_detect_small_objects_example_2.mp4" type="video/mp4">
</video>

=== "ml-pipes"

    ```{ .py hl_lines="12-15 19-22" }
    from ml_pipes.core import Pipeline
    from ml_pipes.standard import Gather, Pick, Recall, Scatter, Select, Store
    from ml_pipes.supervision import BoxAnnotator, Detections, ImageToArray, LabelAnnotator, PlotImage
    from ml_pipes.supervision.inference import RoboflowInference
    from ml_pipes.vision import Decode, LoadFile, Tile

    pipeline = Pipeline(
        [
            LoadFile(),
            Decode(),
            Store("source_image"),
            Tile(slice_wh=(640, 640), overlap_wh=(100, 100)),
            Store("tile_rects", source=1),
            Pick(0),
            Scatter(max_concurrency=4),
            RoboflowInference(model_id="rfdetr-medium"),
            Select(0),
            Detections.FromInference(),
            Gather(),
            Recall("tile_rects"),
            Detections.Stitch(),
            Detections.NMS(threshold=0.5),
            Store("detections"),
            Recall("source_image"),
            Pick(1),
            ImageToArray(),
            Recall("detections"),
            BoxAnnotator(),
            LabelAnnotator(show_class=True, show_confidence=True),
            PlotImage(at=0),
        ]
    )

    annotated_image, detections = pipeline("<SOURCE_IMAGE_PATH>")
    ```

=== "Supervision"

    ```{ .py hl_lines="9-14" }
    import cv2
    import numpy as np
    import supervision as sv
    from inference import get_model

    model = get_model(model_id="rfdetr-medium")
    image = cv2.imread("<SOURCE_IMAGE_PATH>")

    def callback(image_slice: np.ndarray) -> sv.Detections:
        results = model.infer(image_slice)[0]
        return sv.Detections.from_inference(results)

    slicer = sv.InferenceSlicer(callback=callback)
    detections = slicer(image)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = box_annotator.annotate(
        scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)
    ```


![detection-with-inference-slicer](https://media.roboflow.com/supervision_detect_small_objects_example_3.png)

## Small Object Segmentation

`InferenceSlicer` can perform segmentation tasks too, using the same tile and
NMS settings as above. To opt into [compact masks](detect_and_annotate.md#compact-masks),
set `compact_masks=True` in the conversion stage or callback.

=== "ml-pipes"

    ```{ .py hl_lines="12 16 22 28" }
    from ml_pipes.core import Pipeline
    from ml_pipes.standard import Gather, Pick, Recall, Scatter, Select, Store
    from ml_pipes.supervision import Detections, ImageToArray, LabelAnnotator, MaskAnnotator, PlotImage
    from ml_pipes.supervision.inference import RoboflowInference
    from ml_pipes.vision import Decode, LoadFile, Tile

    pipeline = Pipeline(
        [
            LoadFile(),
            Decode(),
            Store("source_image"),
            Tile(slice_wh=(640, 640), overlap_wh=(100, 100)),
            Store("tile_rects", source=1),
            Pick(0),
            Scatter(max_concurrency=4),
            RoboflowInference(model_id="rfdetr-seg-medium"),
            Select(0),
            Detections.FromInference(),
            Gather(),
            Recall("tile_rects"),
            Detections.Stitch(),
            Detections.NMS(threshold=0.5),
            Store("detections"),
            Recall("source_image"),
            Pick(1),
            ImageToArray(),
            Recall("detections"),
            MaskAnnotator(),
            LabelAnnotator(show_class=True, show_confidence=True),
            PlotImage(at=0),
        ]
    )

    annotated_image, detections = pipeline("<SOURCE_IMAGE_PATH>")
    ```

=== "Supervision"

    ```{ .py hl_lines="6 16 19-20" }
    import cv2
    import numpy as np
    import supervision as sv
    from inference import get_model

    model = get_model(model_id="rfdetr-seg-medium")
    image = cv2.imread("<SOURCE_IMAGE_PATH>")

    def callback(image_slice: np.ndarray) -> sv.Detections:
        results = model.infer(image_slice)[0]
        return sv.Detections.from_inference(results)

    slicer = sv.InferenceSlicer(callback=callback)
    detections = slicer(image)

    mask_annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = mask_annotator.annotate(
        scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)
    ```


![detection-with-inference-slicer](https://media.roboflow.com/supervision-docs/inference-slicer-segmentation-example.png)

## Author

- [Piotr Skalski](https://github.com/SkalskiP) — Computer Vision Engineer, Roboflow
