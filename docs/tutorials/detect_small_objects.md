---
title: Detect Small Objects with Supervision
comments: true
description: >-
  Detect small objects using tiled inference with Supervision and ml-pipes. Compare full-image predictions with sliced detection and stitch the results.
authors:
  - name: Piotr Skalski
    role: Computer Vision Engineer, Roboflow
    github: https://github.com/SkalskiP
date_modified: 2026-04-22
---

# Detect Small Objects

This guide shows how to detect small objects with [Inference](https://github.com/roboflow/inference) using [`InferenceSlicer`](https://supervision.roboflow.com/latest/detection/tools/inference_slicer/#supervision.detection.tools.inference_slicer.InferenceSlicer).

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
            RoboflowInference(model_id="yolov8x-640"),
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

    model = get_model(model_id="yolov8x-640")
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

Modifying the input resolution of images before detection can enhance small object identification at the cost of processing speed and increased memory usage. This method is less effective for ultra-high-resolution images (4K and above).

=== "ml-pipes"

    ```{ .py hl_lines="13" }
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
            RoboflowInference(model_id="yolov8x-1280"),
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

    ```{ .py hl_lines="5" }
    import cv2
    import supervision as sv
    from inference import get_model

    model = get_model(model_id="yolov8x-1280")
    image = cv2.imread("<SOURCE_IMAGE_PATH>")
    results = model.infer(image)[0]
    detections = sv.Detections.from_inference(results)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = box_annotator.annotate(
        scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)
    ```


![detection-with-high-input-resolution](https://media.roboflow.com/supervision_detect_small_objects_example_2.png)

## Inference Slicer

[`InferenceSlicer`](https://supervision.roboflow.com/latest/detection/tools/inference_slicer/#supervision.detection.tools.inference_slicer.InferenceSlicer) processes high-resolution images by dividing them into smaller segments, detecting objects within each, and aggregating the results.

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
            Tile(slice_wh=(320, 320), overlap_wh=(80, 80)),
            Store("tile_rects", source=1),
            Pick(0),
            Scatter(max_concurrency=4),
            RoboflowInference(model_id="yolov8x-640"),
            Select(0),
            Detections.FromInference(),
            Gather(),
            Recall("tile_rects"),
            Detections.Stitch(),
            Detections.NMM(iou_threshold=0.5),
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

    model = get_model(model_id="yolov8x-640")
    image = cv2.imread("<SOURCE_IMAGE_PATH>")

    def callback(image_slice: np.ndarray) -> sv.Detections:
        results = model.infer(image_slice)[0]
        return sv.Detections.from_inference(results)

    slicer = sv.InferenceSlicer(callback = callback)
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

[`InferenceSlicer`](https://supervision.roboflow.com/latest/detection/tools/inference_slicer/#supervision.detection.tools.inference_slicer.InferenceSlicer) can perform segmentation tasks too.

=== "ml-pipes"

    ```{ .py hl_lines="16 28" }
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
            Tile(slice_wh=(320, 320), overlap_wh=(80, 80)),
            Store("tile_rects", source=1),
            Pick(0),
            Scatter(max_concurrency=4),
            RoboflowInference(model_id="yolov8x-seg-640"),
            Select(0),
            Detections.FromInference(),
            Gather(),
            Recall("tile_rects"),
            Detections.Stitch(),
            Detections.NMM(iou_threshold=0.5),
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

    model = get_model(model_id="yolov8x-seg-640")
    image = cv2.imread("<SOURCE_IMAGE_PATH>")

    def callback(image_slice: np.ndarray) -> sv.Detections:
        results = model.infer(image_slice)[0]
        return sv.Detections.from_inference(results)

    slicer = sv.InferenceSlicer(callback = callback)
    detections = slicer(image)

    mask_annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = mask_annotator.annotate(
        scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)
    ```


![detection-with-inference-slicer](https://media.roboflow.com/supervision-docs/inference-slicer-segmentation-example.png)

## Frequently Asked Questions

### How do I detect small objects with supervision?

Use `sv.InferenceSlicer` to split a high-resolution image into overlapping tiles, run detection on each tile, and merge results with non-maximum suppression. This dramatically improves recall for tiny targets.

### What overlap should I use between tiles?

`InferenceSlicer` takes overlap in pixels via `overlap_wh`, not as a percentage. The default is `100` pixels in both directions. Increase `overlap_wh` when objects are close to the tile size or often appear on tile boundaries, and decrease it when speed is more important.

### Can I use InferenceSlicer with any detection model?

Yes. Wrap any model or converter path that can produce `sv.Detections` in a callback, pass that callback to `sv.InferenceSlicer(callback=...)`, and then call the slicer with your image.

## Author

- [Piotr Skalski](https://github.com/SkalskiP) — Computer Vision Engineer, Roboflow
