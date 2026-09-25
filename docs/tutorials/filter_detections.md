---
comments: true
description: Filter and query detection results by class, confidence, or spatial overlap using supervision's Detections API — clean predictions in one line.
authors:
  - name: Piotr Skalski
    role: Computer Vision Engineer, Roboflow
    github: https://github.com/SkalskiP
date_modified: 2026-04-22
---

# Filter Detections

The advanced filtering capabilities of the `Detections` class offer users a versatile and efficient way to narrow down and refine object detections. This section outlines various filtering methods, including filtering by specific class or a set of classes, confidence, object area, bounding box area, relative area, box dimensions, and designated zones. Each method is demonstrated with concise code examples to provide users with a clear understanding of how to implement the filters in their applications.

The `ml-pipes` examples abbreviate their shared loading and inference prefix as
`...`; see [Detect and Annotate](detect_and_annotate.md) for the complete
detection pipeline.

### by specific class

Allows you to select detections that belong only to one selected class.

=== "ml-pipes"

    ```{ .py hl_lines="8" }
    from ml_pipes.core import Pipeline
    from ml_pipes.supervision import Detections

    pipeline = Pipeline(
        [
            ...,
            Detections.FromInference(),
            Detections.Filter(lambda detections: detections.class_id == 0),
        ]
    )

    filtered_detections = pipeline(detections)
    ```

=== "Supervision"

    ```python
    detections = detections[detections.class_id == 0]
    ```

<div class="filter-comparison" markdown>
<figure markdown>
![Before](https://media.roboflow.com/open-source/supervision/supervision-detection-original.png)
<figcaption>Before</figcaption>
</figure>
<figure markdown>
![After filtering by class](https://media.roboflow.com/open-source/supervision/supervision-detection-by-specific-class.png)
<figcaption>After</figcaption>
</figure>
</div>

### by set of classes

Allows you to select detections that belong only to selected set of classes.

=== "ml-pipes"

    ```{ .py hl_lines="12-14" }
    import numpy as np

    selected_classes = [0, 2, 3]

    from ml_pipes.core import Pipeline
    from ml_pipes.supervision import Detections

    pipeline = Pipeline(
        [
            ...,
            Detections.FromInference(),
            Detections.Filter(
                lambda detections: np.isin(detections.class_id, selected_classes)
            ),
        ]
    )

    filtered_detections = pipeline(detections)
    ```

=== "Supervision"

    ```python
    import numpy as np

    selected_classes = [0, 2, 3]
    detections = detections[np.isin(detections.class_id, selected_classes)]
    ```

<div class="filter-comparison" markdown>
<figure markdown>
![Before](https://media.roboflow.com/open-source/supervision/supervision-detection-original.png)
<figcaption>Before</figcaption>
</figure>
<figure markdown>
![After filtering by classes](https://media.roboflow.com/open-source/supervision/supervision-detection-by-set-of-classes.png)
<figcaption>After</figcaption>
</figure>
</div>

### by confidence

Allows you to select detections with specific confidence value, for example higher than selected threshold.

=== "ml-pipes"

    ```{ .py hl_lines="8" }
    from ml_pipes.core import Pipeline
    from ml_pipes.supervision import Detections

    pipeline = Pipeline(
        [
            ...,
            Detections.FromInference(),
            Detections.Filter(lambda detections: detections.confidence > 0.5),
        ]
    )

    filtered_detections = pipeline(detections)
    ```

=== "Supervision"

    ```python
    detections = detections[detections.confidence > 0.5]
    ```

<div class="filter-comparison" markdown>
<figure markdown>
![Before](https://media.roboflow.com/open-source/supervision/supervision-detection-original.png)
<figcaption>Before</figcaption>
</figure>
<figure markdown>
![After filtering by confidence](https://media.roboflow.com/open-source/supervision/supervision-detection-by-confidence.png)
<figcaption>After</figcaption>
</figure>
</div>

### by area

Allows you to select detections based on their size. We define the area as the number of pixels occupied by the detection in the image. In the example below, we have sifted out the detections that are too small.

=== "ml-pipes"

    ```{ .py hl_lines="8" }
    from ml_pipes.core import Pipeline
    from ml_pipes.supervision import Detections

    pipeline = Pipeline(
        [
            ...,
            Detections.FromInference(),
            Detections.Filter(lambda detections: detections.area > 1000),
        ]
    )

    filtered_detections = pipeline(detections)
    ```

=== "Supervision"

    ```python
    detections = detections[detections.area > 1000]
    ```

<div class="filter-comparison" markdown>
<figure markdown>
![Before](https://media.roboflow.com/open-source/supervision/supervision-detection-original.png)
<figcaption>Before</figcaption>
</figure>
<figure markdown>
![After filtering by area](https://media.roboflow.com/open-source/supervision/supervision-detection-by-area.png)
<figcaption>After</figcaption>
</figure>
</div>

### by relative area

Allows you to select detections based on their size in relation to the size of whole image. Sometimes the concept of detection size changes depending on the image. Detection occupying 10000 square px can be large on a 1280x720 image but small on a 3840x2160 image. In such cases, we can filter out detections based on the percentage of the image area occupied by them. In the example below, we remove too large detections.

=== "ml-pipes"

    ```{ .py hl_lines="11" }
    from ml_pipes.core import Pipeline
    from ml_pipes.supervision import Detections

    height, width = image.shape[:2]
    image_area = height * width

    pipeline = Pipeline(
        [
            ...,
            Detections.FromInference(),
            Detections.Filter(lambda detections: (detections.area / image_area) < 0.8),
        ]
    )

    filtered_detections = pipeline(detections)
    ```

=== "Supervision"

    ```python
    height, width = image.shape[:2]
    image_area = height * width
    detections = detections[(detections.area / image_area) < 0.8]
    ```

<div class="filter-comparison" markdown>
<figure markdown>
![Before](https://media.roboflow.com/open-source/supervision/supervision-detection-original.png)
<figcaption>Before</figcaption>
</figure>
<figure markdown>
![After filtering by relative area](https://media.roboflow.com/open-source/supervision/supervision-detection-by-relative-area.png?updatedAt=1683207183434)
<figcaption>After</figcaption>
</figure>
</div>

### by box dimensions

Allows you to select detections based on their dimensions. The size of the bounding box, as well as its coordinates, can be criteria for rejecting detection. Implementing such filtering requires a bit of custom code but is relatively simple and fast.

=== "ml-pipes"

    ```{ .py hl_lines="3-6 12" }
    from ml_pipes.core import Pipeline
    from ml_pipes.supervision import Detections
    def filter_by_dimensions(detections):
        width = detections.xyxy[:, 2] - detections.xyxy[:, 0]
        height = detections.xyxy[:, 3] - detections.xyxy[:, 1]
        return (width > 200) & (height > 200)

    pipeline = Pipeline(
        [
            ...,
            Detections.FromInference(),
            Detections.Filter(filter_by_dimensions),
        ]
    )

    filtered_detections = pipeline(detections)
    ```

=== "Supervision"

    ```python
    width = detections.xyxy[:, 2] - detections.xyxy[:, 0]
    height = detections.xyxy[:, 3] - detections.xyxy[:, 1]
    detections = detections[(width > 200) & (height > 200)]
    ```

<div class="filter-comparison" markdown>
<figure markdown>
![Before](https://media.roboflow.com/open-source/supervision/supervision-detection-original.png)
<figcaption>Before</figcaption>
</figure>
<figure markdown>
![After filtering by dimensions](https://media.roboflow.com/open-source/supervision/supervision-detection-by-box-dimensions.png)
<figcaption>After</figcaption>
</figure>
</div>

### by `PolygonZone`

Allows you to use `Detections` in combination with `PolygonZone` to weed out bounding boxes that are in and out of the zone. In the example below you can see how to filter out all detections located in the lower part of the image.

=== "ml-pipes"

    ```{ .py hl_lines="11" }
    import supervision as sv

    from ml_pipes.core import Pipeline
    from ml_pipes.supervision import Detections, TriggerZone

    zone = sv.PolygonZone(...)
    pipeline = Pipeline(
        [
            ...,
            Detections.FromInference(),
            TriggerZone(zone),
        ]
    )

    filtered_detections = pipeline(detections)
    ```

=== "Supervision"

    ```python
    import supervision as sv

    zone = sv.PolygonZone(...)
    detections = detections[zone.trigger(detections=detections)]
    ```

<div class="filter-comparison" markdown>
<figure markdown>
![Before](https://media.roboflow.com/open-source/supervision/supervision-detection-original.png)
<figcaption>Before</figcaption>
</figure>
<figure markdown>
![After filtering by polygon zone](https://media.roboflow.com/open-source/supervision/supervision-detection-by-polygon-zone.png?updatedAt=1683211380445)
<figcaption>After</figcaption>
</figure>
</div>

### by mixed conditions

Combine an ordinary `Detections.Filter` condition with `TriggerZone`. The
pipeline applies the same confidence and zone conditions as the direct
Supervision code.

=== "ml-pipes"

    ```{ .py hl_lines="11-12" }
    import supervision as sv

    from ml_pipes.core import Pipeline
    from ml_pipes.supervision import Detections, TriggerZone

    zone = sv.PolygonZone(...)
    pipeline = Pipeline(
        [
            ...,
            Detections.FromInference(),
            Detections.Filter(lambda detections: detections.confidence > 0.7),
            TriggerZone(zone),
        ]
    )

    filtered_detections = pipeline(detections)
    ```

=== "Supervision"

    ```python
    import supervision as sv

    zone = sv.PolygonZone(...)
    mask = zone.trigger(detections=detections)
    detections = detections[(detections.confidence > 0.7) & mask]
    ```

<div class="filter-comparison" markdown>
<figure markdown>
![Before](https://media.roboflow.com/open-source/supervision/supervision-detection-original.png)
<figcaption>Before</figcaption>
</figure>
<figure markdown>
![After filtering by mixed conditions](https://media.roboflow.com/open-source/supervision/supervision-detection-by-mixed-conditions.png)
<figcaption>After</figcaption>
</figure>
</div>

## Frequently Asked Questions

### How do I filter detections by class in supervision?

Use NumPy-style boolean indexing: `detections[detections.class_id == 0]` for class 0. Combine with `&` or `|` for multiple conditions.

### How do I filter by confidence threshold?

`detections[detections.confidence > 0.5]` returns only detections above the threshold. Chain with class filters for precise results.

### How do I filter by bounding box area?

`detections[detections.area > 1000]` filters by pixel area. If masks are present, `detections.area` uses mask area; otherwise, if oriented-box coordinates are present, it uses oriented polygon area; all remaining detections use bounding box area from `xyxy`. Use `detections.box_area` when you specifically need axis-aligned bounding box area.

### Can I filter by box aspect ratio or dimensions?

Yes. Use `detections.box_aspect_ratio` for aspect ratio filtering. If you need explicit box dimensions, compute them from `detections.xyxy` as `width = detections.xyxy[:, 2] - detections.xyxy[:, 0]` and `height = detections.xyxy[:, 3] - detections.xyxy[:, 1]`.

### How do I remove duplicate detections (NMS) from my results?

Use `detections.with_nms(threshold=0.5)` — it applies non-maximum suppression on the `xyxy` boxes.

## Author

- [Piotr Skalski](https://github.com/SkalskiP) — Computer Vision Engineer, Roboflow
