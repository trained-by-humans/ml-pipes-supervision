---
title: Filter Detections with Supervision
description: >-
  Filter detections from RF-DETR, YOLO, or your preferred model by class, confidence, geometry, or zones with Supervision and ml-pipes.
---

# Filter Detections

Filter detections by class, confidence, size, or location to keep the results
relevant to your task. Each example compares an `ml-pipes` operator with direct
filtering of Supervision `Detections`.

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
            Detections.Filter(lambda detections: detections.class_id == 1),
        ]
    )

    filtered_detections = pipeline(detections)
    ```

=== "Supervision"

    ```python
    detections = detections[detections.class_id == 1]
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

RF-DETR's COCO IDs: `1` (person). Class IDs are model-specific.

### by set of classes

Allows you to select detections that belong only to a selected set of classes.

=== "ml-pipes"

    ```{ .py hl_lines="12-14" }
    import numpy as np

    selected_classes = [1, 3, 4]

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

    selected_classes = [1, 3, 4]
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

RF-DETR's COCO IDs: `1` (person), `3` (car), and `4` (motorcycle). Class IDs are model-specific.

### by confidence

Select detections by confidence, for example those above a chosen threshold.

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

Filter detections by their size. In the example below, detections that are too
small are removed. `detections.area` uses mask area when available, otherwise
oriented-box area, then axis-aligned box area.
Use `detections.box_area` when you specifically need the axis-aligned envelope;
see the [upstream area reference](https://supervision.roboflow.com/0.30.9/detection/core/#supervision.detection.core.Detections.area).

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

Select detections based on their bounding box dimensions or coordinates.
For aspect-ratio filtering, use `detections.box_aspect_ratio` instead.

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

Use `Detections` with `PolygonZone` to select objects inside a zone. The
example filters out detections in the lower part of the image.
See [Count in Zone](count_in_zone.md) for polygon setup.

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

### remove duplicate detections

Non-maximum suppression (NMS) removes overlapping duplicate predictions.
It uses masks when present; otherwise it uses oriented boxes from `detections.data["xyxyxyxy"]`, falling
back to axis-aligned `xyxy` boxes. See the [upstream NMS reference](https://supervision.roboflow.com/0.30.9/detection/core/#supervision.detection.core.Detections.with_nms).

## Author

- [Piotr Skalski](https://github.com/SkalskiP) — Computer Vision Engineer, Roboflow
