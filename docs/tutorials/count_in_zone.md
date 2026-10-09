---
title: Count Objects in Zones with Supervision
description: >-
  Count objects in polygon zones with Supervision and ml-pipes using RF-DETR, YOLO, or your preferred model.
---

# Count in Zone

With Supervision, you can count objects inside a zone in an image or video, this guide counts cars in a traffic video.
[upstream notebook](https://github.com/roboflow/notebooks/blob/main/notebooks/how-to-use-polygonzone-annotate-and-supervision.ipynb).

Start by downloading the source video:

```python
from supervision.assets import download_assets, VideoAssets

download_assets(VideoAssets.VEHICLES_2)
```

## Initialize a Model and Load Video

Use [RF-DETR](https://github.com/roboflow/rf-detr) through Roboflow Inference
with its pretrained COCO checkpoint, then load the source video.

The model processes each frame during inference. A shared color palette ensures
consistent zone coloring throughout the output video.

```python
import numpy as np
import supervision as sv

from inference import get_model
from supervision.assets import VideoAssets, download_assets

model = get_model(model_id="rfdetr-medium")

VIDEO = download_assets(VideoAssets.VEHICLES_2)

colors = sv.ColorPalette.DEFAULT
```

## Calculate Coordinates

Define polygons in source-video pixel coordinates. These coordinates match the sample video:

```python
polygons = [
    np.array([[718, 595], [927, 592], [851, 1062], [42, 1059]]),
    np.array([[987, 595], [1199, 595], [1893, 1056], [1015, 1062]]),
]
```

For another video, draw them with the [PolygonZone web utility](https://roboflow.github.io/polygonzone/);
see the [upstream walkthrough](https://supervision.roboflow.com/0.30.9/how_to/count_in_zone/#calculate-coordinates).

## Define Zones

Create a `PolygonZone` for each polygon, pairing it with a
`PolygonZoneAnnotator` for the zone overlay and a `BoxAnnotator` for detection
boxes. Each zone determines which detections fall inside its boundaries.

Note that counts represent per-frame occupancy, not unique visitors across all frames.

=== "ml-pipes"

    ```python
    import supervision as sv

    from ml_pipes.supervision import BoxAnnotator, PolygonZoneAnnotator

    zones = [sv.PolygonZone(polygon=polygon) for polygon in polygons]
    zone_annotators = [
        PolygonZoneAnnotator(
            zone=zone,
            color=colors.by_idx(index),
            thickness=4,
            text_thickness=8,
            text_scale=4,
        )
        for index, zone in enumerate(zones)
    ]
    box_annotators = [
        BoxAnnotator(
            color=colors.by_idx(index),
            thickness=4,
        )
        for index in range(len(polygons))
    ]
    ```

=== "Supervision"

    ```python
    zones = [sv.PolygonZone(polygon=polygon) for polygon in polygons]
    zone_annotators = [
        sv.PolygonZoneAnnotator(
            zone=zone,
            color=colors.by_idx(index),
            thickness=4,
            text_thickness=8,
            text_scale=4,
        )
        for index, zone in enumerate(zones)
    ]
    box_annotators = [
        sv.BoxAnnotator(
            color=colors.by_idx(index),
            thickness=4,
        )
        for index in range(len(polygons))
    ]
    ```

## Run Inference

Detect objects in each frame and filter the detections by zone. Annotation
needs both the source frame and its detections to draw the boxes and zone
counts on the image.

=== "ml-pipes"

    ```python
    from ml_pipes.core import Pipeline
    from ml_pipes.standard import Pick, Recall, Select, Store
    from ml_pipes.supervision import Detections, TriggerZone
    from ml_pipes.supervision.inference import RoboflowInference

    pipeline = Pipeline(
        [
            Store("source_frame"),
            RoboflowInference(model_id="rfdetr-medium"),
            Select(0),
            Detections.FromInference(),
            Store("detections"),
            TriggerZone(zones[0]),
            Recall("source_frame", prepend=True),
            box_annotators[0],
            zone_annotators[0],
            Pick(0),
            Store("first_zone_scene"),
            Recall("detections", prepend=True),
            Select(0),
            TriggerZone(zones[1]),
            Recall("first_zone_scene", prepend=True),
            box_annotators[1],
            zone_annotators[1],
            Pick(0),
        ]
    )

    def process_frame(frame: np.ndarray, _: int) -> np.ndarray:
        return pipeline(frame)


    sv.process_video(source_path=VIDEO, target_path="result.mp4", callback=process_frame)
    ```

=== "Supervision"

    ```python
    def process_frame(frame: np.ndarray, i) -> np.ndarray:
        results = model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)

        for zone, zone_annotator, box_annotator in zip(
            zones, zone_annotators, box_annotators
        ):
            mask = zone.trigger(detections=detections)
            detections_filtered = detections[mask]
            frame = box_annotator.annotate(scene=frame, detections=detections_filtered)
            frame = zone_annotator.annotate(scene=frame)

        return frame


    sv.process_video(source_path=VIDEO, target_path="result.mp4", callback=process_frame)
    ```

Here is an example of output:

<video width="100%" loop muted autoplay>
  <source src="https://blog.roboflow.com/content/media/2023/03/trim-counting.mp4" type="video/mp4">
</video>

## Author

- [Piotr Skalski](https://github.com/SkalskiP) — Computer Vision Engineer, Roboflow
