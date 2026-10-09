---
title: Track Objects in Video with Supervision
description: >-
  Track objects across video frames and annotate IDs and motion with Supervision and ml-pipes, using RF-DETR, YOLO, or your preferred model.
---

# Track Objects

Track detected objects across video frames to assign persistent IDs and analyze
motion. This guide ports the Supervision object detection and segmentation
workflow to `ml-pipes`; keypoint tracking is not yet covered by this package.

Download the source video used throughout the tutorial:

```python
from supervision.assets import VideoAssets, download_assets

download_assets(VideoAssets.PEOPLE_WALKING)
```

<video controls>
    <source src="https://media.roboflow.com/supervision/video-examples/people-walking.mp4" type="video/mp4">
</video>

## Object Detection & Segmentation

### Run Inference

Run detection on each video frame, then draw boxes on the resulting scene.

=== "ml-pipes"

    ```python
    import supervision as sv

    from ml_pipes.core import Pipeline
    from ml_pipes.standard import Recall, Select, Store
    from ml_pipes.supervision import BoxAnnotator, Detections
    from ml_pipes.supervision.inference import RoboflowInference

    pipeline = Pipeline(
        [
            Store("source_frame"),
            RoboflowInference(model_id="rfdetr-small"),
            Select(0),
            Detections.FromInference(),
            Recall("source_frame", prepend=True),
            BoxAnnotator(),
        ]
    )

    def callback(frame, _: int):
        annotated_frame, _ = pipeline(frame)
        return annotated_frame

    sv.process_video(
        source_path="people-walking.mp4",
        target_path="result.mp4",
        callback=callback,
    )
    ```

=== "Supervision"

    ```python
    import numpy as np
    import supervision as sv
    from inference.models.utils import get_roboflow_model

    model = get_roboflow_model(model_id="rfdetr-small", api_key="<ROBOFLOW_API_KEY>")
    box_annotator = sv.BoxAnnotator()

    def callback(frame: np.ndarray, _: int) -> np.ndarray:
        results = model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)
        return box_annotator.annotate(frame.copy(), detections=detections)

    sv.process_video(
        source_path="people-walking.mp4",
        target_path="result.mp4",
        callback=callback,
    )
    ```

<video controls>
    <source src="https://media.roboflow.com/supervision/video-examples/how-to/track-objects/run-inference.mp4" type="video/mp4">
</video>

### Tracking

After inference, update a stateful tracker with each frame's detections. The
`ml-pipes` `ByteTrack` wrapper and the direct Supervision examples both use
`ByteTrackTracker` from the external `trackers` package.

=== "ml-pipes"

    ```{ .py hl_lines="15" }
    import supervision as sv

    from ml_pipes.core import Pipeline
    from ml_pipes.standard import Recall, Select, Store
    from ml_pipes.supervision import BoxAnnotator, Detections
    from ml_pipes.supervision.inference import RoboflowInference
    from ml_pipes.supervision.trackers import ByteTrack

    pipeline = Pipeline(
        [
            Store("source_frame"),
            RoboflowInference(model_id="rfdetr-small"),
            Select(0),
            Detections.FromInference(),
            ByteTrack(),
            Recall("source_frame", prepend=True),
            BoxAnnotator(),
        ]
    )

    def callback(frame, _: int):
        annotated_frame, _ = pipeline(frame)
        return annotated_frame

    sv.process_video(
        source_path="people-walking.mp4",
        target_path="result.mp4",
        callback=callback,
    )
    ```

=== "Supervision"

    ```{ .py hl_lines="7 13" }
    import numpy as np
    import supervision as sv
    import trackers
    from inference.models.utils import get_roboflow_model

    model = get_roboflow_model(model_id="rfdetr-small", api_key="<ROBOFLOW_API_KEY>")
    tracker = trackers.ByteTrackTracker()
    box_annotator = sv.BoxAnnotator()

    def callback(frame: np.ndarray, _: int) -> np.ndarray:
        results = model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)
        detections = tracker.update(detections)
        return box_annotator.annotate(frame.copy(), detections=detections)

    sv.process_video(
        source_path="people-walking.mp4",
        target_path="result.mp4",
        callback=callback,
    )
    ```

### Annotate Video with Tracking IDs

Add persistent IDs and class names with `LabelAnnotator` after the tracker
updates the detections.

=== "ml-pipes"

    ```{ .py hl_lines="18" }
    import supervision as sv

    from ml_pipes.core import Pipeline
    from ml_pipes.standard import Recall, Select, Store
    from ml_pipes.supervision import BoxAnnotator, Detections, LabelAnnotator
    from ml_pipes.supervision.inference import RoboflowInference
    from ml_pipes.supervision.trackers import ByteTrack

    pipeline = Pipeline(
        [
            Store("source_frame"),
            RoboflowInference(model_id="rfdetr-small"),
            Select(0),
            Detections.FromInference(),
            ByteTrack(),
            Recall("source_frame", prepend=True),
            BoxAnnotator(),
            LabelAnnotator(show_tracker_id=True, show_class=True),
        ]
    )

    def callback(frame, _: int):
        annotated_frame, _ = pipeline(frame)
        return annotated_frame

    sv.process_video(
        source_path="people-walking.mp4",
        target_path="result.mp4",
        callback=callback,
    )
    ```

=== "Supervision"

    ```{ .py hl_lines="9 16-20 24-25" }
    import numpy as np
    import supervision as sv
    import trackers
    from inference.models.utils import get_roboflow_model

    model = get_roboflow_model(model_id="rfdetr-small", api_key="<ROBOFLOW_API_KEY>")
    tracker = trackers.ByteTrackTracker()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    def callback(frame: np.ndarray, _: int) -> np.ndarray:
        results = model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)
        detections = tracker.update(detections)

        labels = [
            f"#{tracker_id} {class_name}"
            for class_name, tracker_id
            in zip(detections.data["class_name"], detections.tracker_id)
        ]

        annotated_frame = box_annotator.annotate(
            frame.copy(), detections=detections)
        return label_annotator.annotate(
            annotated_frame, detections=detections, labels=labels)

    sv.process_video(
        source_path="people-walking.mp4",
        target_path="result.mp4",
        callback=callback,
    )
    ```

<video controls>
    <source src="https://media.roboflow.com/supervision/video-examples/how-to/track-objects/annotate-video-with-tracking-ids.mp4" type="video/mp4">
</video>

### Annotate Video with Traces

Draw each track's historical path with `TraceAnnotator` after the box and
label layers.

=== "ml-pipes"

    ```{ .py hl_lines="19" }
    import supervision as sv

    from ml_pipes.core import Pipeline
    from ml_pipes.standard import Recall, Select, Store
    from ml_pipes.supervision import BoxAnnotator, Detections, LabelAnnotator, TraceAnnotator
    from ml_pipes.supervision.inference import RoboflowInference
    from ml_pipes.supervision.trackers import ByteTrack

    pipeline = Pipeline(
        [
            Store("source_frame"),
            RoboflowInference(model_id="rfdetr-small"),
            Select(0),
            Detections.FromInference(),
            ByteTrack(),
            Recall("source_frame", prepend=True),
            BoxAnnotator(),
            LabelAnnotator(show_tracker_id=True, show_class=True),
            TraceAnnotator(),
        ]
    )

    def callback(frame, _: int):
        annotated_frame, _ = pipeline(frame)
        return annotated_frame

    sv.process_video(
        source_path="people-walking.mp4",
        target_path="result.mp4",
        callback=callback,
    )
    ```

=== "Supervision"

    ```{ .py hl_lines="10 27-28" }
    import numpy as np
    import supervision as sv
    import trackers
    from inference.models.utils import get_roboflow_model

    model = get_roboflow_model(model_id="rfdetr-small", api_key="<ROBOFLOW_API_KEY>")
    tracker = trackers.ByteTrackTracker()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator()

    def callback(frame: np.ndarray, _: int) -> np.ndarray:
        results = model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)
        detections = tracker.update(detections)

        labels = [
            f"#{tracker_id} {class_name}"
            for class_name, tracker_id
            in zip(detections.data["class_name"], detections.tracker_id)
        ]

        annotated_frame = box_annotator.annotate(
            frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(
            annotated_frame, detections=detections, labels=labels)
        return trace_annotator.annotate(
            annotated_frame, detections=detections)

    sv.process_video(
        source_path="people-walking.mp4",
        target_path="result.mp4",
        callback=callback,
    )
    ```

<video controls>
    <source src="https://media.roboflow.com/supervision/video-examples/how-to/track-objects/annotate-video-with-traces.mp4" type="video/mp4">
</video>

## Smooth Tracked Detections

For detection-only video, optionally place `DetectionsSmoother` after `ByteTrack`
and before annotation. It requires tracker IDs and does not support segmentation;
`length` is the smoothing window in frames. See the [upstream smoother reference](https://supervision.roboflow.com/0.30.9/detection/tools/smoother/#supervision.detection.tools.smoother.DetectionsSmoother).

Extend the previous trace-annotation flow with the highlighted smoothing stage.
In `ml-pipes`, insert it before `Recall`, while the payload is still detections.
Reuse the same pipeline or stateful objects for the whole video.

=== "ml-pipes"

    ```{ .py hl_lines="5 16" }
    import supervision as sv

    from ml_pipes.core import Pipeline
    from ml_pipes.standard import Recall, Select, Store
    from ml_pipes.supervision import BoxAnnotator, Detections, DetectionsSmoother, LabelAnnotator, TraceAnnotator
    from ml_pipes.supervision.inference import RoboflowInference
    from ml_pipes.supervision.trackers import ByteTrack

    pipeline = Pipeline(
        [
            Store("source_frame"),
            RoboflowInference(model_id="rfdetr-small"),
            Select(0),
            Detections.FromInference(),
            ByteTrack(),
            DetectionsSmoother(length=5),
            Recall("source_frame", prepend=True),
            BoxAnnotator(),
            LabelAnnotator(show_tracker_id=True, show_class=True),
            TraceAnnotator(),
        ]
    )

    def callback(frame, _: int):
        annotated_frame, _ = pipeline(frame)
        return annotated_frame

    sv.process_video(
        source_path="people-walking.mp4",
        target_path="result.mp4",
        callback=callback,
    )
    ```

=== "Supervision"

    ```{ .py hl_lines="8 17" }
    import numpy as np
    import supervision as sv
    import trackers
    from inference.models.utils import get_roboflow_model

    model = get_roboflow_model(model_id="rfdetr-small", api_key="<ROBOFLOW_API_KEY>")
    tracker = trackers.ByteTrackTracker()
    smoother = sv.DetectionsSmoother(length=5)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator()

    def callback(frame: np.ndarray, _: int) -> np.ndarray:
        results = model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)
        detections = tracker.update(detections)
        detections = smoother.update_with_detections(detections)

        labels = [
            f"#{tracker_id} {class_name}"
            for class_name, tracker_id
            in zip(detections.data["class_name"], detections.tracker_id)
        ]

        annotated_frame = box_annotator.annotate(
            frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(
            annotated_frame, detections=detections, labels=labels)
        return trace_annotator.annotate(
            annotated_frame, detections=detections)

    sv.process_video(
        source_path="people-walking.mp4",
        target_path="result.mp4",
        callback=callback,
    )
    ```

## Frequently Asked Questions

### How do I track objects across video frames with supervision?

Create a `trackers.ByteTrackTracker()` and pass `sv.Detections` to its `update()`
method on each frame. The tracker assigns persistent IDs. Combine it with
`sv.TraceAnnotator` to visualize trajectories. The `ml-pipes` `ByteTrack` operator
wraps the same tracker.

### What should I know about ByteTrack?

ByteTrack uses low-confidence detections during association, which can improve
continuity during missed or weak detections. Use `trackers.ByteTrackTracker` for
direct tracking or the `ml-pipes` `ByteTrack` operator in a pipeline; Supervision's
built-in `sv.ByteTrack` is deprecated.

### Can I track instances instead of bounding boxes?

Yes. ByteTrack tracks bounding boxes. For instance masks, use `sv.MaskAnnotator` with the tracker IDs to color-code each tracked object consistently.

### Does ByteTrack work with any detection model?

Yes. ByteTrack is model-agnostic - it accepts any `Detections` object with bounding boxes, regardless of the supported converter or model output that produced it.

## Authors

- [Piotr Skalski](https://github.com/SkalskiP) - Computer Vision Engineer, Roboflow
- [Soumik Mandal](https://github.com/soumik12345) - ML Engineer, Roboflow
