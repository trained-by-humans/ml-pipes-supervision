# Count Objects Crossing a Line

Pipeline source: `examples/run_count_objects_crossing_line.py`

Count vehicles as they cross a horizontal line. The pipeline adds one concern at a time: detection, visualization, line configuration, tracking, crossing state, then video output.

## Install dependencies

```bash
python -m pip install ml-pipes-supervision
```

## Download the source video

Use Supervision's vehicles asset as the input video. The line coordinates below are defined for its original `3840 x 2160` resolution.

```python
import supervision as sv
from supervision.assets import VideoAssets, download_assets

source_path = download_assets(VideoAssets.VEHICLES)
video_info = sv.VideoInfo.from_video_path(source_path)
```

<video controls>
    <source src="https://storage.googleapis.com/com-roboflow-marketing/supervision/cookbooks/vehicles-1280x720.mp4" type="video/mp4">
</video>

## Run object detection

Run the model and convert its first response to `sv.Detections`. The resulting detections become the input to the later pipeline stages.

```{ .py hl_lines="10-12" }
from ml_pipes.core import Pipeline
from ml_pipes.standard import Select
from ml_pipes.supervision import Detections
from ml_pipes.supervision.inference import RoboflowInference

model_id = "yolo11x-640"

detection_pipeline = Pipeline(
    [
        RoboflowInference(model_id=model_id),
        Select(0),
        Detections.FromInference(),
    ],
    auto_validate=True,
)
```

## Visualizing Detection

Detections do not contain the source image. Store and recall the frame when drawing its boxes and labels.

```{ .py hl_lines="1-2 6 10-13" }
from ml_pipes.standard import Pick, Recall, Store
from ml_pipes.supervision import BoxAnnotator, LabelAnnotator

visualization_pipeline = Pipeline(
    [
        Store("source_frame"),
        RoboflowInference(model_id=model_id),
        Select(0),
        Detections.FromInference(),
        Recall("source_frame", prepend=True),
        BoxAnnotator(thickness=6),
        LabelAnnotator(text_thickness=4, text_scale=2, show_class=True, show_confidence=True),
        Pick(0),
    ],
    auto_validate=True,
)
```

![Vehicle detections with labels](../assets/count_objects_crossing_line/labels.png)

## Define the line position

`LineZone` holds the crossing state. Configure its endpoints in source-video coordinates and add its annotator directly to the visualization pipeline.

```{ .py hl_lines="1 3-5 16" }
from ml_pipes.supervision import LineZoneAnnotator

line_start = sv.Point(0, 1500)
line_end = sv.Point(3840, 1500)
line_zone = sv.LineZone(start=line_start, end=line_end)

line_pipeline = Pipeline(
    [
        Store("source_frame"),
        RoboflowInference(model_id=model_id),
        Select(0),
        Detections.FromInference(),
        Recall("source_frame", prepend=True),
        BoxAnnotator(thickness=6),
        LabelAnnotator(text_thickness=4, text_scale=2, show_class=True, show_confidence=True),
        LineZoneAnnotator(line_zone=line_zone, thickness=4, text_thickness=4, text_scale=2),
        Pick(0),
    ],
    auto_validate=True,
)
```

![Line zone on the source frame](../assets/count_objects_crossing_line/line_zone.png)

## Track objects crossing

Counting requires stable object identities. Add `ByteTrack`, trigger the line zone with tracked detections, draw each object's path, and show its tracker ID in the label.

```{ .py hl_lines="1-2 10-11 13 18" }
from ml_pipes.supervision import TraceAnnotator, TriggerLineZone
from ml_pipes.supervision.trackers import ByteTrack

frame_pipeline = Pipeline(
    [
        Store("source_frame"),
        RoboflowInference(model_id=model_id),
        Select(0),
        Detections.FromInference(),
        ByteTrack(),
        TriggerLineZone(line_zone),
        Recall("source_frame", prepend=True),
        TraceAnnotator(thickness=4),
        BoxAnnotator(thickness=4),
        LabelAnnotator(
            text_thickness=4,
            text_scale=2,
            show_tracker_id=True,
            show_class=True,
            show_confidence=True,
        ),
        LineZoneAnnotator(line_zone=line_zone, thickness=4, text_thickness=4, text_scale=2),
        Pick(0),
    ],
    auto_validate=True,
)
```

![Tracked vehicles and line-crossing counts](../assets/count_objects_crossing_line/tracked_crossings.jpg)

## Process Video

Pass the completed frame pipeline to `sv.process_video` to write the annotated result.

```python
sv.process_video(
    source_path=source_path,
    target_path="count-objects-crossing-the-line-result.mp4",
    callback=lambda frame, _: frame_pipeline(frame),
)
```

<video controls>
    <source src="https://storage.googleapis.com/com-roboflow-marketing/supervision/cookbooks/count-objects-crossing-the-line-result-1280x720.mp4" type="video/mp4">
</video>

## Inspect the Pipeline

Use `Pipeline.inspect()` to capture the value at every operator boundary
without changing the pipeline's final output. The inspection renderer turns
that captured run into a shareable HTML report.

```python
from ml_pipes.inspection import PipelineInspector

inspection = frame_pipeline.inspect(representative_frame)
PipelineInspector().save(inspection, "inspection.html")
```

The report below captures the complete line-crossing pipeline on a frame from
the source video.

[![Line-crossing pipeline inspection](../assets/count_objects_crossing_line/inspection.png)](../assets/count_objects_crossing_line/inspection.html)

*Click the image to open the interactive inspection report.*
