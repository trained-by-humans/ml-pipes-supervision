---
title: Save Detections to CSV or JSON with Supervision
description: Save detections from RF-DETR, YOLO, or your preferred model to CSV or JSON with Supervision and ml-pipes.
---

# Save Detections

Save detections to CSV or JSON for offline processing. The examples run
video inference with [Inference](https://github.com/roboflow/inference) and
export the results with Supervision's CSV and JSON sinks.

## Run Detection

Run your detector on each video frame and convert its predictions to
Supervision `Detections`. See [Detect and Annotate](detect_and_annotate.md)
for detection and conversion examples.

=== "ml-pipes"

    ```python
    import supervision as sv

    from ml_pipes.core import Pipeline
    from ml_pipes.standard import Select
    from ml_pipes.supervision import Detections
    from ml_pipes.supervision.inference import RoboflowInference

    pipeline = Pipeline(
        [
            RoboflowInference(model_id="rfdetr-small"),
            Select(0),
            Detections.FromInference(),
        ]
    )

    frames_generator = sv.get_video_frames_generator("<SOURCE_VIDEO_PATH>")

    for frame in frames_generator:
        detections = pipeline(frame)
    ```

=== "Supervision"

    ```python
    import supervision as sv
    from inference import get_model

    model = get_model(model_id="rfdetr-small")
    frames_generator = sv.get_video_frames_generator("<SOURCE_VIDEO_PATH>")

    for frame in frames_generator:
        results = model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)
    ```

## Save Detections as CSV

[`sv.CSVSink`](https://supervision.roboflow.com/0.30.9/detection/tools/save_detections/#supervision.detection.tools.csv_sink.CSVSink)
exports detection fields as CSV rows. Keep the sink open for the video and
append the detections from each frame. To export only selected classes or
confidence levels, [filter detections](filter_detections.md) before saving.

=== "ml-pipes"

    ```{ .py hl_lines="18 21" }
    import supervision as sv

    from ml_pipes.core import Pipeline
    from ml_pipes.standard import Select
    from ml_pipes.supervision import Detections
    from ml_pipes.supervision.inference import RoboflowInference

    pipeline = Pipeline(
        [
            RoboflowInference(model_id="rfdetr-small"),
            Select(0),
            Detections.FromInference(),
        ]
    )

    frames_generator = sv.get_video_frames_generator("<SOURCE_VIDEO_PATH>")

    with sv.CSVSink("<TARGET_CSV_PATH>") as sink:
        for frame in frames_generator:
            detections = pipeline(frame)
            sink.append(detections, {})
    ```

=== "Supervision"

    ```{ .py hl_lines="7 12" }
    import supervision as sv
    from inference import get_model

    model = get_model(model_id="rfdetr-small")
    frames_generator = sv.get_video_frames_generator("<SOURCE_VIDEO_PATH>")

    with sv.CSVSink("<TARGET_CSV_PATH>") as sink:
        for frame in frames_generator:

            results = model.infer(frame)[0]
            detections = sv.Detections.from_inference(results)
            sink.append(detections, {})
    ```

| x_min   | y_min   | x_max   | y_max   | class_id | confidence | tracker_id | class_name |
| ------- | ------- | ------- | ------- | -------- | ---------- | ---------- | ---------- |
| 2941.14 | 1269.31 | 3220.77 | 1500.67 | 2        | 0.8517     |            | car        |
| 944.889 | 899.641 | 1235.42 | 1308.80 | 7        | 0.6752     |            | truck      |
| 1439.78 | 1077.79 | 1621.27 | 1231.40 | 2        | 0.6450     |            | car        |

## Custom Fields

Besides detection fields, the sinks can export custom information such as
the source frame index. The example adds `frame_index` to each detection row;
fields in `detections.data`, such as `class_name`, are also exported.

=== "ml-pipes"

    ```{ .py hl_lines="19 21" }
    import supervision as sv

    from ml_pipes.core import Pipeline
    from ml_pipes.standard import Select
    from ml_pipes.supervision import Detections
    from ml_pipes.supervision.inference import RoboflowInference

    pipeline = Pipeline(
        [
            RoboflowInference(model_id="rfdetr-small"),
            Select(0),
            Detections.FromInference(),
        ]
    )

    frames_generator = sv.get_video_frames_generator("<SOURCE_VIDEO_PATH>")

    with sv.CSVSink("<TARGET_CSV_PATH>") as sink:
        for frame_index, frame in enumerate(frames_generator):
            detections = pipeline(frame)
            sink.append(detections, {"frame_index": frame_index})
    ```

=== "Supervision"

    ```{ .py hl_lines="8 12" }
    import supervision as sv
    from inference import get_model

    model = get_model(model_id="rfdetr-small")
    frames_generator = sv.get_video_frames_generator("<SOURCE_VIDEO_PATH>")

    with sv.CSVSink("<TARGET_CSV_PATH>") as sink:
        for frame_index, frame in enumerate(frames_generator):

            results = model.infer(frame)[0]
            detections = sv.Detections.from_inference(results)
            sink.append(detections, {"frame_index": frame_index})
    ```

## Save Detections as JSON

Replace the CSV sink with [`sv.JSONSink`](https://supervision.roboflow.com/0.30.9/detection/tools/save_detections/#supervision.detection.tools.json_sink.JSONSink);
the detection and custom metadata fields stay unchanged. The JSON array is written
when the sink context exits.

=== "ml-pipes"

    ```{ .py hl_lines="18" }
    import supervision as sv

    from ml_pipes.core import Pipeline
    from ml_pipes.standard import Select
    from ml_pipes.supervision import Detections
    from ml_pipes.supervision.inference import RoboflowInference

    pipeline = Pipeline(
        [
            RoboflowInference(model_id="rfdetr-small"),
            Select(0),
            Detections.FromInference(),
        ]
    )

    frames_generator = sv.get_video_frames_generator("<SOURCE_VIDEO_PATH>")

    with sv.JSONSink("<TARGET_JSON_PATH>") as sink:
        for frame_index, frame in enumerate(frames_generator):
            detections = pipeline(frame)
            sink.append(detections, {"frame_index": frame_index})
    ```

=== "Supervision"

    ```{ .py hl_lines="7" }
    import supervision as sv
    from inference import get_model

    model = get_model(model_id="rfdetr-small")
    frames_generator = sv.get_video_frames_generator("<SOURCE_VIDEO_PATH>")

    with sv.JSONSink("<TARGET_JSON_PATH>") as sink:
        for frame_index, frame in enumerate(frames_generator):

            results = model.infer(frame)[0]
            detections = sv.Detections.from_inference(results)
            sink.append(detections, {"frame_index": frame_index})
    ```

## Author

- [Piotr Skalski](https://github.com/SkalskiP) — Computer Vision Engineer, Roboflow
