---
title: Save Detections to CSV or JSON with Supervision
description: Save object detection results to CSV or JSON with supervision's CSVSink and JSONSink — export predictions for analysis and downstream pipelines.
---

# Save Detections

Supervision enables an easy way to save detections in .CSV and .JSON files for offline processing. This guide demonstrates video inference with [Inference](https://github.com/roboflow/inference) and export with [`sv.CSVSink`](https://supervision.roboflow.com/latest/detection/tools/save_detections/#supervision.detection.tools.csv_sink.CSVSink) and [`sv.JSONSink`](https://supervision.roboflow.com/latest/detection/tools/save_detections/#supervision.detection.tools.json_sink.JSONSink).

## Run Detection

First, you'll need to obtain predictions from your object detection or segmentation model. You can learn more on this topic in our [How to Detect and Annotate](https://supervision.roboflow.com/latest/how_to/detect_and_annotate/) guide.

To generate predictions for saving, initialize your model and iterate over video frames using `sv.get_video_frames_generator`. Each frame is passed to the model, and the raw output is converted into a `sv.Detections` object. This detection loop forms the foundation for both CSV and JSON export workflows shown below.

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

To save detections to a `.CSV` file, open our [`sv.CSVSink`](https://supervision.roboflow.com/latest/detection/tools/save_detections/#supervision.detection.tools.csv_sink.CSVSink) and then pass the [`sv.Detections`](https://supervision.roboflow.com/latest/detection/core/#supervision.detection.core.Detections) object resulting from the inference to it. Its fields are parsed and saved on disk.

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

Besides regular fields in [`sv.Detections`](https://supervision.roboflow.com/latest/detection/core/#supervision.detection.core.Detections), [`sv.CSVSink`](https://supervision.roboflow.com/latest/detection/tools/save_detections/#supervision.detection.tools.csv_sink.CSVSink) also allows you to add custom information to each row, which can be passed via the `custom_data` dictionary. Let's utilize this feature to save information about the frame index from which the detections originate.

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

| x_min   | y_min   | x_max   | y_max   | class_id | confidence | tracker_id | class_name | frame_index |
| ------- | ------- | ------- | ------- | -------- | ---------- | ---------- | ---------- | ----------- |
| 2941.14 | 1269.31 | 3220.77 | 1500.67 | 2        | 0.8517     |            | car        | 0           |
| 944.889 | 899.641 | 1235.42 | 1308.80 | 7        | 0.6752     |            | truck      | 0           |
| 1439.78 | 1077.79 | 1621.27 | 1231.40 | 2        | 0.6450     |            | car        | 0           |

## Save Detections as JSON

If you prefer to save the result in a `.JSON` file instead of a `.CSV` file, all you need to do is replace [`sv.CSVSink`](https://supervision.roboflow.com/latest/detection/tools/save_detections/#supervision.detection.tools.csv_sink.CSVSink) with [`sv.JSONSink`](https://supervision.roboflow.com/latest/detection/tools/save_detections/#supervision.detection.tools.json_sink.JSONSink).

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

## Frequently Asked Questions

### How do I save detections to CSV with supervision?

Open `sv.CSVSink("output.csv")` as a context manager and call `sink.append(detections)` for each frame. The CSV includes box coordinates, confidence, class ID, tracker ID, and any fields stored in `detections.data`.

### Can I save detections to JSON instead?

Yes. Open `sv.JSONSink("output.json")` as a context manager and call `sink.append(detections)` for each frame. The file is written as a JSON array when the context exits.

### Can I add custom fields to the saved output?

Yes. Pass a dict as the second argument: `sink.append(detections, {"frame_index": 5})` — the keys become extra columns in the CSV or extra fields in the JSON.

### Can I save only specific classes or confidence levels?

Filter the `Detections` object before saving: `sink.append(detections[detections.confidence > 0.7])`.

## Author

- [Piotr Skalski](https://github.com/SkalskiP) — Computer Vision Engineer, Roboflow
