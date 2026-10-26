---
title: Supervision Pipeline Recipe Reference
description: >-
  Find runnable ml-pipes source examples for face blurring, line-crossing counts, zero-shot detection, oriented bounding boxes, and vision-language descriptions and attribute labels.
---

# Pipeline Recipe Reference

This reference maps each complete, runnable pipeline to its source example and
upstream Supervision notebook, where applicable. For the visual recipe overview, see
[Pipeline Recipes](gallery.md).

| Recipe | Runnable Pipeline | Upstream Source |
|---|---|---|
| [Blurring Faces](blurring_faces.md) | `examples/run_blur_faces.py` | [`blurring_faces.ipynb`](https://github.com/roboflow/supervision/blob/develop/docs/notebooks/blurring_faces.ipynb) |
| [Count Objects Crossing a Line](count_objects_crossing_line.md) | `examples/run_count_objects_crossing_line.py` | [`count-objects-crossing-the-line.ipynb`](https://github.com/roboflow/supervision/blob/develop/docs/notebooks/count-objects-crossing-the-line.ipynb) |
| [Zero-Shot Object Detection with YOLO-World](zero_shot_object_detection_with_yolo_world.md) | `examples/run_zero_shot_object_detection.py` | [`zero-shot-object-detection-with-yolo-world.ipynb`](https://github.com/roboflow/supervision/blob/develop/docs/notebooks/zero-shot-object-detection-with-yolo-world.ipynb) |
| [Oriented Bounding Boxes](oriented_bounding_boxes.md) | `examples/run_oriented_bounding_boxes.py` | [Oriented Bounding Boxes](https://supervision.roboflow.com/0.30.0/notebooks/oriented-bounding-boxes/) |
| [Detect and Describe](detect_and_describe.md) | `examples/run_detect_and_describe.py` | Original ml-pipes recipe |
| [Detect and Attribute](detect_and_attribute.md) | `examples/run_detect_and_attribute.py` | Original ml-pipes recipe |
