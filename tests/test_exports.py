from ml_pipes import supervision


def test_public_operator_exports() -> None:
    assert supervision.BoxAnnotator
    assert supervision.LabelAnnotator
    assert supervision.Detection
    assert supervision.Detections
    assert supervision.TrackingTimer
    assert supervision.TriggerZone
    assert supervision.DetectionsSmoother
    assert supervision.TriggerLineZone
    assert supervision.MaskAnnotator
    assert supervision.ImageWindow

    for name in supervision.__all__:
        assert getattr(supervision, name) is not None


def test_detection_transformations_are_exposed_through_namespace() -> None:
    for name in ("AttachData", "AttachDataField", "PadBoxes", "ScaleBoxes", "MoveBoxes", "ClipBoxes", "Crop", "Merge"):
        assert getattr(supervision.Detections, name) is not None
        assert name not in supervision.__all__
