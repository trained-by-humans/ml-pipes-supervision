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
