from unittest.mock import Mock

import numpy as np
import pytest
import supervision as sv

from ml_pipes.supervision import ImageWindow
from ml_pipes.vision import ImagePayload


def test_image_window_constructs_upstream_window_without_opening_gui() -> None:
    # Supervision creates the actual GUI lazily on show(), not construction.
    operator = ImageWindow(title="preview", keep_aspect_ratio=False, at=0)

    assert isinstance(operator.window, sv.ImageWindow)
    assert operator.window.title == "preview"
    assert operator.window.keep_aspect_ratio is False
    assert operator.window.is_open is False


def test_image_window_displays_tuple_frame_and_preserves_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    show = Mock()
    # Patch only the GUI boundary, retaining the real upstream constructor/API.
    monkeypatch.setattr(sv.ImageWindow, "show", show)
    operator = ImageWindow(at=0)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)[:, ::2]
    payload = (frame, "detections")

    returned = operator(payload)

    assert returned is payload
    show.assert_called_once()
    displayed = show.call_args.args[0]
    np.testing.assert_array_equal(displayed, frame)
    assert displayed.flags.c_contiguous


def test_image_window_converts_rgb_payload_to_bgr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    show = Mock()
    monkeypatch.setattr(sv.ImageWindow, "show", show)
    operator = ImageWindow()
    rgb = np.asarray([[[255, 0, 0]]], dtype=np.uint8)
    payload = ImagePayload(array=rgb, color_space="RGB")

    returned = operator(payload)

    assert returned is payload
    show.assert_called_once()
    displayed = show.call_args.args[0]
    np.testing.assert_array_equal(displayed, [[[0, 0, 255]]])
    np.testing.assert_array_equal(rgb, [[[255, 0, 0]]])
    assert displayed.flags.c_contiguous
