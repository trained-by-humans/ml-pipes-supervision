"""ml-pipes operators for the external :mod:`inference` package."""
from __future__ import annotations

from typing import Any, cast, overload

import numpy as np
import numpy.typing as npt

try:
    from inference import get_model
    from inference.core.models.base import Model
except ModuleNotFoundError as error:
    if error.name == "inference":
        raise ModuleNotFoundError(
            "RoboflowInference requires the optional Inference dependency. "
            "Install it with 'python -m pip install \"ml-pipes-supervision[inference]\"'."
        ) from error
    raise

from ml_pipes.operator import Operator
from ml_pipes.vision import ImagePayload

__all__ = ["RoboflowInference"]


@Operator
class RoboflowInference:
    @overload
    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        **infer_kwargs: Any,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        model: Model,
        /,
        **infer_kwargs: Any,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        *,
        model: Model,
        **infer_kwargs: Any,
    ) -> None:
        ...

    def __init__(
        self,
        model_id: str | Model | None = None,
        api_key: str | None = None,
        *,
        model: Model | None = None,
        **infer_kwargs: Any,
    ) -> None:
        if model is not None:
            if model_id is not None:
                raise ValueError("RoboflowInference accepts either model_id or model, not both.")
            model_id = model

        if isinstance(model_id, str):
            self.model_id: str | None = model_id
            self._model: Model = self._build_model(model_id, api_key)
        else:
            if model_id is None:
                raise ValueError("RoboflowInference requires model_id or model.")
            if api_key is not None:
                raise ValueError(
                    "RoboflowInference accepts api_key only when initialized with a model id."
                )
            self.model_id = None
            self._model = model_id
        self.api_key = api_key
        self.infer_kwargs = infer_kwargs

    def __call__(self, image: Any) -> Any:
        return self._model.infer(self._normalize_image(image), **self.infer_kwargs)

    @staticmethod
    def _build_model(model_id: str, api_key: str | None) -> Model:
        kwargs = {"model_id": model_id}
        if api_key:
            kwargs["api_key"] = api_key
        return get_model(**kwargs)

    @staticmethod
    def _payload_to_bgr_hwc(image: ImagePayload) -> npt.NDArray[np.uint8]:
        if image.layout != "HWC":
            raise ValueError(f"RoboflowInference expects HWC images, got {image.layout!r}")

        converted = np.ascontiguousarray(image.array)
        if image.color_space == "RGB":
            return cast(npt.NDArray[np.uint8], converted[..., ::-1].copy())
        return cast(npt.NDArray[np.uint8], converted)

    @classmethod
    def _normalize_image(cls, image: Any) -> Any:
        if not isinstance(image, ImagePayload):
            return image
        return cls._payload_to_bgr_hwc(image)
