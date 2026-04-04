import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .sparse_kernel_model import SparseKernelRbfModel, load_model_json


@dataclass
class QuadraticPoly6:
    a2: np.ndarray  # (6,)
    a1: np.ndarray  # (6,)
    a0: np.ndarray  # (6,)

    def predict(self, T_c: np.ndarray) -> np.ndarray:
        T2 = T_c * T_c
        return (T2[:, None] * self.a2[None, :] +
                T_c[:, None] * self.a1[None, :] +
                self.a0[None, :])


@dataclass
class BlendModel6:
    kernel_model: SparseKernelRbfModel
    poly6: QuadraticPoly6

    raw_min: float
    raw_max: float
    t_min_c: float
    t_max_c: float

    T_start_c: float
    margin_c: float

    clip_low: np.ndarray  # (6,)
    clip_high: np.ndarray  # (6,)

    def map_raw_to_c(self, T_raw: np.ndarray) -> np.ndarray:
        if abs(self.raw_max - self.raw_min) < 1e-12:
            return np.full_like(T_raw, (self.t_min_c + self.t_max_c) / 2.0, dtype=np.float64)
        scale = (self.t_max_c - self.t_min_c) / (self.raw_max - self.raw_min)
        return self.t_min_c + (T_raw - self.raw_min) * scale

    def blend_weight(self, T_c: np.ndarray) -> np.ndarray:
        lo = self.T_start_c - self.margin_c
        hi = self.T_start_c + self.margin_c
        w = np.zeros_like(T_c, dtype=np.float64)
        mid = (T_c > lo) & (T_c < hi)
        w[T_c >= hi] = 1.0
        w[mid] = (T_c[mid] - lo) / (hi - lo)
        return w

    def predict_delta6(self, X_raw_7col: np.ndarray, chunk: int = 50000) -> np.ndarray:
        T_raw = X_raw_7col[:, 6]
        T_c = self.map_raw_to_c(T_raw)
        w = self.blend_weight(T_c)

        delta_kernel = self.kernel_model.predict_delta6(X_raw_7col, chunk=chunk)
        delta_poly = self.poly6.predict(T_c)

        delta_blend = (1.0 - w)[:, None] * delta_kernel + w[:, None] * delta_poly

        if self.clip_low is not None and self.clip_high is not None:
            delta_blend = np.minimum(np.maximum(delta_blend, self.clip_low[None, :]), self.clip_high[None, :])
        return delta_blend


def load_blend_model_json(path: str) -> BlendModel6:
    with open(path, "r", encoding="utf-8") as f:
        obj: dict[str, Any] = json.load(f)

    kernel_model = load_model_json(obj["kernel_model_path"])

    poly6 = QuadraticPoly6(
        a2=np.array(obj["poly6_a2"], dtype=np.float64),
        a1=np.array(obj["poly6_a1"], dtype=np.float64),
        a0=np.array(obj["poly6_a0"], dtype=np.float64),
    )

    return BlendModel6(
        kernel_model=kernel_model,
        poly6=poly6,
        raw_min=float(obj["raw_min"]),
        raw_max=float(obj["raw_max"]),
        t_min_c=float(obj["t_min_c"]),
        t_max_c=float(obj["t_max_c"]),
        T_start_c=float(obj["T_start_c"]),
        margin_c=float(obj["margin_c"]),
        clip_low=np.array(obj["clip_low"], dtype=np.float64) if obj.get("clip_low") is not None else None,
        clip_high=np.array(obj["clip_high"], dtype=np.float64) if obj.get("clip_high") is not None else None,
    )
