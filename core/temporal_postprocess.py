"""
沿时间轴的因果后处理（用于 δ 或任意 (N, D) 序列）。

典型用法：对网络输出的逐样本 δ 做 EMA，抑制高频抖动；α 越小曲线越平滑、瞬态滞后越大。
"""
from __future__ import annotations

import numpy as np


def ema_causal_along_time(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    因果指数滑动平均（各维独立、共享同一时间索引）::

        s[0] = x[0]
        s[t] = α * x[t] + (1 - α) * s[t-1],  t >= 1

    Parameters
    ----------
    x : (N, D) array
    alpha : (0, 1]
        α=1 时恒等（无平滑）；α 越小越平滑。

    Returns
    -------
    (N, D), float64
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"ema_causal_along_time 需要二维数组 (N, D)，得到 shape={x.shape}")
    a = float(alpha)
    if not (0.0 < a <= 1.0):
        raise ValueError(f"alpha 须在 (0, 1] 内，得到 {alpha}")
    n = x.shape[0]
    if n == 0:
        return x.copy()
    if a >= 1.0 - 1e-15:
        return x.copy()
    out = np.empty_like(x)
    out[0] = x[0]
    om = 1.0 - a
    for t in range(1, n):
        out[t] = a * x[t] + om * out[t - 1]
    return out
