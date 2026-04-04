"""
验证集 / 评测指标：在 δ（6 轴校正量）LSB 空间上定义。

- 各轴 **RMSE**: sqrt(mean(e_i^2))，沿样本维。
- 各轴 **bad**: 该轴 |e_i| > threshold 的样本占比。
- **总 RMSE**: sqrt(mean(e^2))，对所有样本 × 6 轴。
- **总 bad**: 任一分量超阈的样本占比（与逐轴 bad 不同；亦可要元素级 bad 见 ``bad_element_rate``）。
"""
from __future__ import annotations

import numpy as np

AXIS_NAMES = ("ax", "ay", "az", "gx", "gy", "gz")


def delta_rmse_lsb(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """总 RMSE（LSB）：对所有 N×6 元素取均方再开方。"""
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def delta_rmse_per_dim(y_true: np.ndarray, y_pred: np.ndarray) -> list[float]:
    """各轴 RMSE（LSB），长度为 6。"""
    err = y_pred - y_true
    return np.sqrt(np.mean(err ** 2, axis=0)).tolist()


def delta_bad_per_dim(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> list[float]:
    """各轴：|e_i| > threshold 的样本占比 ∈ [0,1]。"""
    err = np.abs(y_pred - y_true)
    thr = float(threshold)
    return np.mean(err > thr, axis=0).tolist()


def delta_bad_any_rate(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    """总 bad（样本级）：任一分量超阈的样本占比。"""
    err = np.abs(y_pred - y_true)
    return float(np.mean(np.any(err > float(threshold), axis=1)))


def delta_bad_element_rate(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    """元素级：所有 N×6 个误差中 |e|>threshold 的比例。"""
    err = np.abs(y_pred - y_true)
    return float(np.mean(err > float(threshold)))


def summarize_delta_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float
) -> dict:
    """评测用：一次返回各轴与总体的 RMSE / bad。"""
    th = float(threshold)
    return {
        "axis_labels": list(AXIS_NAMES),
        "rmse_per_dim": delta_rmse_per_dim(y_true, y_pred),
        "bad_per_dim": delta_bad_per_dim(y_true, y_pred, th),
        "rmse_total": delta_rmse_lsb(y_true, y_pred),
        "bad_total": delta_bad_any_rate(y_true, y_pred, th),
        "bad_element_rate": delta_bad_element_rate(y_true, y_pred, th),
    }


def make_delta_val_metrics(bad_threshold: float) -> tuple[list, tuple[str, ...]]:
    """供 DeepXDE ``model.compile(..., metrics=...)``：14 个标量 + 名称元组。"""
    th = float(bad_threshold)
    metrics: list = []

    for i in range(6):
        def _rmse_i(yt, yp, axis=i):
            return float(np.sqrt(np.mean((yp[:, axis] - yt[:, axis]) ** 2)))

        metrics.append(_rmse_i)

    for i in range(6):
        def _bad_i(yt, yp, axis=i):
            e = np.abs(yp[:, axis] - yt[:, axis])
            return float(np.mean(e > th))

        metrics.append(_bad_i)

    def _rmse_tot(yt, yp):
        return delta_rmse_lsb(yt, yp)

    def _bad_tot(yt, yp):
        return delta_bad_any_rate(yt, yp, th)

    metrics.extend([_rmse_tot, _bad_tot])

    names = (
        tuple(f"rmse_{AXIS_NAMES[i]}" for i in range(6))
        + tuple(f"bad_{AXIS_NAMES[i]}" for i in range(6))
        + ("rmse_total", "bad_total")
    )
    return metrics, names


# 默认与 make_delta_val_metrics 返回顺序一致
VAL_METRIC_NAMES = (
    tuple(f"rmse_{AXIS_NAMES[i]}" for i in range(6))
    + tuple(f"bad_{AXIS_NAMES[i]}" for i in range(6))
    + ("rmse_total", "bad_total")
)


def correction_error_bad_stats(err: np.ndarray, threshold: float) -> tuple[list[float], float]:
    """err = corr_pred - corr_true（LSB）。保留兼容：各轴 bad 占比 + 任一分量超阈样本占比。"""
    a = np.abs(err)
    thr = float(threshold)
    per_dim = np.mean(a > thr, axis=0).tolist()
    any_bad = float(np.mean(np.any(a > thr, axis=1)))
    return per_dim, any_bad
