"""
RNN（LSTM/GRU）基线：用过去 seq_len 步的 IMU 特征预测当前步 δ（6 维 LSB）。

与 PINN/核回归对比时：本模型显式利用 **时序上下文**；seq_len=1 时退化为逐步 MLP（需另写）。
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class IMURNNBaseline(nn.Module):
    """(B, T, D) → (B, 6)，取 RNN 最后一步隐状态后接 Linear。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        rnn_type: str = "LSTM",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        rnn_type = str(rnn_type).upper()
        self.rnn_type = "GRU" if rnn_type == "GRU" else "LSTM"
        klass = nn.GRU if self.rnn_type == "GRU" else nn.LSTM
        self.rnn = klass(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last)


def _pad_window(X: np.ndarray, i: int, seq_len: int) -> np.ndarray:
    """取以 i 结尾、长度为 seq_len 的窗口（不足则在左侧重复首行填充）。"""
    start = i - seq_len + 1
    if start >= 0:
        return X[start : i + 1].copy()
    chunk = X[: i + 1]
    pad = np.tile(chunk[0:1], (seq_len - len(chunk), X.shape[1]))
    return np.vstack([pad, chunk])


@torch.no_grad()
def predict_delta6_sequence(
    model: IMURNNBaseline,
    X_raw: np.ndarray,
    meta: dict[str, Any],
    device: str = "cpu",
    chunk: int = 4096,
) -> np.ndarray:
    """
    X_raw: (N, D) 与训练时一致（7 或 8 维）；按 ``seq_len`` 滑窗预测每步 δ。
    """
    model.eval()
    seq_len = int(meta["seq_len"])
    x_mean = np.asarray(meta["x_mean"], dtype=np.float64)
    x_std = np.asarray(meta["x_std"], dtype=np.float64)
    y_mean = np.asarray(meta["y_mean"], dtype=np.float64)
    y_std = np.asarray(meta["y_std"], dtype=np.float64)

    N = X_raw.shape[0]
    out = np.zeros((N, 6), dtype=np.float64)
    dev = torch.device(device)

    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        batch = []
        for i in range(s, e):
            w = _pad_window(X_raw, i, seq_len)
            w = (w - x_mean) / x_std
            batch.append(w.astype(np.float32))
        xt = torch.from_numpy(np.stack(batch, axis=0)).to(dev)
        yp = model(xt).cpu().numpy()
        yp = yp * y_std[None, :] + y_mean[None, :]
        out[s:e] = yp.astype(np.float64)

    return out


def save_checkpoint(
    path: str,
    model: IMURNNBaseline,
    meta: dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> tuple[IMURNNBaseline, dict[str, Any]]:
    blob = torch.load(path, map_location=map_location, weights_only=False)
    meta = blob["meta"]
    m = IMURNNBaseline(
        input_dim=int(meta["input_dim"]),
        hidden_dim=int(meta.get("hidden_dim", 128)),
        num_layers=int(meta.get("num_layers", 2)),
        rnn_type=str(meta.get("rnn_type", "LSTM")),
        dropout=float(meta.get("dropout", 0.0)),
    )
    m.load_state_dict(blob["state_dict"])
    return m, meta


def export_meta_json(path: str, meta: dict[str, Any]) -> None:
    """便于人读；与 .pt 中 meta 一致。"""
    def _ser(o: Any):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return float(o) if isinstance(o, np.floating) else int(o)
        return o

    out = {k: _ser(v) if not isinstance(v, dict) else v for k, v in meta.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
