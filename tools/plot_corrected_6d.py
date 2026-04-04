import argparse
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import matplotlib.pyplot as plt
import numpy as np

from core.sparse_kernel_model import load_model_json


def load_data_7col(data_path: str, n_lines: int | None = None):
    xs = []
    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if n_lines is not None and n_lines >= 0 and i >= n_lines:
                break
            line = line.strip()
            if not line:
                continue
            arr = np.fromstring(line, sep="\t")
            if arr.size != 7:
                arr = np.fromstring(line, sep=" ")
            if arr.size != 7:
                continue
            xs.append(arr.astype(np.float64))

    X = np.stack(xs, axis=0).astype(np.float64)
    return X


def map_temp_linear(T_raw: np.ndarray, t_min_c: float = -35.0, t_max_c: float = 65.0):
    r_min = float(np.min(T_raw))
    r_max = float(np.max(T_raw))
    if abs(r_max - r_min) < 1e-12:
        return np.full_like(T_raw, (t_min_c + t_max_c) / 2.0), r_min, r_max
    scale = (t_max_c - t_min_c) / (r_max - r_min)
    T_c = t_min_c + (T_raw - r_min) * scale
    return T_c, r_min, r_max


def bin_mean(x: np.ndarray, y: np.ndarray):
    bins = np.unique(x)
    y_mean = np.zeros_like(bins, dtype=np.float64)
    for i, b in enumerate(bins):
        idx = np.where(x == b)[0]
        y_mean[i] = float(np.mean(y[idx])) if idx.size else np.nan
    return bins, y_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.path.join(_REPO, "data", "1.txt"))
    ap.add_argument("--N_used", type=int, default=300000)
    ap.add_argument("--model_path", default=os.path.join(_REPO, "outputs", "scheme_c_sparse_kernel_model_train.json"))
    ap.add_argument("--out_png", default=os.path.join(_REPO, "outputs", "scheme_c_corrected_6d_vs_temp.png"))
    ap.add_argument("--t_min_c", type=float, default=-35.0)
    ap.add_argument("--t_max_c", type=float, default=65.0)
    ap.add_argument("--bin_round", type=int, default=0, help="0: round to integer °C; 1: round to 0.1°C not supported in this script")
    ap.add_argument("--use_raw_temp_axis", type=int, default=1, help="1: use 7th column raw temperature as x-axis (no mapping); 0: use linear mapping")
    args = ap.parse_args()

    X = load_data_7col(args.data_path, n_lines=args.N_used)
    raw6 = X[:, :6]
    T_raw = X[:, 6]

    model = load_model_json(args.model_path)

    delta6_pred = model.predict_delta6(X)
    corrected6 = raw6 + delta6_pred

    if args.use_raw_temp_axis == 1:
        T_plot = T_raw
        r_min, r_max = float(np.min(T_raw)), float(np.max(T_raw))
    else:
        T_plot, r_min, r_max = map_temp_linear(T_raw, t_min_c=args.t_min_c, t_max_c=args.t_max_c)

    if args.bin_round != 0:
        raise ValueError("Currently only integer °C binning is implemented (bin_round=0).")
    T_bin = np.rint(T_plot).astype(np.int64)

    ideal6 = np.array([0.0, 0.0, 2048.0, 0.0, 0.0, 0.0], dtype=np.float64)
    names = ["ax", "ay", "az", "gx", "gy", "gz"]

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()

    for dim in range(6):
        ax = axes[dim]
        bins, y_mean = bin_mean(T_bin, corrected6[:, dim])
        ax.plot(bins, y_mean, marker="o", linestyle="-", linewidth=1.2, markersize=3)
        ax.axhline(ideal6[dim], linewidth=1.0, linestyle="--")
        ax.set_title(f"{names[dim]} corrected")
        ax.set_ylabel("LSB")
        ax.grid(True, alpha=0.3)

    axes[-2].set_xlabel("Temperature (°C)")
    axes[-1].set_xlabel("Temperature (°C)")
    fig.suptitle(
        "Scheme C (sparse RBF) corrected 6D vs mapped temperature\n"
        f"axis: T_raw[{r_min:.3f},{r_max:.3f}] -> {'no mapping' if args.use_raw_temp_axis==1 else f'[{args.t_min_c},{args.t_max_c}]'}; N_used={len(X)}",
        y=0.98,
        fontsize=12,
    )

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out_png, dpi=160)
    print(f"[INFO] saved figure: {args.out_png}")


if __name__ == "__main__":
    main()
