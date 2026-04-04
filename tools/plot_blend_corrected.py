import argparse
import json
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import matplotlib.pyplot as plt
import numpy as np

from core.blend_model import load_blend_model_json


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
    return np.stack(xs, axis=0).astype(np.float64)


def bin_mean(x: np.ndarray, y: np.ndarray):
    bins = np.unique(x)
    y_mean = np.zeros_like(bins, dtype=np.float64)
    for i, b in enumerate(bins):
        idx = np.where(x == b)[0]
        y_mean[i] = float(np.mean(y[idx])) if idx.size else np.nan
    return bins, y_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split_meta",
        default=os.path.join("outputs_stream_full", "train_test_split.json"),
        help="多文件训练时写入的划分信息；用于默认读取测试集 txt 路径",
    )
    ap.add_argument("--test_data_path", default=None, help="指定绘图用数据 txt（测试集）；优先于 split_meta")
    ap.add_argument("--data_path", default=None, help="兼容旧版单文件路径")
    ap.add_argument("--N_used", type=int, default=300000, help="-1 表示读满文件")
    ap.add_argument("--blend_model_path", default=os.path.join("outputs_stream_full", "scheme_c_blend_model_stream.json"))
    ap.add_argument("--out_png", default=os.path.join("outputs_stream_full", "scheme_c_corrected_6d_vs_temp_blend_stream_raw.png"))
    ap.add_argument("--bin_round_int", type=int, default=1, help="round mapped Celsius to integer bins")
    ap.add_argument("--use_raw_temp_axis", type=int, default=1, help="1: use 7th column raw temperature as x-axis (no mapping); 0: use mapped Celsius for binning/axis")
    args = ap.parse_args()

    if args.test_data_path:
        plot_path = args.test_data_path
    elif args.data_path:
        plot_path = args.data_path
    else:
        with open(args.split_meta, "r", encoding="utf-8") as f:
            plot_path = json.load(f)["test_file"]

    n_lines = None if args.N_used < 0 else args.N_used
    X = load_data_7col(plot_path, n_lines=n_lines)
    raw6 = X[:, :6]
    T_raw = X[:, 6]

    blend_model = load_blend_model_json(args.blend_model_path)
    delta6_pred = blend_model.predict_delta6(X)
    corrected6 = raw6 + delta6_pred

    if args.use_raw_temp_axis == 1:
        T_plot = T_raw
        T_bin = np.rint(T_plot).astype(np.int64) if args.bin_round_int == 1 else T_plot
    else:
        T_c = blend_model.map_raw_to_c(T_raw)
        T_plot = T_c
        T_bin = np.rint(T_plot).astype(np.int64) if args.bin_round_int == 1 else T_plot

    ideal6 = np.array([0.0, 0.0, 2048.0, 0.0, 0.0, 0.0], dtype=np.float64)
    names = ["ax", "ay", "az", "gx", "gy", "gz"]

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()

    for dim in range(6):
        ax = axes[dim]
        bins, y_mean = bin_mean(T_bin, corrected6[:, dim])
        ax.plot(bins, y_mean, marker="o", linestyle="-", linewidth=1.2, markersize=3)
        ax.axhline(ideal6[dim], linewidth=1.0, linestyle="--")
        ax.set_title(f"{names[dim]} corrected (blend)")
        ax.set_ylabel("LSB")
        ax.grid(True, alpha=0.3)

    axes[-2].set_xlabel("Temperature (°C)")
    axes[-1].set_xlabel("Temperature (°C)")
    fig.suptitle("Scheme C Blend corrected 6D vs mapped temperature")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    fig.savefig(args.out_png, dpi=160)
    print(f"[INFO] saved figure: {args.out_png}")


if __name__ == "__main__":
    main()
