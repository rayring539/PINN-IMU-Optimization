"""
RNN 基线评测 —— 与 ``eval_sparse_kernel`` / ``eval_pinn`` 相同测试集与 δ 指标。

用法:
  python eval_rnn_baseline.py --model_path outputs_pinn/rnn_baseline.pt \\
      --split_meta outputs_pinn/train_test_split.json --device cuda
  # 六轴实际 vs 预测图（校正后 LSB）：加 --plot，输出 *_timeseries.png 与 *_scatter.png
  python eval_rnn_baseline.py ... --plot --plot_start 0 --plot_len 15000
  # 对预测 δ 沿时间做因果 EMA（0<α≤1，越小越平滑）：--ema_alpha 0.15
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.dirname(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from core.data_pipeline import compute_dTdt, parse_data_file, stratified_split_by_temp_bin_round_int
from core.imu_data_io import test_path_from_split_meta
from core.pinn_metrics import summarize_delta_metrics
from core.rnn_imu_baseline import load_checkpoint, predict_delta6_sequence
from core.temporal_postprocess import ema_causal_along_time


def mae_rmse(a: np.ndarray, b: np.ndarray):
    err = b - a
    return np.mean(np.abs(err), axis=0).tolist(), np.sqrt(np.mean(err ** 2, axis=0)).tolist()


def coverage_within_threshold(corrected_true: np.ndarray, corrected_pred: np.ndarray, eps: float):
    err = np.abs(corrected_pred - corrected_true)
    return np.mean(err <= eps, axis=0).tolist()


AXIS_LABELS = ("ax", "ay", "az", "gx", "gy", "gz")


def save_pred_vs_true_figures(
    corrected_true: np.ndarray,
    corrected_pred: np.ndarray,
    plot_base: str,
    plot_start: int,
    plot_len: int,
    scatter_n: int,
) -> tuple[str | None, str | None]:
    """保存六轴「校正后 LSB」真值/预测：时间序列图 + 可选散点图（子采样）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = int(len(corrected_true))
    s0 = max(0, min(plot_start, max(0, n - 1)))
    L = max(1, min(plot_len, n - s0))
    sl = slice(s0, s0 + L)
    t = np.arange(s0, s0 + L, dtype=np.int64)
    yt = corrected_true[sl]
    yp = corrected_pred[sl]

    out_ts = f"{plot_base}_timeseries.png"
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for i, ax in enumerate(axes.flat):
        ax.plot(t, yt[:, i], label="实际", color="C0", linewidth=0.8, alpha=0.9)
        ax.plot(t, yp[:, i], label="预测", color="C1", linewidth=0.8, alpha=0.85)
        ax.set_ylabel(f"{AXIS_LABELS[i]} (LSB)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1, 0].set_xlabel("样本索引")
    axes[-1, 1].set_xlabel("样本索引")
    fig.suptitle(f"校正后 LSB：实际 vs 预测（索引 [{s0}, {s0 + L})，共 {L} 点）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_ts) or ".", exist_ok=True)
    fig.savefig(out_ts, dpi=150)
    plt.close(fig)

    out_sc: str | None = None
    if scatter_n > 0:
        m = min(scatter_n, n)
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=m, replace=False)
        at = corrected_true[idx]
        ap = corrected_pred[idx]
        out_sc = f"{plot_base}_scatter.png"
        fig2, axes2 = plt.subplots(3, 2, figsize=(11, 10))
        for i, ax in enumerate(axes2.flat):
            ax.scatter(at[:, i], ap[:, i], s=1, alpha=0.25, c="C0", rasterized=True)
            lo = float(min(at[:, i].min(), ap[:, i].min()))
            hi = float(max(at[:, i].max(), ap[:, i].max()))
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, alpha=0.7)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(f"实际 {AXIS_LABELS[i]}")
            ax.set_ylabel(f"预测 {AXIS_LABELS[i]}")
            ax.grid(True, alpha=0.3)
        fig2.suptitle(f"散点：预测 vs 实际（子采样 m={m}）", fontsize=12)
        fig2.tight_layout(rect=[0, 0, 1, 0.97])
        fig2.savefig(out_sc, dpi=150)
        plt.close(fig2)

    return out_ts, out_sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=os.path.join("outputs_pinn", "rnn_baseline.pt"))
    ap.add_argument("--split_meta", default=os.path.join("outputs_pinn", "train_test_split.json"))
    ap.add_argument("--test_data_path", default=None)
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--N_used", type=int, default=-1)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=5.0)
    ap.add_argument("--bad_x", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--plot",
        action="store_true",
        help="保存六轴「校正后 LSB」实际 vs 预测图（时间序列 + 散点）",
    )
    ap.add_argument(
        "--plot_out",
        default=None,
        help="图文件路径前缀（无扩展名）；默认与模型同目录 rnn_pred_vs_true_<模型名>_test_<数据名>",
    )
    ap.add_argument("--plot_start", type=int, default=0, help="时间序列起始样本索引")
    ap.add_argument("--plot_len", type=int, default=12000, help="时间序列连续样本数")
    ap.add_argument(
        "--plot_scatter_n",
        type=int,
        default=50000,
        help="散点图随机子采样点数；0 表示不生成散点图",
    )
    ap.add_argument(
        "--ema_alpha",
        type=float,
        default=None,
        help="对预测 δ 做因果 EMA：s_t=α·δ_t+(1-α)·s_{t-1}；不设则不做后处理（建议 0.05–0.3 试）",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.model_path):
        print(
            f"[ERROR] 找不到模型: {args.model_path}\n"
            "        若尚未训练出 *_best_val.pt，请改用 rnn_baseline.pt 或先完成带 save_best_val 的训练。",
            file=sys.stderr,
        )
        sys.exit(1)

    _torch = __import__("torch")
    dev = args.device or ("cuda" if _torch.cuda.is_available() else "cpu")
    if isinstance(dev, str):
        dev = dev.replace("：", ":").strip()
    model, meta = load_checkpoint(args.model_path, map_location=dev)
    model.to(dev)

    use_dTdt = bool(meta.get("use_dTdt", meta.get("input_dim", 7) >= 8))

    test_path: str | None
    if args.test_data_path:
        test_path = args.test_data_path
        split_note = "explicit_test_data_path"
    elif args.data_path:
        X_raw, y_delta = parse_data_file(args.data_path, n_lines=None if args.N_used < 0 else args.N_used)
        _, _, X_test, y_delta_test = stratified_split_by_temp_bin_round_int(
            X_raw, y_delta, test_ratio=args.test_ratio, seed=args.seed
        )
        split_note = "legacy_stratified"
        test_path = None
    else:
        with open(args.split_meta, "r", encoding="utf-8") as f:
            smeta = json.load(f)
        test_path, note = test_path_from_split_meta(smeta)
        split_note = f"split_meta test={note}"

    if test_path is not None:
        n_lines = None if args.N_used < 0 else args.N_used
        X_test, y_delta_test = parse_data_file(test_path, n_lines=n_lines)

    if use_dTdt:
        Tdot = compute_dTdt(X_test[:, 6])
        X_in = np.hstack([X_test.astype(np.float64), Tdot.astype(np.float64)])
    else:
        X_in = X_test.astype(np.float64)

    # meta 中 x_mean 等可能为 list（来自 json）；predict 内会 np.asarray
    y_delta_pred = predict_delta6_sequence(model, X_in, meta, device=dev)

    ema_alpha = args.ema_alpha
    if ema_alpha is not None:
        if not (0.0 < float(ema_alpha) <= 1.0):
            print(f"[ERROR] --ema_alpha 须在 (0, 1] 内，得到 {ema_alpha}", file=sys.stderr)
            sys.exit(1)
        y_delta_pred = ema_causal_along_time(y_delta_pred, float(ema_alpha))

    raw6 = X_test[:, :6]
    corrected_true = raw6 + y_delta_test
    corrected_pred = raw6 + y_delta_pred

    mae, rmse = mae_rmse(corrected_true, corrected_pred)
    cov_per_dim = coverage_within_threshold(corrected_true, corrected_pred, eps=float(args.eps))
    bad_x = float(args.bad_x) if args.bad_x is not None else float(args.eps)
    delta_lsb = summarize_delta_metrics(y_delta_test, y_delta_pred, bad_x)

    result = {
        "model_path": args.model_path,
        "split_type": split_note,
        "test_data_path": test_path if test_path is not None else args.data_path,
        "rnn_meta": {
            k: meta[k]
            for k in ("seq_len", "rnn_type", "bidirectional", "hidden_dim", "input_dim", "use_dTdt")
            if k in meta
        },
        "N_used": int(args.N_used),
        "eps": float(args.eps),
        "bad_x": bad_x,
        "metrics_corrected_lsb": {"mae_per_dim": mae, "rmse_per_dim": rmse},
        "coverage_per_dim": cov_per_dim,
        "delta_lsb": delta_lsb,
        "N_test": int(len(X_test)),
        "postprocess": {
            "ema_alpha": None if ema_alpha is None else float(ema_alpha),
        },
    }

    print("[INFO] eval result:\n" + json.dumps(result, ensure_ascii=False, indent=2, default=str))

    safe = os.path.splitext(os.path.basename(test_path if test_path else args.data_path or "test"))[0]
    model_stem = os.path.splitext(os.path.basename(args.model_path))[0]
    out_dir = os.path.dirname(args.model_path) or "outputs_pinn"
    ema_slug = "" if ema_alpha is None else f"_ema{float(ema_alpha):g}"
    out_path = os.path.join(out_dir, f"rnn_eval_{model_stem}_bad{bad_x}_eps{args.eps}_test_{safe}{ema_slug}.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"[INFO] saved: {out_path}")

    if args.plot:
        plot_base = args.plot_out
        if not plot_base:
            plot_base = os.path.join(out_dir, f"rnn_pred_vs_true_{model_stem}_test_{safe}{ema_slug}")
        ts_png, sc_png = save_pred_vs_true_figures(
            corrected_true,
            corrected_pred,
            plot_base,
            plot_start=int(args.plot_start),
            plot_len=int(args.plot_len),
            scatter_n=int(args.plot_scatter_n),
        )
        print(f"[INFO] plot (timeseries): {ts_png}")
        if sc_png:
            print(f"[INFO] plot (scatter): {sc_png}")


if __name__ == "__main__":
    main()
