"""
RNN 基线：离群点（δ 误差超阈）在哪些工况上集中 —— 按原始温度、|dT/dt|、时间分段统计。

与 eval_rnn_baseline 相同数据与预测流程；不写图，输出 JSON 便于审查。

用法:
  python tools/analyze_rnn_outlier_distribution.py \\
      --model_path outputs_pinn/rnn_baseline_best_val.pt \\
      --split_meta outputs_pinn/train_test_split.json --device cuda:0

  # 子集快速试跑:
  python tools/analyze_rnn_outlier_distribution.py ... --N_used 200000
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from core.data_pipeline import compute_dTdt, parse_data_file
from core.imu_data_io import test_path_from_split_meta
from core.rnn_imu_baseline import load_checkpoint, predict_delta6_sequence
from core.temporal_postprocess import ema_causal_along_time

AXIS = ("ax", "ay", "az", "gx", "gy", "gz")


def _equal_freq_bins(
    x: np.ndarray,
    bad_any: np.ndarray,
    n_bins: int,
    label: str,
) -> list[dict]:
    """按 x 排序后切成等频 n_bins 段，每段样本数近似 N/n_bins，报告每段离群率与 x 范围。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.shape[0]
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    bs = bad_any.astype(np.float64)[order]
    out: list[dict] = []
    base = n // n_bins
    rem = n % n_bins
    start = 0
    for k in range(n_bins):
        sz = base + (1 if k < rem else 0)
        if sz <= 0:
            continue
        end = start + sz
        seg = bs[start:end]
        out.append(
            {
                "bin": k,
                "label": label,
                "x_min": float(xs[start]),
                "x_max": float(xs[end - 1]),
                "n": int(end - start),
                "outlier_rate_any_dim": float(np.mean(seg)),
            }
        )
        start = end
    return out


def _time_deciles(bad_any: np.ndarray) -> list[dict]:
    n = len(bad_any)
    out = []
    for d in range(10):
        lo = (n * d) // 10
        hi = (n * (d + 1)) // 10 if d < 9 else n
        seg = bad_any[lo:hi]
        out.append(
            {
                "decile": d,
                "index_lo": int(lo),
                "index_hi": int(hi),
                "n": int(hi - lo),
                "outlier_rate_any_dim": float(np.mean(seg)),
            }
        )
    return out


def _worst_bins(rows: list[dict], min_n: int, top: int) -> list[dict]:
    elig = [r for r in rows if r["n"] >= min_n]
    elig.sort(key=lambda r: -r["outlier_rate_any_dim"])
    return elig[:top]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--split_meta", default=os.path.join("outputs_pinn", "train_test_split.json"))
    ap.add_argument("--test_data_path", default=None)
    ap.add_argument("--N_used", type=int, default=-1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--bad_threshold", type=float, default=5.0, help="δ 空间 |e|>thr 视为该维离群")
    ap.add_argument("--ema_alpha", type=float, default=None)
    ap.add_argument("--n_bins", type=int, default=32, help="温度 / |dT/dt| 等频分箱数")
    ap.add_argument("--out_json", default=None, help="默认 <model_dir>/rnn_outlier_analysis_<stem>.json")
    ap.add_argument(
        "--chunk",
        type=int,
        default=512,
        help="推理时同时前向的样本数（BiLSTM 显存大时减小，如 256）",
    )
    args = ap.parse_args()

    import torch

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(dev, str):
        dev = dev.replace("：", ":").strip()

    model, meta = load_checkpoint(args.model_path, map_location=dev)
    model.to(dev)
    use_dTdt = bool(meta.get("use_dTdt", meta.get("input_dim", 7) >= 8))

    if args.test_data_path:
        test_path = args.test_data_path
    else:
        with open(args.split_meta, "r", encoding="utf-8") as f:
            smeta = json.load(f)
        test_path, _ = test_path_from_split_meta(smeta)

    n_lines = None if args.N_used < 0 else args.N_used
    X_test, y_true = parse_data_file(test_path, n_lines=n_lines)

    if use_dTdt:
        Tdot = compute_dTdt(X_test[:, 6])
        X_in = np.hstack([X_test.astype(np.float64), Tdot.astype(np.float64)])
    else:
        X_in = X_test.astype(np.float64)

    y_pred = predict_delta6_sequence(model, X_in, meta, device=dev, chunk=max(1, int(args.chunk)))
    if args.ema_alpha is not None:
        a = float(args.ema_alpha)
        if not (0 < a <= 1):
            print("[ERROR] ema_alpha 须在 (0,1]", file=sys.stderr)
            sys.exit(1)
        y_pred = ema_causal_along_time(y_pred, a)

    thr = float(args.bad_threshold)
    abs_e = np.abs(y_pred - y_true)
    bad_any = np.any(abs_e > thr, axis=1)
    bad_per_dim = np.mean(abs_e > thr, axis=0).tolist()

    # 在「任一分量超阈」的子集上，误差最大的一维（主导轴）
    worst_axis = np.argmax(abs_e, axis=1)
    mask_bad = bad_any
    if np.any(mask_bad):
        wa = worst_axis[mask_bad]
        dom = {AXIS[i]: float(np.mean(wa == i)) for i in range(6)}
    else:
        dom = {a: 0.0 for a in AXIS}

    T_raw = X_test[:, 6].astype(np.float64)
    dT = compute_dTdt(T_raw).ravel()
    abs_dtdt = np.abs(dT)

    by_T = _equal_freq_bins(T_raw, bad_any, args.n_bins, "T_raw_col7")
    by_dtdt = _equal_freq_bins(abs_dtdt, bad_any, args.n_bins, "abs_dTdt")
    by_time = _time_deciles(bad_any)

    min_n = max(500, len(bad_any) // (args.n_bins * 20))

    report = {
        "test_data_path": os.path.abspath(test_path),
        "model_path": os.path.abspath(args.model_path),
        "N": int(len(X_test)),
        "bad_threshold_delta_lsb": thr,
        "postprocess": {"ema_alpha": args.ema_alpha},
        "global": {
            "outlier_rate_any_dim": float(np.mean(bad_any)),
            "outlier_rate_per_dim": {AXIS[i]: bad_per_dim[i] for i in range(6)},
            "among_outlier_samples_argmax_axis_fraction": dom,
        },
        "by_T_raw_equal_freq_quantiles": by_T,
        "by_abs_dTdt_equal_freq_quantiles": by_dtdt,
        "by_time_index_decile": by_time,
        "worst_T_bins_by_outlier_rate": _worst_bins(by_T, min_n=min_n, top=10),
        "worst_abs_dTdt_bins_by_outlier_rate": _worst_bins(by_dtdt, min_n=min_n, top=10),
        "worst_time_deciles_by_outlier_rate": sorted(by_time, key=lambda r: -r["outlier_rate_any_dim"])[:5],
    }

    txt = json.dumps(report, ensure_ascii=False, indent=2)
    print(txt)

    out_dir = os.path.dirname(os.path.abspath(args.model_path)) or "outputs_pinn"
    stem = os.path.splitext(os.path.basename(args.model_path))[0]
    out_json = args.out_json or os.path.join(out_dir, f"rnn_outlier_analysis_{stem}.json")
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"\n[INFO] saved: {out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
