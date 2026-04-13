"""
RNN 基线评测 —— 与 ``eval_sparse_kernel`` / ``eval_pinn`` 相同测试集与 δ 指标。

用法:
  python eval_rnn_baseline.py --model_path outputs_pinn/rnn_baseline.pt \\
      --split_meta outputs_pinn/train_test_split.json --device cuda
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


def mae_rmse(a: np.ndarray, b: np.ndarray):
    err = b - a
    return np.mean(np.abs(err), axis=0).tolist(), np.sqrt(np.mean(err ** 2, axis=0)).tolist()


def coverage_within_threshold(corrected_true: np.ndarray, corrected_pred: np.ndarray, eps: float):
    err = np.abs(corrected_pred - corrected_true)
    return np.mean(err <= eps, axis=0).tolist()


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
    args = ap.parse_args()

    if not os.path.isfile(args.model_path):
        print(f"[ERROR] 找不到模型: {args.model_path}", file=sys.stderr)
        sys.exit(1)

    dev = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
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
        "rnn_meta": {k: meta[k] for k in ("seq_len", "rnn_type", "hidden_dim", "input_dim", "use_dTdt") if k in meta},
        "N_used": int(args.N_used),
        "eps": float(args.eps),
        "bad_x": bad_x,
        "metrics_corrected_lsb": {"mae_per_dim": mae, "rmse_per_dim": rmse},
        "coverage_per_dim": cov_per_dim,
        "delta_lsb": delta_lsb,
        "N_test": int(len(X_test)),
    }

    print("[INFO] eval result:\n" + json.dumps(result, ensure_ascii=False, indent=2, default=str))

    safe = os.path.splitext(os.path.basename(test_path if test_path else args.data_path or "test"))[0]
    model_stem = os.path.splitext(os.path.basename(args.model_path))[0]
    out_dir = os.path.dirname(args.model_path) or "outputs_pinn"
    out_path = os.path.join(out_dir, f"rnn_eval_{model_stem}_bad{bad_x}_eps{args.eps}_test_{safe}.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"[INFO] saved: {out_path}")


if __name__ == "__main__":
    main()
