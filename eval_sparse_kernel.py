import argparse
import json
import os

import numpy as np

from core.data_pipeline import parse_data_file, stratified_split_by_temp_bin_round_int
from core.sparse_kernel_model import load_model_json


def mae_rmse(y_true: np.ndarray, y_pred: np.ndarray):
    err = y_pred - y_true
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err * err, axis=0))
    return mae.tolist(), rmse.tolist()


def coverage_within_threshold(corrected_true: np.ndarray, corrected_pred: np.ndarray, eps: float):
    err = np.abs(corrected_pred - corrected_true)
    ok = err <= eps
    cov = np.mean(ok, axis=0)
    return cov.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.path.join("data", "1.txt"))
    ap.add_argument("--N_used", type=int, default=300000)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_path", default=os.path.join("outputs", "scheme_c_sparse_kernel_model.json"))
    ap.add_argument("--eps", type=float, default=5.0, help="coverage threshold |corrected_pred-corrected_true|<=eps")
    args = ap.parse_args()

    X_raw, y_delta_true = parse_data_file(args.data_path, n_lines=args.N_used)
    _, _, X_test, y_delta_test = stratified_split_by_temp_bin_round_int(
        X_raw, y_delta_true, test_ratio=args.test_ratio, seed=args.seed
    )

    model = load_model_json(args.model_path)
    y_delta_pred = model.predict_delta6(X_test)

    raw6_test = X_test[:, :6]
    corrected_true = raw6_test + y_delta_test
    corrected_pred = raw6_test + y_delta_pred

    mae, rmse = mae_rmse(corrected_true, corrected_pred)
    cov_per_dim = coverage_within_threshold(corrected_true, corrected_pred, eps=float(args.eps))

    result = {
        "model_path": args.model_path,
        "split_type": "temp_bin_stratified_round_int",
        "N_used": int(args.N_used),
        "test_ratio": float(args.test_ratio),
        "seed": int(args.seed),
        "eps": float(args.eps),
        "metrics_corrected_lsb": {
            "mae_per_dim": mae,
            "rmse_per_dim": rmse,
        },
        "coverage_per_dim": cov_per_dim,
        "N_test": int(len(X_test)),
    }

    print("[INFO] eval result:\n" + json.dumps(result, ensure_ascii=False, indent=2))

    out_path = os.path.join(
        "outputs",
        f"scheme_c_sparse_kernel_eval_eps{args.eps}_ratio{args.test_ratio}_seed{args.seed}.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"[INFO] saved: {out_path}")


if __name__ == "__main__":
    main()
