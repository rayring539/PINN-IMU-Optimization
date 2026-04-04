import argparse
import json
import os

import numpy as np

from core.blend_model import load_blend_model_json
from core.data_pipeline import parse_data_file


def mae_rmse(a: np.ndarray, b: np.ndarray):
    err = b - a
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
    ap.add_argument(
        "--split_meta",
        default=os.path.join("outputs_stream_full", "train_test_split.json"),
        help="多文件训练时由 train_blend_stream.py 写入，内含 test_file 绝对路径",
    )
    ap.add_argument("--test_data_path", default=None, help="直接指定测试集 txt；若设置则忽略 split_meta")
    ap.add_argument("--data_path", default=None, help="兼容旧版：单文件路径（与 split_meta 二选一）")
    ap.add_argument("--N_used", type=int, default=300000, help="测试集最多读取行数；-1 表示读满整个测试文件")
    ap.add_argument("--blend_model_path", default=os.path.join("outputs_stream_full", "scheme_c_blend_model_stream.json"))
    ap.add_argument("--eps", type=float, default=5.0, help="coverage threshold on corrected6: |pred-true|<=eps")
    args = ap.parse_args()

    if args.test_data_path:
        test_path = args.test_data_path
        split_note = "explicit_test_data_path"
    elif args.data_path:
        test_path = args.data_path
        split_note = "legacy_single_file_all_rows_as_test_subset"
    else:
        with open(args.split_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        test_path = meta["test_file"]
        split_note = f"split_meta test_file={meta.get('test_basename', '')}"

    n_lines = None if args.N_used < 0 else args.N_used
    X_test, y_test = parse_data_file(test_path, n_lines=n_lines)

    blend_model = load_blend_model_json(args.blend_model_path)
    delta_pred = blend_model.predict_delta6(X_test)

    raw6_test = X_test[:, :6]
    corrected_true = raw6_test + y_test
    corrected_pred = raw6_test + delta_pred

    mae, rmse = mae_rmse(corrected_true, corrected_pred)
    cov_per_dim = coverage_within_threshold(corrected_true, corrected_pred, eps=float(args.eps))

    result = {
        "blend_model_path": args.blend_model_path,
        "split_type": split_note,
        "test_data_path": test_path,
        "N_used": int(args.N_used),
        "eps": float(args.eps),
        "metrics_corrected_lsb": {
            "mae_per_dim": mae,
            "rmse_per_dim": rmse,
        },
        "coverage_per_dim": cov_per_dim,
        "N_test": int(len(X_test)),
        "note": "delta6_pred = (1-w)*delta_kernel + w*delta_poly, then per-dim clip",
    }

    print("[INFO] eval result:\n" + json.dumps(result, ensure_ascii=False, indent=2))

    safe_name = os.path.splitext(os.path.basename(test_path))[0]
    out_path = os.path.join(
        os.path.dirname(args.blend_model_path) or "outputs",
        f"scheme_c_blend_eval_eps{args.eps}_test_{safe_name}.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"[INFO] saved: {out_path}")


if __name__ == "__main__":
    main()
