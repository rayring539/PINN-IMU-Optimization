import argparse
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np

from core.blend_model import load_blend_model_json
from core.data_pipeline import parse_data_file
from core.sparse_kernel_model import load_model_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.path.join(_REPO, "data", "1.txt"))
    ap.add_argument("--N_used", type=int, default=300000)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kernel_model_path", default=os.path.join(_REPO, "outputs", "scheme_c_kernel_model_blend_M12.json"))
    ap.add_argument("--blend_model_path", default=os.path.join(_REPO, "outputs", "scheme_c_blend_model.json"))
    ap.add_argument("--T_high_c", type=float, default=55.0)
    ap.add_argument("--eps", type=float, default=5.0)
    args = ap.parse_args()

    X_raw, y_delta = parse_data_file(args.data_path, n_lines=args.N_used)

    kernel = load_model_json(args.kernel_model_path)
    blend = load_blend_model_json(args.blend_model_path)

    delta_kernel = kernel.predict_delta6(X_raw)
    delta_blend = blend.predict_delta6(X_raw)

    raw6 = X_raw[:, :6]
    corrected_true_all = raw6 + y_delta
    corrected_kernel_all = raw6 + delta_kernel
    corrected_blend_all = raw6 + delta_blend

    T_c_all = blend.map_raw_to_c(X_raw[:, 6])
    hi_mask = T_c_all >= args.T_high_c
    if np.sum(hi_mask) == 0:
        print("[WARN] no samples in high temp region (check mapping/raw range)")
        return

    names = ["ax", "ay", "az", "gx", "gy", "gz"]
    dims = 6

    def summarize(corrected_pred, corrected_true, label: str):
        err = np.abs(corrected_pred - corrected_true)
        mae = np.mean(err, axis=0)
        rmse = np.sqrt(np.mean((corrected_pred - corrected_true) ** 2, axis=0))
        cov = np.mean(err[hi_mask] <= args.eps, axis=0)
        max_err = np.max(err[hi_mask], axis=0)
        viol = np.sum(err[hi_mask] > args.eps, axis=0)
        print(f"\n=== {label} | high_temp_count={int(np.sum(hi_mask))} | eps={args.eps} ===")
        for d in range(dims):
            print(f"{names[d]}: high_cov={cov[d]:.4f}, high_max_abs_err={max_err[d]:.2f}, viol={int(viol[d])}")
        print(f"overall_mae_per_dim: {mae.tolist()}")
        print(f"overall_rmse_per_dim: {rmse.tolist()}")

    summarize(corrected_kernel_all, corrected_true_all, "Kernel")
    summarize(corrected_blend_all, corrected_true_all, "Blend")


if __name__ == "__main__":
    main()
