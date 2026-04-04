import argparse
import json
import os

import numpy as np

from core.blend_helpers import fit_quadratic_poly6, map_raw_to_c
from core.data_pipeline import parse_data_file, stratified_split_by_temp_bin_round_int
from core.sparse_kernel_fit import fit_sparse_kernel_rbf
from core.sparse_kernel_model import export_model_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.path.join("data", "1.txt"))
    ap.add_argument("--N_used", type=int, default=300000)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--M", type=int, default=12)
    ap.add_argument("--lam_kernel", type=float, default=1e-3)

    ap.add_argument("--poly_lam", type=float, default=1e-3)
    ap.add_argument("--T_start_c", type=float, default=55.0)
    ap.add_argument("--margin_c", type=float, default=3.0)

    ap.add_argument("--clip_p_low", type=float, default=1.0)
    ap.add_argument("--clip_p_high", type=float, default=99.0)

    ap.add_argument("--t_min_c", type=float, default=-35.0)
    ap.add_argument("--t_max_c", type=float, default=65.0)

    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X_raw, y_delta = parse_data_file(args.data_path, n_lines=args.N_used)

    T_raw = X_raw[:, 6]
    raw_min = float(np.min(T_raw))
    raw_max = float(np.max(T_raw))

    X_train, y_train, _, _ = stratified_split_by_temp_bin_round_int(
        X_raw, y_delta, test_ratio=args.test_ratio, seed=args.seed
    )

    kernel_out = os.path.join(args.out_dir, f"scheme_c_kernel_model_blend_M{args.M}.json")
    kernel_model, meta = fit_sparse_kernel_rbf(
        X_raw=X_train,
        y=y_train,
        M=args.M,
        lam=args.lam_kernel,
        sigma=None,
        seed=args.seed,
        show_eta=True,
    )
    export_model_json(
        model=kernel_model,
        meta={**meta, "N_used": int(args.N_used), "train_split": "temp_bin_stratified_round_int"},
        export_path=kernel_out,
    )

    T_raw_train = X_train[:, 6]
    T_c_train = map_raw_to_c(T_raw_train, raw_min=raw_min, raw_max=raw_max, t_min_c=args.t_min_c, t_max_c=args.t_max_c)
    poly6 = fit_quadratic_poly6(T_c_train, y_train, lam=args.poly_lam)

    clip_low = np.percentile(y_train, args.clip_p_low, axis=0).astype(np.float64)
    clip_high = np.percentile(y_train, args.clip_p_high, axis=0).astype(np.float64)

    blend_model_out = os.path.join(args.out_dir, "scheme_c_blend_model.json")
    blend_payload = {
        "kernel_model_path": kernel_out,
        "poly6_a2": poly6.a2.tolist(),
        "poly6_a1": poly6.a1.tolist(),
        "poly6_a0": poly6.a0.tolist(),
        "raw_min": raw_min,
        "raw_max": raw_max,
        "t_min_c": float(args.t_min_c),
        "t_max_c": float(args.t_max_c),
        "T_start_c": float(args.T_start_c),
        "margin_c": float(args.margin_c),
        "clip_low": clip_low.tolist(),
        "clip_high": clip_high.tolist(),
        "meta": {
            "N_used": int(args.N_used),
            "test_ratio": float(args.test_ratio),
            "seed": int(args.seed),
            "M": int(args.M),
            "lam_kernel": float(args.lam_kernel),
            "poly_lam": float(args.poly_lam),
            "clip_p_low": float(args.clip_p_low),
            "clip_p_high": float(args.clip_p_high),
        },
    }
    with open(blend_model_out, "w", encoding="utf-8") as f:
        json.dump(blend_payload, f, ensure_ascii=False)

    print("[INFO] saved kernel:", kernel_out)
    print("[INFO] saved blend model:", blend_model_out)


if __name__ == "__main__":
    main()
