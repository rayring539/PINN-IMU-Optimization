import argparse
import os

import numpy as np

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
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--out_model", default=os.path.join("outputs", "scheme_c_sparse_kernel_model.json"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)

    X_raw, y = parse_data_file(args.data_path, n_lines=args.N_used)
    X_train, y_train, _, _ = stratified_split_by_temp_bin_round_int(
        X_raw, y, test_ratio=args.test_ratio, seed=args.seed
    )

    print(f"[INFO] train split: N_train={len(X_train)} N_used={len(X_raw)}")

    model, meta = fit_sparse_kernel_rbf(
        X_raw=X_train,
        y=y_train,
        M=args.M,
        lam=args.lam,
        sigma=None,
        seed=args.seed,
        show_eta=True,
    )
    export_model_json(
        model=model,
        meta={
            **meta,
            "train_split_type": "temp_bin_stratified_round_int",
            "test_ratio": float(args.test_ratio),
            "seed": int(args.seed),
            "N_used": int(args.N_used),
        },
        export_path=args.out_model,
    )
    print(f"[INFO] exported model: {args.out_model}")


if __name__ == "__main__":
    main()
