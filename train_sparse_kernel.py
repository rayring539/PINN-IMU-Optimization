"""
稀疏 RBF 核岭回归基线 —— 与 PINN 相同数据划分与 YAML 配置。

用法:
  # 与 train_pinn_dde 相同 data_dir + explicit 划分（读 config/pinn_train.yaml）
  python train_sparse_kernel.py --config config/pinn_train.yaml

  # 仍支持单文件 + 温度分层旧流程
  python train_sparse_kernel.py --data_path data/1.txt --test_ratio 0.2
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
from core.imu_data_io import (
    DEFAULT_TEST_FILES,
    DEFAULT_TRAIN_FILES,
    load_xy_train_test_from_dir,
    parse_file_list_arg,
)
from core.sparse_kernel_fit import fit_sparse_kernel_rbf
from core.sparse_kernel_model import export_model_json


def _load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _deep_get(d: dict, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def main():
    ap = argparse.ArgumentParser(description="核岭回归基线训练（与 pinn_train.yaml 数据一致）")
    ap.add_argument("--config", default=None, help="与 train_pinn_dde 相同，含 data / model / output")
    ap.add_argument("--data_path", default=None, help="单文件模式：数据路径（与 --config 二选一）")
    ap.add_argument("--data_dir", default=None, help="覆盖 YAML 的 data.data_dir")
    ap.add_argument("--split_mode", choices=("explicit", "random"), default=None)
    ap.add_argument("--train_files", default=None, help="逗号分隔，如 1.txt,2.txt,3.txt,4.txt")
    ap.add_argument("--test_files", default=None, help="逗号分隔，如 5.txt")
    ap.add_argument("--N_used", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out_dir", default=None, help="覆盖 YAML output.out_dir；写出 train_test_split.json")
    ap.add_argument("--out_model", default=None, help="模型 JSON 路径；默认 <out_dir>/sparse_kernel_model.json")
    ap.add_argument("--M", type=int, default=12)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--use_dTdt", type=int, choices=(0, 1), default=None,
                    help="1=输入拼 Tdot（8 维），与 model.use_dTdt 一致；不设则从 YAML 读")
    args = ap.parse_args()

    use_config = args.config and os.path.isfile(args.config)
    if use_config:
        ycfg = _load_yaml(args.config)
        data_dir = args.data_dir or _deep_get(ycfg, "data", "data_dir")
        if not data_dir:
            print("[ERROR] config 中需有 data.data_dir，或传 --data_dir", file=sys.stderr)
            sys.exit(1)
        out_dir = args.out_dir or _deep_get(ycfg, "output", "out_dir", default="outputs_pinn")
        split_mode = args.split_mode or _deep_get(ycfg, "data", "split_mode", default="explicit")
        seed = int(args.seed) if args.seed is not None else int(_deep_get(ycfg, "data", "seed", default=0))
        N_used = int(args.N_used) if args.N_used is not None else int(_deep_get(ycfg, "data", "N_used", default=-1))
        tf = parse_file_list_arg(args.train_files)
        te = parse_file_list_arg(args.test_files)
        train_files = tf if tf is not None else _deep_get(ycfg, "data", "train_files", default=None)
        test_files = te if te is not None else _deep_get(ycfg, "data", "test_files", default=None)
        if split_mode == "explicit":
            if not train_files:
                train_files = list(DEFAULT_TRAIN_FILES)
            if not test_files:
                test_files = list(DEFAULT_TEST_FILES)
        use_dTdt = bool(args.use_dTdt) if args.use_dTdt is not None else bool(
            _deep_get(ycfg, "model", "use_dTdt", default=False)
        )

        cfg = {
            "data_dir": data_dir,
            "out_dir": out_dir,
            "split_mode": split_mode,
            "train_files": train_files,
            "test_files": test_files,
            "seed": seed,
            "N_used": N_used,
        }
        X_train, y_train, _, _, split_meta = load_xy_train_test_from_dir(cfg)
        split_note = f"yaml+explicit {split_meta.get('train_basenames')}"
    else:
        if not args.data_path:
            print("[ERROR] 请使用 --config config/pinn_train.yaml 或 --data_path 单文件", file=sys.stderr)
            sys.exit(1)
        N_used = int(args.N_used) if args.N_used is not None else 300000
        seed = int(args.seed) if args.seed is not None else 0
        test_ratio = 0.2
        use_dTdt = bool(args.use_dTdt) if args.use_dTdt is not None else False
        X_raw, y = parse_data_file(args.data_path, n_lines=N_used)
        X_train, y_train, _, _ = stratified_split_by_temp_bin_round_int(
            X_raw, y, test_ratio=test_ratio, seed=seed
        )
        split_meta = {"train_split_type": "temp_bin_stratified_round_int", "test_ratio": test_ratio}
        split_note = "legacy single-file stratified"

    if use_dTdt:
        Tdot = compute_dTdt(X_train[:, 6])
        X_fit = np.hstack([X_train.astype(np.float64), Tdot.astype(np.float64)])
    else:
        X_fit = X_train.astype(np.float64)

    if use_config:
        out_model = args.out_model or os.path.join(out_dir, "sparse_kernel_model.json")
    else:
        out_model = args.out_model or os.path.join("outputs", "scheme_c_sparse_kernel_model.json")
    _od = os.path.dirname(os.path.abspath(out_model))
    if _od:
        os.makedirs(_od, exist_ok=True)

    print(f"[INFO] split: {split_note}  N_train={len(X_train)}  input_dim={X_fit.shape[1]}  use_dTdt={use_dTdt}")

    model, meta_fit = fit_sparse_kernel_rbf(
        X_raw=X_fit,
        y=y_train,
        M=args.M,
        lam=args.lam,
        sigma=None,
        seed=seed,
        show_eta=True,
    )
    export_model_json(
        model=model,
        meta={
            **meta_fit,
            "train_split_type": split_note,
            "use_dTdt": use_dTdt,
            "config_path": os.path.abspath(args.config) if use_config else None,
            "M": args.M,
            "lam": args.lam,
            "seed": seed,
            "N_train": len(X_train),
        },
        export_path=out_model,
    )
    print(f"[INFO] exported: {out_model}")


if __name__ == "__main__":
    main()
