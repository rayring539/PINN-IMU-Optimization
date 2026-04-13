"""
RNN（LSTM/GRU）基线训练 —— 与 ``train_sparse_kernel`` 相同数据划分（``config/pinn_train.yaml``）。

滑动窗口：用过去 ``seq_len`` 步特征预测 **当前步** δ；大数据集用 ``max_train_windows`` 子采样窗口。

用法:
  python train_rnn_baseline.py --config config/pinn_train.yaml
  python train_rnn_baseline.py --config config/pinn_train.yaml --seq_len 128 --epochs 10 --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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
from core.rnn_imu_baseline import IMURNNBaseline, export_meta_json, save_checkpoint


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


class _WindowDataset(Dataset):
    """每个样本：``X[s:s+seq_len]`` → 监督 ``y[s+seq_len-1]``。"""

    def __init__(self, Xn: np.ndarray, yn: np.ndarray, seq_len: int, indices: np.ndarray):
        self.Xn = Xn.astype(np.float32)
        self.yn = yn.astype(np.float32)
        self.seq_len = int(seq_len)
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        s = int(self.indices[i])
        xw = self.Xn[s : s + self.seq_len]
        y_t = self.yn[s + self.seq_len - 1]
        return torch.from_numpy(xw), torch.from_numpy(y_t)


def main():
    ap = argparse.ArgumentParser(description="RNN 基线训练（LSTM/GRU）")
    ap.add_argument("--config", default=None)
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--split_mode", choices=("explicit", "random"), default=None)
    ap.add_argument("--train_files", default=None)
    ap.add_argument("--test_files", default=None)
    ap.add_argument("--N_used", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--out_model", default=None, help="默认 <out_dir>/rnn_baseline.pt")
    ap.add_argument("--use_dTdt", type=int, choices=(0, 1), default=None)

    ap.add_argument("--rnn_type", choices=("LSTM", "GRU"), default="LSTM")
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--max_train_windows", type=int, default=500_000,
                    help="训练窗口数上限（随机子采样）；全量过大时必用")
    ap.add_argument("--device", default=None, help="cuda / cpu；默认自动")
    args = ap.parse_args()

    use_config = args.config and os.path.isfile(args.config)
    if use_config:
        ycfg = _load_yaml(args.config)
        data_dir = args.data_dir or _deep_get(ycfg, "data", "data_dir")
        if not data_dir:
            print("[ERROR] 需 data.data_dir 或 --data_dir", file=sys.stderr)
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
        X_train, y_train, _, _, _ = load_xy_train_test_from_dir(cfg)
        split_note = "yaml_explicit"
    else:
        if not args.data_path:
            print("[ERROR] 使用 --config 或 --data_path", file=sys.stderr)
            sys.exit(1)
        N_used = int(args.N_used) if args.N_used is not None else 300000
        seed = int(args.seed) if args.seed is not None else 0
        use_dTdt = bool(args.use_dTdt) if args.use_dTdt is not None else False
        out_dir = args.out_dir or "outputs_pinn"
        X_raw, y = parse_data_file(args.data_path, n_lines=N_used)
        X_train, y_train, _, _ = stratified_split_by_temp_bin_round_int(
            X_raw, y, test_ratio=0.2, seed=seed
        )
        split_note = "legacy_stratified"

    if use_dTdt:
        Tdot = compute_dTdt(X_train[:, 6])
        X_fit = np.hstack([X_train.astype(np.float64), Tdot.astype(np.float64)])
    else:
        X_fit = X_train.astype(np.float64)

    input_dim = X_fit.shape[1]
    N = len(X_fit)
    seq_len = max(2, int(args.seq_len))
    if N < seq_len:
        print(f"[ERROR] 训练样本数 N={N} < seq_len={seq_len}", file=sys.stderr)
        sys.exit(1)

    x_mean = X_fit.mean(axis=0)
    x_std = X_fit.std(axis=0) + 1e-8
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0) + 1e-8
    Xn = (X_fit - x_mean) / x_std
    yn = (y_train - y_mean) / y_std

    n_win = N - seq_len + 1
    rng = np.random.default_rng(seed)
    if n_win > args.max_train_windows:
        idx = rng.choice(n_win, size=args.max_train_windows, replace=False)
    else:
        idx = np.arange(n_win, dtype=np.int64)

    ds = _WindowDataset(Xn, yn, seq_len, idx)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 2),
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )

    model = IMURNNBaseline(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        total = 0.0
        n_b = 0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.size(0)
            n_b += xb.size(0)
        print(f"[epoch {ep+1}/{args.epochs}] train_mse_norm={total / max(n_b,1):.6f}  elapsed={time.time()-t0:.1f}s")

    out_model = args.out_model or os.path.join(out_dir, "rnn_baseline.pt")
    _od = os.path.dirname(os.path.abspath(out_model))
    if _od:
        os.makedirs(_od, exist_ok=True)

    meta = {
        "model_type": "IMURNNBaseline",
        "split_note": split_note,
        "use_dTdt": use_dTdt,
        "input_dim": input_dim,
        "seq_len": seq_len,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "rnn_type": args.rnn_type,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_train_windows": int(args.max_train_windows),
        "n_windows_used": len(idx),
        "N_train_rows": N,
        "seed": seed,
        "x_mean": x_mean.astype(np.float64),
        "x_std": x_std.astype(np.float64),
        "y_mean": y_mean.astype(np.float64),
        "y_std": y_std.astype(np.float64),
        "config_path": os.path.abspath(args.config) if use_config else None,
    }
    save_checkpoint(out_model, model, meta)
    json_path = os.path.splitext(out_model)[0] + "_meta.json"
    export_meta_json(json_path, meta)
    print(f"[INFO] saved: {out_model}\n[INFO] meta: {json_path}")


if __name__ == "__main__":
    main()
