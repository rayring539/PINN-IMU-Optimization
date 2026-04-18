"""
RNN（LSTM/GRU）基线训练 —— 与 ``train_sparse_kernel`` 相同数据划分（``config/pinn_train.yaml``）。

滑动窗口：用过去 ``seq_len`` 步特征预测 **当前步** δ；大数据集用 ``max_train_windows`` 子采样窗口。

验证集：由 ``rnn_baseline.val_source`` 决定——``test_files`` 用配置中的测试文件；``stratified_temp`` / ``random`` / ``temporal_tail`` 仅从 ``train_files`` 读入后按 ``val_ratio`` 划分（见 YAML 注释）。

用法:
  python train_rnn_baseline.py --config config/pinn_train.yaml
  # 默认另存验证 loss 最优: outputs_pinn/rnn_baseline_best_val.pt；关闭加 --no_save_best_val
  # 损失：rnn_baseline.loss_type（mse|smooth_l1）与 axis_loss_weights；可 CLI 覆盖
  # 学习率策略见 rnn_baseline.lr_schedule；可 --lr_schedule cosine

  python train_rnn_baseline.py --config config/pinn_train.yaml --seq_len 128 --epochs 10 --device cuda
  python train_rnn_baseline.py --config config/pinn_train.yaml --loss_type smooth_l1 --axis_loss_weights 1,1,1,1,1.15,1.15
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, Dataset

_REPO = os.path.abspath(os.path.dirname(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from core.data_pipeline import (
    compute_dTdt,
    parse_data_file,
    split_train_val_random_rows,
    split_train_val_temporal_tail,
    stratified_split_by_temp_bin_round_int,
)
from core.imu_data_io import (
    DEFAULT_TEST_FILES,
    DEFAULT_TRAIN_FILES,
    load_xy_train_only_from_dir,
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


def _normalize_and_resolve_device(device: str) -> str:
    """全角 ``：`` → ``:``；若 ``CUDA_VISIBLE_DEVICES`` 只暴露一张卡，``cuda:1`` 会无效，回退 ``cuda:0``。"""
    device = (device or "cpu").replace("：", ":").strip()
    if device == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    if device == "cuda":
        return "cuda:0"
    if not device.startswith("cuda:"):
        return device
    try:
        idx = int(device.split(":", 1)[1])
    except ValueError:
        return "cuda:0"
    nvis = torch.cuda.device_count()
    if idx >= nvis:
        print(
            f"[WARN] 请求 {device}，但当前进程可见 GPU 数为 {nvis}。"
            "若使用 CUDA_VISIBLE_DEVICES=单卡，进程内设备应为 cuda:0。已改用 cuda:0。",
            file=sys.stderr,
            flush=True,
        )
        return "cuda:0"
    return device


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


def _create_lr_scheduler(
    opt: torch.optim.Optimizer,
    schedule: str,
    epochs: int,
    rb: dict,
) -> tuple[object | None, str, dict[str, object]]:
    """返回 ``(scheduler, name, meta)``；``name`` 为 ``none``/``plateau``/``cosine``/``step``。"""
    name = (schedule or "none").strip().lower()
    meta: dict[str, object] = {"lr_schedule": name}
    if name in ("none", ""):
        return None, "none", meta
    if name == "plateau":
        fac = float(rb.get("lr_plateau_factor", 0.5))
        pat = int(rb.get("lr_plateau_patience", 5))
        min_lr = float(rb.get("lr_plateau_min", 1e-6))
        sch = ReduceLROnPlateau(
            opt, mode="min", factor=fac, patience=pat, min_lr=min_lr
        )
        meta.update(
            {
                "lr_plateau_factor": fac,
                "lr_plateau_patience": pat,
                "lr_plateau_min": min_lr,
            }
        )
        return sch, "plateau", meta
    if name == "cosine":
        tmax = int(rb.get("lr_cosine_t_max", epochs))
        eta_min = float(rb.get("lr_cosine_eta_min", 1e-6))
        sch = CosineAnnealingLR(opt, T_max=tmax, eta_min=eta_min)
        meta.update({"lr_cosine_t_max": tmax, "lr_cosine_eta_min": eta_min})
        return sch, "cosine", meta
    if name == "step":
        step_size = int(rb.get("lr_step_size", 30))
        gamma = float(rb.get("lr_step_gamma", 0.5))
        sch = StepLR(opt, step_size=step_size, gamma=gamma)
        meta.update({"lr_step_size": step_size, "lr_step_gamma": gamma})
        return sch, "step", meta
    raise ValueError(f"未知 lr_schedule: {schedule!r}，可选: none, plateau, cosine, step")


def _parse_axis_loss_weights(rb: dict, cli: str | None) -> np.ndarray:
    """返回长度 6 的逐轴权重（ax..gz）；再在外部除以均值归一化。"""
    if cli is not None and str(cli).strip():
        parts = [float(x.strip()) for x in str(cli).split(",")]
        if len(parts) != 6:
            print(
                "[ERROR] --axis_loss_weights 须为 6 个逗号分隔浮点数（ax,ay,az,gx,gy,gz）",
                file=sys.stderr,
            )
            sys.exit(1)
        return np.asarray(parts, dtype=np.float64)
    aw = rb.get("axis_loss_weights")
    if aw is None:
        return np.ones(6, dtype=np.float64)
    return np.asarray(aw, dtype=np.float64).reshape(6)


def _make_weighted_loss(
    loss_type: str,
    axis_w: torch.Tensor,
    smooth_l1_beta: float,
):
    """axis_w: 已 ``/mean`` 的 (6,) 张量，与 pred 同 device。"""

    def forward(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if loss_type == "mse":
            d = pred - target
            return (axis_w * d * d).mean()
        if loss_type == "smooth_l1":
            sl = F.smooth_l1_loss(pred, target, reduction="none", beta=smooth_l1_beta)
            return (axis_w * sl).mean()
        raise ValueError(loss_type)

    return forward


def _build_val_loader(
    X_raw: np.ndarray,
    y_row: np.ndarray,
    *,
    use_dTdt: bool,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    seq_len: int,
    seed: int,
    max_val_windows: int,
    batch_size: int,
    device: str,
):
    """与训练集同一套 ``x_mean/x_std/y_mean/y_std`` 的归一化监督（验证/留出集；损失与训练一致）。"""
    if use_dTdt:
        td = compute_dTdt(X_raw[:, 6])
        X_fit = np.hstack([X_raw.astype(np.float64), td.astype(np.float64)])
    else:
        X_fit = X_raw.astype(np.float64)
    if len(X_fit) < seq_len:
        return None
    Xn = (X_fit - x_mean) / x_std
    yn = (y_row - y_mean) / y_std
    n_win = len(Xn) - seq_len + 1
    rng_v = np.random.default_rng(int(seed) + 100_003)
    if n_win > max_val_windows:
        idx_val = rng_v.choice(n_win, size=max_val_windows, replace=False)
    else:
        idx_val = np.arange(n_win, dtype=np.int64)
    ds_v = _WindowDataset(Xn, yn, seq_len, idx_val)
    return DataLoader(
        ds_v,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 2),
        pin_memory=str(device).startswith("cuda"),
        drop_last=False,
    )


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

    ap.add_argument("--rnn_type", choices=("LSTM", "GRU"), default=None)
    ap.add_argument(
        "--bidirectional",
        type=int,
        choices=(0, 1),
        default=None,
        help="1=BiLSTM/BiGRU（双向）；0=单向；默认读 rnn_baseline.bidirectional",
    )
    ap.add_argument("--seq_len", type=int, default=None)
    ap.add_argument("--hidden_dim", type=int, default=None)
    ap.add_argument("--num_layers", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument(
        "--lr_schedule",
        choices=("none", "plateau", "cosine", "step"),
        default=None,
        help="学习率策略：none | plateau(按 val_mse 无改善则降) | cosine | step；默认读 rnn_baseline.lr_schedule",
    )
    ap.add_argument("--max_train_windows", type=int, default=None,
                    help="训练窗口数上限（随机子采样）；全量过大时必用")
    ap.add_argument("--max_val_windows", type=int, default=None,
                    help="验证集窗口子采样上限（与 train 同一归一化统计；默认读 rnn_baseline.max_val_windows）")
    ap.add_argument(
        "--val_source",
        choices=("test_files", "stratified_temp", "random", "temporal_tail"),
        default=None,
        help="val_mse 数据来源：test_files=配置中测试文件；其余=仅从 train_files 读入后划 val_ratio",
    )
    ap.add_argument("--val_ratio", type=float, default=None, help="从训练数据中留出验证比例（默认读 rnn_baseline.val_ratio）")
    ap.add_argument(
        "--no_save_best_val",
        action="store_true",
        help="关闭「验证集 MSE 最优」权重的额外保存（默认开启，见 rnn_baseline.save_best_val）",
    )
    ap.add_argument("--device", default=None, help="cuda / cpu；默认读 YAML training / rnn_baseline.device")
    ap.add_argument(
        "--loss_type",
        choices=("mse", "smooth_l1"),
        default=None,
        help="归一化 δ 监督：mse | smooth_l1；默认读 rnn_baseline.loss_type",
    )
    ap.add_argument(
        "--smooth_l1_beta",
        type=float,
        default=None,
        help="SmoothL1 的 beta（PyTorch）；仅 smooth_l1；默认读 rnn_baseline.smooth_l1_beta",
    )
    ap.add_argument(
        "--axis_loss_weights",
        default=None,
        help="6 个逗号分隔权重 ax,ay,az,gx,gy,gz；默认读 rnn_baseline.axis_loss_weights",
    )
    args = ap.parse_args()

    use_config = args.config and os.path.isfile(args.config)
    ycfg: dict | None = None
    if use_config:
        ycfg = _load_yaml(args.config)
        rb0 = ycfg.get("rnn_baseline") or {}
        if args.val_source is None:
            args.val_source = str(rb0.get("val_source", "test_files"))
        if args.val_ratio is None:
            args.val_ratio = float(rb0.get("val_ratio", 0.2))
    else:
        if args.val_source is None:
            args.val_source = "test_files"
        if args.val_ratio is None:
            args.val_ratio = 0.2

    if use_config:
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
        vr = float(args.val_ratio)
        if args.val_source != "test_files":
            X_all, y_all, _ = load_xy_train_only_from_dir(cfg)
            if args.val_source == "stratified_temp":
                X_train, y_train, X_test_split, y_test_split = stratified_split_by_temp_bin_round_int(
                    X_all, y_all, test_ratio=vr, seed=seed
                )
            elif args.val_source == "random":
                X_train, y_train, X_test_split, y_test_split = split_train_val_random_rows(
                    X_all, y_all, vr, seed
                )
            elif args.val_source == "temporal_tail":
                X_train, y_train, X_test_split, y_test_split = split_train_val_temporal_tail(
                    X_all, y_all, vr
                )
            else:
                print(f"[ERROR] 未知 val_source: {args.val_source}", file=sys.stderr)
                sys.exit(1)
            split_note = f"yaml_val_from_train_{args.val_source}_r{vr}"
            print(
                f"[INFO] 训练仅用 train_files 子集 N={len(X_train)}；"
                f"val_mse 用留出验证 N={len(X_test_split)}（val_ratio={vr}，{args.val_source}）",
                flush=True,
            )
        else:
            X_train, y_train, X_test_split, y_test_split, _ = load_xy_train_test_from_dir(cfg)
            split_note = "yaml_val_test_files"
    else:
        if not args.data_path:
            print("[ERROR] 使用 --config 或 --data_path", file=sys.stderr)
            sys.exit(1)
        N_used = int(args.N_used) if args.N_used is not None else 300000
        seed = int(args.seed) if args.seed is not None else 0
        use_dTdt = bool(args.use_dTdt) if args.use_dTdt is not None else False
        out_dir = args.out_dir or "outputs_pinn"
        X_raw, y = parse_data_file(args.data_path, n_lines=N_used)
        X_train, y_train, X_test_split, y_test_split = stratified_split_by_temp_bin_round_int(
            X_raw, y, test_ratio=0.2, seed=seed
        )
        split_note = "legacy_stratified"

    rb = (ycfg.get("rnn_baseline") or {}) if ycfg else {}
    if args.rnn_type is None:
        rt = str(rb.get("rnn_type", "LSTM")).upper()
        args.rnn_type = "GRU" if rt == "GRU" else "LSTM"
    if args.bidirectional is None:
        args.bidirectional = bool(rb.get("bidirectional", False))
    else:
        args.bidirectional = bool(args.bidirectional)
    if args.seq_len is None:
        args.seq_len = int(rb.get("seq_len", 64))
    if args.hidden_dim is None:
        args.hidden_dim = int(rb.get("hidden_dim", 128))
    if args.num_layers is None:
        args.num_layers = int(rb.get("num_layers", 2))
    if args.dropout is None:
        args.dropout = float(rb.get("dropout", 0.0))
    if args.epochs is None:
        args.epochs = int(rb.get("epochs", 5))
    if args.batch_size is None:
        args.batch_size = int(rb.get("batch_size", 256))
    if args.lr is None:
        args.lr = float(rb.get("lr", 1e-3))
    if args.weight_decay is None:
        args.weight_decay = float(rb.get("weight_decay", 1e-5))
    if args.max_train_windows is None:
        args.max_train_windows = int(rb.get("max_train_windows", 500_000))
    if args.max_val_windows is None:
        args.max_val_windows = int(rb.get("max_val_windows", 200_000))
    save_best_val = bool(rb.get("save_best_val", True))
    if args.no_save_best_val:
        save_best_val = False
    if args.lr_schedule is None:
        args.lr_schedule = str(rb.get("lr_schedule", "none"))

    if args.loss_type is None:
        args.loss_type = str(rb.get("loss_type", "mse")).strip().lower()
    if args.loss_type not in ("mse", "smooth_l1"):
        args.loss_type = "mse"
    if args.smooth_l1_beta is None:
        args.smooth_l1_beta = float(rb.get("smooth_l1_beta", 1.0))
    axis_w_raw = _parse_axis_loss_weights(rb, args.axis_loss_weights)
    if axis_w_raw.size != 6 or (not np.all(np.isfinite(axis_w_raw))) or np.any(axis_w_raw <= 0):
        print("[ERROR] axis_loss_weights 须为 6 个有限正数（ax..gz）", file=sys.stderr)
        sys.exit(1)

    if args.device is None:
        if rb.get("device"):
            args.device = str(rb["device"])
        elif ycfg:
            tr = ycfg.get("training") or {}
            if tr.get("force_cpu"):
                args.device = "cpu"
            elif tr.get("cuda_device") is not None:
                args.device = f"cuda:{int(tr['cuda_device'])}"
            elif isinstance(tr.get("gpu_ids"), (list, tuple)) and len(tr["gpu_ids"]) > 0:
                args.device = f"cuda:{int(tr['gpu_ids'][0])}"
            else:
                args.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            args.device = "cuda" if torch.cuda.is_available() else "cpu"

    args.device = _normalize_and_resolve_device(str(args.device))

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
    device = args.device
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 2),
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )

    dl_val = _build_val_loader(
        X_test_split,
        y_test_split,
        use_dTdt=use_dTdt,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        seq_len=seq_len,
        seed=seed,
        max_val_windows=int(args.max_val_windows),
        batch_size=args.batch_size,
        device=device,
    )
    if dl_val is not None:
        if split_note.startswith("yaml_val_from_train"):
            print(
                "[INFO] val_mse_norm: 从 train_files 按 val_source/val_ratio 留出验证集（与训练集同一归一化统计）",
                flush=True,
            )
        elif split_note == "yaml_val_test_files":
            print(
                "[INFO] val_mse_norm: 使用 config 中 test_files 拼接集（与训练集同一归一化统计）",
                flush=True,
            )
        else:
            print(
                "[INFO] val_mse_norm: legacy 单文件分层划分中的留出集（与训练集同一归一化统计）",
                flush=True,
            )

    model = IMURNNBaseline(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    axis_w_norm = axis_w_raw / axis_w_raw.mean()
    axis_w_t = torch.tensor(axis_w_norm, dtype=torch.float32, device=device)
    loss_fn = _make_weighted_loss(
        args.loss_type, axis_w_t, float(args.smooth_l1_beta)
    )
    print(
        f"[INFO] loss_type={args.loss_type}  smooth_l1_beta={args.smooth_l1_beta}  "
        f"axis_loss_weights={axis_w_raw.tolist()}  "
        f"(mean-norm={np.round(axis_w_norm, 4).tolist()})",
        flush=True,
    )
    sched, sched_name, sched_meta = _create_lr_scheduler(
        opt, args.lr_schedule, int(args.epochs), rb
    )
    if sched_name != "none":
        print(f"[INFO] lr_schedule={sched_name}  config={sched_meta}", flush=True)

    t0 = time.time()
    train_mse = float("nan")
    val_mse_epoch: float | None = None
    best_val_mse: float | None = None
    best_epoch = 0
    best_state_cpu: dict | None = None
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
        train_loss = total / max(n_b, 1)
        val_part = ""
        val_loss_epoch = None
        if dl_val is not None:
            model.eval()
            tv = 0.0
            nv = 0
            with torch.no_grad():
                for xbv, ybv in dl_val:
                    xbv = xbv.to(device)
                    ybv = ybv.to(device)
                    pred_v = model(xbv)
                    tv += float(loss_fn(pred_v, ybv).item()) * xbv.size(0)
                    nv += xbv.size(0)
            val_loss_epoch = tv / max(nv, 1)
            val_part = f"  val_loss_norm={val_loss_epoch:.6f}"
            if save_best_val and (
                best_val_mse is None or val_loss_epoch < best_val_mse
            ):
                best_val_mse = val_loss_epoch
                best_epoch = ep + 1
                best_state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                val_part += "  [best_val]"
        cur_lr = float(opt.param_groups[0]["lr"])
        print(
            f"[epoch {ep+1}/{args.epochs}] train_loss_norm={train_loss:.6f}{val_part}  "
            f"lr={cur_lr:.2e}  "
            f"elapsed={time.time()-t0:.1f}s",
            flush=True,
        )
        if sched is not None:
            if sched_name == "plateau":
                metric = val_loss_epoch if val_loss_epoch is not None else train_loss
                prev_lr = float(opt.param_groups[0]["lr"])
                sched.step(metric)
                new_lr = float(opt.param_groups[0]["lr"])
                if new_lr < prev_lr - 1e-20:
                    print(
                        f"[INFO] ReduceLROnPlateau: lr {prev_lr:.2e} -> {new_lr:.2e}",
                        flush=True,
                    )
            else:
                sched.step()
        train_mse = train_loss
        val_mse_epoch = val_loss_epoch

    out_model = args.out_model or os.path.join(out_dir, "rnn_baseline.pt")
    _od = os.path.dirname(os.path.abspath(out_model))
    if _od:
        os.makedirs(_od, exist_ok=True)

    def _meta_dict(checkpoint_kind: str, **extra):
        m = {
            "model_type": "IMURNNBaseline",
            "checkpoint_kind": checkpoint_kind,
            "split_note": split_note,
            "val_source": args.val_source,
            "val_ratio": float(args.val_ratio),
            "N_val_rows": int(len(X_test_split)),
            "use_dTdt": use_dTdt,
            "input_dim": input_dim,
            "seq_len": seq_len,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "rnn_type": args.rnn_type,
            "bidirectional": args.bidirectional,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_train_windows": int(args.max_train_windows),
            "max_val_windows": int(args.max_val_windows),
            "n_windows_used": len(idx),
            "N_train_rows": N,
            "seed": seed,
            "x_mean": x_mean.astype(np.float64),
            "x_std": x_std.astype(np.float64),
            "y_mean": y_mean.astype(np.float64),
            "y_std": y_std.astype(np.float64),
            "config_path": os.path.abspath(args.config) if use_config else None,
            "loss_type": args.loss_type,
            "axis_loss_weights": axis_w_raw.astype(np.float64).tolist(),
            "axis_loss_weights_mean_normalized": axis_w_norm.astype(np.float64).tolist(),
            "smooth_l1_beta": float(args.smooth_l1_beta) if args.loss_type == "smooth_l1" else None,
            "train_loss_norm_last": float(train_mse),
            "val_loss_norm_last": None if val_mse_epoch is None else float(val_mse_epoch),
            "train_mse_last": float(train_mse),
            "val_mse_last": None if val_mse_epoch is None else float(val_mse_epoch),
        }
        m.update(extra)
        return m

    meta_last = _meta_dict(
        "last",
        **sched_meta,
    )
    save_checkpoint(out_model, model, meta_last)
    json_path = os.path.splitext(out_model)[0] + "_meta.json"
    export_meta_json(json_path, meta_last)
    print(f"[INFO] saved (last epoch): {out_model}\n[INFO] meta: {json_path}")

    if save_best_val and best_state_cpu is not None and best_val_mse is not None:
        stem, ext = os.path.splitext(out_model)
        out_best = f"{stem}_best_val{ext}"
        model.load_state_dict({k: v.to(device) for k, v in best_state_cpu.items()})
        meta_best = _meta_dict(
            "best_val",
            best_val_mse_norm=float(best_val_mse),
            best_epoch=int(best_epoch),
            epochs_trained=int(args.epochs),
            **sched_meta,
        )
        save_checkpoint(out_best, model, meta_best)
        json_best = os.path.splitext(out_best)[0] + "_meta.json"
        export_meta_json(json_best, meta_best)
        print(
            f"[INFO] saved (best val_loss_norm={best_val_mse:.6f} @ epoch {best_epoch}): {out_best}\n"
            f"[INFO] meta: {json_best}",
            flush=True,
        )
    elif save_best_val and dl_val is None:
        print("[INFO] 无验证集，跳过 best_val 权重保存", flush=True)


if __name__ == "__main__":
    main()
