"""
PINN 训练入口 v2。

优先从 YAML 读取配置，CLI 参数可覆盖。

用法:
  python train_pinn.py --config config/pinn_train.yaml
  python train_pinn.py --config config/pinn_train.yaml --epochs 500
  python train_pinn.py --data_dir /path/to/IMU_data --epochs 300   # 无 YAML 也可
  python train_pinn.py --data_dir D:\\IMU_data --epochs 300         # Windows
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from core.data_pipeline import parse_data_file, stratified_split_by_temp_bin_round_int
from core.imu_data_io import (
    DEFAULT_DATA_DIR,
    DEFAULT_TEST_FILES,
    DEFAULT_TRAIN_FILES,
    load_xy_train_test_from_dir,
    parse_file_list_arg,
)
from core.pinn_model import PINN_IMU, PINNPhysicsLoss, save_pinn_checkpoint


def print_model_summary(model: nn.Module) -> int:
    """按分支统计参数量并打印摘要，返回总参数量。"""
    groups: dict[str, int] = {}
    for name, p in model.named_parameters():
        prefix = name.split(".")[0]
        if any(prefix.startswith(x) for x in ("acc_", "gyro_", "T_ref", "log_")):
            key = "physics"
        elif prefix == "thermal_net":
            key = "thermal"
        elif prefix == "residual_net":
            key = "residual"
        elif prefix.startswith("hyst"):
            key = "hysteresis"
        else:
            key = prefix
        groups[key] = groups.get(key, 0) + p.numel()
    total = sum(groups.values())
    kb = total * 4 / 1024
    print("[MODEL] 参数统计 (float32):")
    for k, v in groups.items():
        print(f"  {k:12s} {v:>8,d}  ({100 * v / total:5.1f}%)")
    print(f"  {'--------':12s} {'--------':>8s}")
    print(f"  {'TOTAL':12s} {total:>8,d}  ({kb:.1f} KB)")
    edge_ok = "YES" if total < 100_000 else "NO"
    print(f"  边缘设备适配: {edge_ok}  (<100K 参数阈值)")
    return total


# -----------------------------------------------------------------------
#  配置加载
# -----------------------------------------------------------------------
def _load_yaml(path: str) -> dict:
    import yaml  # pyyaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _deep_get(d: dict, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def _coerce_int(val, field: str) -> int:
    """YAML 中若写 2048*10 会得到字符串；统一转为 int。"""
    if isinstance(val, bool):
        raise TypeError(f"{field}: 需要整数，不要用 YAML 布尔值")
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str):
        s = "".join(val.split())
        if not s:
            raise ValueError(f"{field}: 空字符串")
        if "*" in s:
            p = 1
            for part in s.split("*"):
                if not part:
                    raise ValueError(f"{field}: 无效的乘法表达式 {val!r}")
                p *= int(float(part))
            return p
        return int(float(s))
    raise TypeError(f"{field}: 需要整数，得到 {type(val).__name__}")


def build_cfg() -> dict:
    """合并 YAML + CLI → 统一配置字典。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML 配置文件路径")

    # 以下 CLI 参数可覆盖 YAML 中同名字段
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--N_used", type=int, default=None)
    ap.add_argument("--test_ratio", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--hidden_dim", type=int, default=None)
    ap.add_argument("--n_hidden", type=int, default=None)
    ap.add_argument("--thermal_dim", type=int, default=None)
    ap.add_argument("--act", default=None)
    ap.add_argument("--use_dTdt", type=int, default=None)
    ap.add_argument("--use_hysteresis", type=int, default=None)
    ap.add_argument("--physics_warmup", type=int, default=None)
    ap.add_argument("--split_mode", choices=("random", "explicit"), default=None)
    ap.add_argument("--train_files", default=None)
    ap.add_argument("--test_files", default=None)
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--cuda_device", type=int, default=None)
    args = ap.parse_args()

    # 基础配置 (若无 YAML 则用全默认)
    cfg: dict = {}
    if args.config and os.path.isfile(args.config):
        cfg = _load_yaml(args.config)

    def g(*keys, default=None):
        return _deep_get(cfg, *keys, default=default)

    # 合并: CLI 优先 > YAML > 默认值
    c: dict = {}

    # 数据
    c["data_dir"]    = args.data_dir   or g("data", "data_dir")
    c["data_path"]   = args.data_path  or g("data", "data_path")
    c["N_used"]      = args.N_used     if args.N_used is not None else g("data", "N_used", default=-1)
    c["test_ratio"]  = args.test_ratio if args.test_ratio is not None else g("data", "test_ratio", default=0.2)
    c["seed"]        = args.seed       if args.seed is not None else g("data", "seed", default=0)

    sm = args.split_mode or g("data", "split_mode", default="explicit")
    c["split_mode"] = sm
    tf = parse_file_list_arg(args.train_files)
    te = parse_file_list_arg(args.test_files)
    c["train_files"] = tf if tf is not None else g("data", "train_files", default=None)
    c["test_files"] = te if te is not None else g("data", "test_files", default=None)
    if c["split_mode"] == "explicit":
        if not c["train_files"]:
            c["train_files"] = list(DEFAULT_TRAIN_FILES)
        if not c["test_files"]:
            c["test_files"] = list(DEFAULT_TEST_FILES)

    # 传感器标度因子
    c["acc_scale"]  = g("sensor", "acc_scale",  default=2048.0)
    c["gyro_scale"] = g("sensor", "gyro_scale", default=16.0)
    c["temp_scale"] = g("sensor", "temp_scale", default=256.0)

    # 网络
    c["hidden_dim"]       = args.hidden_dim   if args.hidden_dim is not None else g("model", "hidden_dim", default=64)
    c["n_hidden"]         = args.n_hidden     if args.n_hidden is not None else g("model", "n_hidden", default=4)
    c["thermal_dim"]      = args.thermal_dim  if args.thermal_dim is not None else g("model", "thermal_dim", default=16)
    c["act"]              = args.act          or g("model", "act", default="tanh")
    c["use_dTdt"]         = bool(args.use_dTdt) if args.use_dTdt is not None else g("model", "use_dTdt", default=True)
    c["use_hysteresis"]   = bool(args.use_hysteresis) if args.use_hysteresis is not None else g("model", "use_hysteresis", default=False)
    c["hysteresis_hidden"] = g("model", "hysteresis_hidden", default=16)
    c["temp_only"]        = bool(g("model", "temp_only", default=False))

    # 训练
    c["epochs"]          = args.epochs     if args.epochs is not None else g("training", "epochs", default=300)
    c["batch_size"]      = args.batch_size if args.batch_size is not None else g("training", "batch_size", default=2048)
    c["lr"]              = args.lr         if args.lr is not None else g("training", "lr", default=1e-3)
    c["weight_decay"]    = g("training", "weight_decay", default=1e-5)
    c["grad_clip"]       = g("training", "grad_clip", default=5.0)
    c["log_interval"]    = g("training", "log_interval", default=10)
    c["physics_warmup"]  = args.physics_warmup if args.physics_warmup is not None else g("training", "physics_warmup", default=30)

    # 物理损失权重
    ploss = g("physics_loss") or {}
    c["lam"] = {
        "L_heat_smooth":    ploss.get("L_heat_smooth",    1e-2),
        "L_physics_prior":  ploss.get("L_physics_prior",  1e-1),
        "L_stiffness_mono": ploss.get("L_stiffness_mono", 1e-2),
        "L_acc_tdb":        ploss.get("L_acc_tdb",        1e-2),
        "L_three_factor":   ploss.get("L_three_factor",   1e-2),
        "L_gyro_smooth":    ploss.get("L_gyro_smooth",    1e-2),
        "L_residual_small": ploss.get("L_residual_small", 1e-3),
        "L_grad_smooth":    ploss.get("L_grad_smooth",    1e-3),
    }

    # 输出
    c["out_dir"] = args.out_dir or g("output", "out_dir", default="outputs_pinn")

    def _parse_gpu_ids(s):
        if s is None:
            return None
        if isinstance(s, list):
            return [int(x) for x in s]
        s = str(s).strip()
        if not s:
            return None
        return [int(x.strip()) for x in s.split(",") if x.strip()]

    cli_gpus = _parse_gpu_ids(args.gpus)
    yaml_gpus = g("training", "gpu_ids", default=None)
    cuda_def = args.cuda_device if args.cuda_device is not None else g("training", "cuda_device", default=0)
    if cli_gpus is not None:
        c["gpu_ids"] = cli_gpus
    elif yaml_gpus is not None:
        c["gpu_ids"] = _parse_gpu_ids(yaml_gpus)
    else:
        c["gpu_ids"] = [_coerce_int(cuda_def, "training.cuda_device")]

    c["N_used"] = _coerce_int(c["N_used"], "data.N_used")
    c["seed"] = _coerce_int(c["seed"], "data.seed")
    c["epochs"] = _coerce_int(c["epochs"], "training.epochs")
    c["batch_size"] = _coerce_int(c["batch_size"], "training.batch_size")
    c["physics_warmup"] = _coerce_int(c["physics_warmup"], "training.physics_warmup")
    c["log_interval"] = _coerce_int(c["log_interval"], "training.log_interval")
    c["hidden_dim"] = _coerce_int(c["hidden_dim"], "model.hidden_dim")
    c["n_hidden"] = _coerce_int(c["n_hidden"], "model.n_hidden")
    c["thermal_dim"] = _coerce_int(c["thermal_dim"], "model.thermal_dim")
    c["hysteresis_hidden"] = _coerce_int(c["hysteresis_hidden"], "model.hysteresis_hidden")
    return c


# -----------------------------------------------------------------------
#  数据加载 + dT/dt 计算
# -----------------------------------------------------------------------
def compute_dTdt(T: np.ndarray) -> np.ndarray:
    """有限差分近似 dT/dt (假设等间隔采样)。返回 (N,1)。"""
    dT = np.zeros_like(T)
    dT[1:-1] = (T[2:] - T[:-2]) / 2.0  # 中心差分
    dT[0]    = T[1] - T[0] if len(T) > 1 else 0.0
    dT[-1]   = T[-1] - T[-2] if len(T) > 1 else 0.0
    return dT.reshape(-1, 1)


def load_data(cfg: dict):
    """加载数据并计算各集合上的 dT/dt。

    - 若配置 ``data_path``：单文件 + 按温度分层划分 train/test。
    - 否则：多文件目录 ``data_dir``（未设则用 ``DEFAULT_DATA_DIR``），走 explicit/random。

    需在调用前保证 ``cfg['out_dir']`` 已存在（``load_xy_train_test_from_dir`` 会写划分元数据）。
    返回 (X_train, y_train, Tdot_train, X_test, y_test, Tdot_test, split_meta)。
    """
    if cfg.get("data_path"):
        n_lines = None if cfg["N_used"] < 0 else cfg["N_used"]
        X_raw, y = parse_data_file(cfg["data_path"], n_lines=n_lines)
        X_train, y_train, X_test, y_test = stratified_split_by_temp_bin_round_int(
            X_raw, y, test_ratio=cfg["test_ratio"], seed=cfg["seed"]
        )
        split_meta = None
    else:
        if not cfg.get("data_dir"):
            cfg["data_dir"] = DEFAULT_DATA_DIR
        X_train, y_train, X_test, y_test, split_meta = load_xy_train_test_from_dir(
            cfg
        )

    Tdot_train = compute_dTdt(X_train[:, 6])
    Tdot_test = compute_dTdt(X_test[:, 6])

    return X_train, y_train, Tdot_train, X_test, y_test, Tdot_test, split_meta


# -----------------------------------------------------------------------
#  训练主函数
# -----------------------------------------------------------------------
def main():
    cfg = build_cfg()
    os.makedirs(cfg["out_dir"], exist_ok=True)

    # 保存本次使用的配置
    with open(os.path.join(cfg["out_dir"], "train_config.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in cfg.items()}, f, ensure_ascii=False, indent=2, default=str)

    # ---- 数据 ----
    print("[DATA] 加载数据 ...")
    (X_train, y_train, Tdot_train,
     X_test, y_test, Tdot_test,
     split_meta) = load_data(cfg)

    T_c_range = X_train[:, 6] / cfg["temp_scale"]
    print(f"[DATA] train={len(X_train)}  test={len(X_test)}  "
          f"T_℃=[{T_c_range.min():.1f}, {T_c_range.max():.1f}]")

    x_mean = X_train.mean(axis=0)
    x_std  = X_train.std(axis=0) + 1e-8
    y_mean = y_train.mean(axis=0)
    y_std  = y_train.std(axis=0) + 1e-8

    gpu_ids = [int(x) for x in cfg["gpu_ids"]]
    use_dp = torch.cuda.is_available() and len(gpu_ids) > 1
    if torch.cuda.is_available():
        for gid in gpu_ids:
            if gid < 0 or gid >= torch.cuda.device_count():
                raise ValueError(
                    f"无效 GPU id={gid}，当前可见设备数={torch.cuda.device_count()}")
        torch.cuda.set_device(gpu_ids[0])
        device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        device = torch.device("cpu")
        use_dp = False

    if use_dp:
        print(f"[DEVICE] DataParallel  gpu_ids={gpu_ids}  主卡 {device}")
    else:
        print(f"[DEVICE] {device}")

    # ---- 模型 ----
    model = PINN_IMU(
        hidden_dim=cfg["hidden_dim"],
        n_hidden=cfg["n_hidden"],
        thermal_dim=cfg["thermal_dim"],
        act=cfg["act"],
        use_dTdt=cfg["use_dTdt"],
        use_hysteresis=cfg["use_hysteresis"],
        hysteresis_hidden=cfg["hysteresis_hidden"],
        temp_only=cfg.get("temp_only", False),
        acc_scale=cfg["acc_scale"],
        gyro_scale=cfg["gyro_scale"],
        temp_scale=cfg["temp_scale"],
        x_mean=x_mean, x_std=x_std,
        y_mean=y_mean, y_std=y_std,
    ).to(device)
    if use_dp:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    n_params = print_model_summary(
        model.module if hasattr(model, "module") else model)
    print(f"  use_dTdt={cfg['use_dTdt']}  use_hysteresis={cfg['use_hysteresis']}  "
          f"temp_only={cfg.get('temp_only', False)}")

    # ---- DataLoader (包含 Tdot) ----
    Xt = torch.from_numpy(X_train.astype(np.float32))
    yt = torch.from_numpy(y_train.astype(np.float32))
    Tt = torch.from_numpy(Tdot_train.astype(np.float32))
    loader = DataLoader(
        TensorDataset(Xt, yt, Tt),
        batch_size=cfg["batch_size"], shuffle=True,
        drop_last=False, pin_memory=(device.type == "cuda"),
    )

    # ---- 优化器 ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01
    )

    lam = cfg["lam"]
    best_test_rmse = float("inf")
    t0 = time.time()

    tb_dir = os.path.join(
        cfg["out_dir"], "tb", f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    writer = SummaryWriter(log_dir=tb_dir)
    print(f"[TB] TensorBoard log → {tb_dir}")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        ep_data = ep_phys = 0.0
        ep_terms: dict[str, float] = {k: 0.0 for k in lam}
        nb = 0
        warmup = min(1.0, epoch / max(1, cfg["physics_warmup"]))

        for xb, yb, tb in loader:
            xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)

            T_col = xb[:, 6:7].detach().requires_grad_(True)
            xb_grad = torch.cat([xb[:, :6], T_col], dim=1)

            delta_pred, phys_part, hyst_part, res_part, _ = model(
                xb_grad, Tdot=tb, return_parts=True
            )

            loss_data = F.mse_loss(delta_pred, yb)

            ploss = PINNPhysicsLoss.compute(
                model, xb_grad, delta_pred, T_col,
                phys_part, res_part, Tdot=tb,
            )
            loss_phys = sum(lam[k] * warmup * v for k, v in ploss.items())

            loss = loss_data + loss_phys

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["grad_clip"])
            optimizer.step()

            ep_data += loss_data.item()
            ep_phys += (loss_phys.item() if isinstance(loss_phys, torch.Tensor)
                        else loss_phys)
            for k, v in ploss.items():
                ep_terms[k] += (v.item() if isinstance(v, torch.Tensor) else v)
            nb += 1

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        avg_d = ep_data / max(nb, 1)
        avg_p = ep_phys / max(nb, 1)

        # ---- TensorBoard: 每 epoch 写 loss / meta / lr ----
        writer.add_scalar("train/data", avg_d, epoch)
        writer.add_scalar("train/physics_total", avg_p, epoch)
        writer.add_scalar("train/total", avg_d + avg_p, epoch)
        writer.add_scalar("meta/epoch", float(epoch), epoch)
        writer.add_scalar("train/lr", lr_now, epoch)
        for k, v in ep_terms.items():
            writer.add_scalar(f"train/physics_{k}", v / max(nb, 1), epoch)

        # ---- 验证 ----
        do_log = (epoch % cfg["log_interval"] == 0
                  or epoch == 1 or epoch == cfg["epochs"])
        if do_log:
            model.eval()
            delta_test = model.predict_delta6(
                X_test, Tdot_np=Tdot_test if cfg["use_dTdt"] else None
            )
            corr_true = X_test[:, :6] + y_test
            corr_pred = X_test[:, :6] + delta_test
            test_rmse = float(np.sqrt(np.mean((corr_pred - corr_true) ** 2)))

            print(f"[E{epoch:04d}] data={avg_d:.4f}  phys={avg_p:.6f}  "
                  f"test_rmse={test_rmse:.4f}  lr={lr_now:.2e}  "
                  f"elapsed={time.time()-t0:.1f}s")

            writer.add_scalar("test/rmse_lsb", test_rmse, epoch)

            if test_rmse < best_test_rmse:
                best_test_rmse = test_rmse
                _save(model, cfg, x_mean, x_std, y_mean, y_std,
                      split_meta, epoch, test_rmse, "best")

    _save(model, cfg, x_mean, x_std, y_mean, y_std,
          split_meta, cfg["epochs"], best_test_rmse, "final")
    writer.flush()
    writer.close()
    print(f"[DONE] best_test_rmse={best_test_rmse:.4f}  "
          f"elapsed={time.time()-t0:.1f}s")
    print(f"[TB] tensorboard --logdir {os.path.join(cfg['out_dir'], 'tb')}")


def _save(model, cfg, x_mean, x_std, y_mean, y_std,
          split_meta, epoch, test_rmse, tag):
    meta = {
        "hyperparams": {
            "hidden_dim": cfg["hidden_dim"],
            "n_hidden": cfg["n_hidden"],
            "thermal_dim": cfg["thermal_dim"],
            "act": cfg["act"],
            "use_dTdt": cfg["use_dTdt"],
            "use_hysteresis": cfg["use_hysteresis"],
            "hysteresis_hidden": cfg["hysteresis_hidden"],
            "temp_only": bool(cfg.get("temp_only", False)),
            "acc_scale": cfg["acc_scale"],
            "gyro_scale": cfg["gyro_scale"],
            "temp_scale": cfg["temp_scale"],
            "x_mean": x_mean.tolist(),
            "x_std": x_std.tolist(),
            "y_mean": y_mean.tolist(),
            "y_std": y_std.tolist(),
        },
        "training": {
            "epochs": cfg["epochs"],
            "lr": cfg["lr"],
            "batch_size": cfg["batch_size"],
            "lam": cfg["lam"],
            "physics_warmup": cfg["physics_warmup"],
            "seed": cfg["seed"],
            "gpu_ids": cfg["gpu_ids"],
            "split_mode": cfg.get("split_mode"),
        },
        "best_epoch": epoch,
        "test_rmse": test_rmse,
        "split_meta": split_meta,
    }
    path = os.path.join(cfg["out_dir"], f"pinn_model_{tag}.pt")
    save_pinn_checkpoint(model, meta, path)
    print(f"  -> saved {path}")


if __name__ == "__main__":
    main()
