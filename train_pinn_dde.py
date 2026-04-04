"""
PINN 训练入口 (DeepXDE 版)。

与 train_pinn.py 功能等价, 但利用 DeepXDE 框架简化训练循环:
  - dde.Model.compile() + train() 替代手动 epoch 循环
  - dde.callbacks 替代手动日志 / 检查点
  - dde.Variable  用于可学习物理参数 (反问题)
  - 支持 Adam → L-BFGS 两阶段训练

用法:
  python train_pinn_dde.py --config config/pinn_train.yaml
  python train_pinn_dde.py --config config/pinn_train.yaml --epochs 500
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

# DeepXDE 路径 & 后端
_REPO = os.path.abspath(os.path.dirname(__file__))
_DDE  = os.path.join(_REPO, "deepxde")
if _DDE not in sys.path:
    sys.path.insert(0, _DDE)
os.environ.setdefault("DDE_BACKEND", "pytorch")

import deepxde as dde  # noqa: E402
import torch  # noqa: E402

from core.pinn_dde import (  # noqa: E402
    ACC_SCALE, GYRO_SCALE, TEMP_SCALE,
    IMUNet, IMUPINNData, PhysicsWarmup, ProgressBar, TensorBoardCallback,
    save_dde_checkpoint, export_onnx,
)
from core.data_pipeline import (  # noqa: E402
    parse_data_file,
    stratified_split_by_temp_bin_round_int,
)
from core.imu_data_io import (  # noqa: E402
    save_split_meta,
    split_train_test_by_random_file,
)


# -----------------------------------------------------------------------
#  工具
# -----------------------------------------------------------------------
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


def compute_dTdt(T: np.ndarray) -> np.ndarray:
    dT = np.zeros_like(T)
    dT[1:-1] = (T[2:] - T[:-2]) / 2.0
    if len(T) > 1:
        dT[0]  = T[1] - T[0]
        dT[-1] = T[-1] - T[-2]
    return dT.reshape(-1, 1)


# -----------------------------------------------------------------------
#  配置
# -----------------------------------------------------------------------
def build_cfg() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--N_used", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--lbfgs_iters", type=int, default=None,
                    help="Adam 后接 L-BFGS 精调迭代数 (0=禁用)")
    args = ap.parse_args()

    cfg: dict = {}
    if args.config and os.path.isfile(args.config):
        cfg = _load_yaml(args.config)

    def g(*keys, default=None):
        return _deep_get(cfg, *keys, default=default)

    c: dict = {}
    c["data_dir"]   = args.data_dir  or g("data", "data_dir")
    c["data_path"]  = args.data_path or g("data", "data_path")
    c["N_used"]     = args.N_used    if args.N_used is not None else g("data", "N_used", default=-1)
    c["test_ratio"] = g("data", "test_ratio", default=0.2)
    c["seed"]       = g("data", "seed", default=0)

    c["acc_scale"]  = g("sensor", "acc_scale",  default=ACC_SCALE)
    c["gyro_scale"] = g("sensor", "gyro_scale", default=GYRO_SCALE)
    c["temp_scale"] = g("sensor", "temp_scale", default=TEMP_SCALE)

    c["hidden_dim"]        = g("model", "hidden_dim", default=64)
    c["n_hidden"]          = g("model", "n_hidden", default=4)
    c["thermal_dim"]       = g("model", "thermal_dim", default=16)
    c["act"]               = g("model", "act", default="tanh")
    c["use_dTdt"]          = g("model", "use_dTdt", default=True)
    c["use_hysteresis"]    = g("model", "use_hysteresis", default=False)
    c["hysteresis_hidden"] = g("model", "hysteresis_hidden", default=16)

    c["epochs"]         = args.epochs     if args.epochs     is not None else g("training", "epochs", default=300)
    c["batch_size"]     = args.batch_size if args.batch_size is not None else g("training", "batch_size", default=2048)
    c["lr"]             = args.lr         if args.lr         is not None else g("training", "lr", default=1e-3)
    c["weight_decay"]   = g("training", "weight_decay", default=1e-5)
    c["physics_warmup"] = g("training", "physics_warmup", default=30)
    c["log_interval"]   = g("training", "log_interval", default=10)
    c["lbfgs_iters"]    = args.lbfgs_iters if args.lbfgs_iters is not None else g("training", "lbfgs_iters", default=0)

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

    c["out_dir"] = args.out_dir or g("output", "out_dir", default="D:\\IMU_output")
    return c


# -----------------------------------------------------------------------
#  数据加载
# -----------------------------------------------------------------------
def load_data(cfg: dict):
    if cfg.get("data_dir"):
        train_paths, test_path, split_meta = split_train_test_by_random_file(
            cfg["data_dir"], seed=cfg["seed"])
        os.makedirs(cfg["out_dir"], exist_ok=True)
        save_split_meta(split_meta,
                        os.path.join(cfg["out_dir"], "train_test_split.json"))
        xs, ys, total = [], [], 0
        n_limit = cfg["N_used"]
        for fp in train_paths:
            remain = None if n_limit < 0 else max(0, n_limit - total)
            if remain is not None and remain <= 0:
                break
            xf, yf = parse_data_file(fp, n_lines=remain)
            xs.append(xf); ys.append(yf); total += len(xf)
        X_train, y_train = np.concatenate(xs), np.concatenate(ys)
        test_n = None if cfg["N_used"] < 0 else cfg["N_used"]
        X_test, y_test = parse_data_file(test_path, n_lines=test_n)
    else:
        n_lines = None if cfg["N_used"] < 0 else cfg["N_used"]
        X_raw, y = parse_data_file(cfg["data_path"], n_lines=n_lines)
        X_train, y_train, X_test, y_test = stratified_split_by_temp_bin_round_int(
            X_raw, y, test_ratio=cfg["test_ratio"], seed=cfg["seed"])
    return X_train, y_train, X_test, y_test


# -----------------------------------------------------------------------
#  主流程
# -----------------------------------------------------------------------
def main():
    cfg = build_cfg()
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2, default=str)

    # ---- 数据 ----
    print("[DATA] 加载数据 ...")
    X_train, y_train, X_test, y_test = load_data(cfg)

    if cfg["use_dTdt"]:
        Tdot_train = compute_dTdt(X_train[:, 6])
        Tdot_test  = compute_dTdt(X_test[:, 6])
        X_train_aug = np.hstack([X_train, Tdot_train])
        X_test_aug  = np.hstack([X_test,  Tdot_test])
    else:
        X_train_aug, X_test_aug = X_train, X_test

    T_c = X_train[:, 6] / cfg["temp_scale"]
    print(f"[DATA] train={len(X_train)}  test={len(X_test)}  "
          f"T_℃=[{T_c.min():.1f}, {T_c.max():.1f}]")

    x_mean = X_train.mean(axis=0)
    x_std  = X_train.std(axis=0) + 1e-8
    y_std  = y_train.std(axis=0) + 1e-8

    # ---- DeepXDE Data ----
    data = IMUPINNData(X_train_aug, y_train, X_test_aug, y_test)

    # ---- Network ----
    net = IMUNet(cfg, x_mean, x_std, y_std)
    if cfg["weight_decay"] > 0:
        net.regularizer = ("l2", cfg["weight_decay"])

    n_params = net.num_trainable_parameters()
    kb = n_params * 4 / 1024
    groups: dict[str, int] = {}
    for name, p in net.named_parameters():
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
    print("[MODEL] 参数统计 (float32):")
    for k, v in groups.items():
        print(f"  {k:12s} {v:>8,d}  ({100 * v / n_params:5.1f}%)")
    print(f"  {'--------':12s} {'--------':>8s}")
    print(f"  {'TOTAL':12s} {n_params:>8,d}  ({kb:.1f} KB)")
    edge_ok = "YES" if n_params < 100_000 else "NO"
    print(f"  边缘设备适配: {edge_ok}  (<100K 参数阈值)")
    print(f"  use_dTdt={cfg['use_dTdt']}  use_hysteresis={cfg['use_hysteresis']}")

    # ---- 外部可学习物理参数 (dde.Variable) ----
    log_k_E   = dde.Variable(math.log(60e-6))
    log_alpha = dde.Variable(math.log(2.6e-6))
    log_da    = dde.Variable(math.log(5e-6))
    ext_vars  = [log_k_E, log_alpha, log_da]

    # ---- dde.Model ----
    model = dde.Model(data, net)

    lam = cfg["lam"]
    loss_weights = [
        1.0,
        lam["L_heat_smooth"],
        lam["L_physics_prior"],
        lam["L_stiffness_mono"],
        lam["L_acc_tdb"],
        lam["L_three_factor"],
        lam["L_gyro_smooth"],
        lam["L_residual_small"],
        lam["L_grad_smooth"],
    ]

    N_train = len(X_train_aug)
    bs = cfg["batch_size"]
    iters_per_epoch = max(1, N_train // bs)
    total_iters = cfg["epochs"] * iters_per_epoch
    warmup_iters = cfg["physics_warmup"] * iters_per_epoch
    display_every = max(1, cfg["log_interval"] * iters_per_epoch)

    model.compile(
        "adam",
        lr=cfg["lr"],
        loss="MSE",
        loss_weights=loss_weights,
        external_trainable_variables=ext_vars,
        decay=("cosine", total_iters, cfg["lr"] * 0.01),
    )

    # ---- Callbacks ----
    tb_dir = os.path.join(out_dir, "tb")
    callbacks = [
        PhysicsWarmup(loss_weights, warmup_iters),
        ProgressBar(total_iters, desc="Adam"),
        TensorBoardCallback(tb_dir),
        dde.callbacks.ModelCheckpoint(
            os.path.join(out_dir, "dde_ckpt"),
            save_better_only=True,
            period=display_every,
        ),
        dde.callbacks.VariableValue(
            ext_vars,
            period=display_every,
            filename=os.path.join(out_dir, "physics_vars.dat"),
            precision=8,
        ),
    ]
    print(f"[TB] TensorBoard log → {tb_dir}")

    # ---- 第一阶段: Adam ----
    print(f"\n[TRAIN] Adam × {total_iters} iters "
          f"({cfg['epochs']} epochs × {iters_per_epoch} iters/epoch)")
    t0 = time.time()
    losshistory, train_state = model.train(
        iterations=total_iters,
        batch_size=bs,
        display_every=display_every,
        callbacks=callbacks,
        model_save_path=os.path.join(out_dir, "model_adam"),
    )

    # ---- 第二阶段: L-BFGS (可选) ----
    lbfgs_iters = cfg["lbfgs_iters"]
    if lbfgs_iters and lbfgs_iters > 0:
        print(f"\n[TRAIN] L-BFGS × {lbfgs_iters} iters (fine-tuning)")
        dde.optimizers.set_LBFGS_options(maxiter=lbfgs_iters)
        model.compile(
            "L-BFGS",
            loss="MSE",
            loss_weights=loss_weights,
            external_trainable_variables=ext_vars,
        )
        lbfgs_display = max(1, lbfgs_iters // 10)
        losshistory, train_state = model.train(
            display_every=lbfgs_display,
            callbacks=[
                ProgressBar(lbfgs_iters, desc="L-BFGS"),
                dde.callbacks.VariableValue(
                    ext_vars, period=lbfgs_display,
                    precision=8,
                ),
            ],
        )

    elapsed = time.time() - t0
    print(f"\n[DONE] elapsed={elapsed:.1f}s")
    print(f"[TB] tensorboard --logdir {tb_dir}")

    # ---- 保存完整检查点 (兼容 eval/plot) ----
    meta = {
        "hyperparams": {
            "hidden_dim": cfg["hidden_dim"],
            "n_hidden": cfg["n_hidden"],
            "thermal_dim": cfg["thermal_dim"],
            "act": cfg["act"],
            "use_dTdt": cfg["use_dTdt"],
            "use_hysteresis": cfg["use_hysteresis"],
            "hysteresis_hidden": cfg["hysteresis_hidden"],
            "acc_scale": cfg["acc_scale"],
            "gyro_scale": cfg["gyro_scale"],
            "temp_scale": cfg["temp_scale"],
            "x_mean": x_mean.tolist(),
            "x_std":  x_std.tolist(),
            "y_mean": y_train.mean(axis=0).tolist(),
            "y_std":  y_std.tolist(),
        },
        "training": {
            "framework": "deepxde",
            "epochs": cfg["epochs"],
            "lr": cfg["lr"],
            "batch_size": bs,
            "lam": cfg["lam"],
            "lbfgs_iters": lbfgs_iters,
        },
        "best_step": int(train_state.best_step),
        "best_loss": float(sum(train_state.best_loss_test)),
    }
    ckpt_path = os.path.join(out_dir, "pinn_model_best.pt")
    save_dde_checkpoint(model, ext_vars, meta, ckpt_path)
    print(f"  -> saved {ckpt_path}")

    dde.saveplot(losshistory, train_state,
                 issave=True, isplot=False, output_dir=out_dir)

    # ---- ONNX 导出 (用于量化 / 边缘部署) ----
    onnx_path = os.path.join(out_dir, "pinn_model.onnx")
    export_onnx(model.net, onnx_path)


if __name__ == "__main__":
    main()
