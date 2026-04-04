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
  python train_pinn_dde.py --config config/pinn_train.yaml --cpu   # 强制 CPU
  python train_pinn_dde.py --config config/pinn_train.yaml --gpus 0,1,2,3
  python train_pinn_dde.py --config config/pinn_train.yaml --mixed  # DeepXDE 内置 AMP（非 HF accelerate）
  python train_pinn_dde.py --split_mode explicit --train_files 1.txt,2.txt,3.txt,4.txt --test_files 5.txt
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
import torch.nn as nn  # noqa: E402

from core.pinn_dde import (  # noqa: E402
    ACC_SCALE, GYRO_SCALE, TEMP_SCALE,
    IMUNet, IMUPINNData, PhysicsWarmup, ProgressBar, TensorBoardCallback,
    export_onnx, save_dde_checkpoint, unwrap_parallel_net,
)
from core.pinn_metrics import make_delta_val_metrics  # noqa: E402
from core.data_pipeline import (  # noqa: E402
    parse_data_file,
    stratified_split_by_temp_bin_round_int,
)
from core.imu_data_io import (  # noqa: E402
    DEFAULT_TEST_FILES,
    DEFAULT_TRAIN_FILES,
    load_xy_train_test_from_dir,
    parse_file_list_arg,
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
    ap.add_argument("--cpu", action="store_true",
                    help="强制使用 CPU（默认在可用时使用 CUDA）")
    ap.add_argument("--cuda_device", type=int, default=None,
                    help="单卡训练时的 GPU 索引（默认 0；多卡请用 --gpus）")
    ap.add_argument("--gpus", default=None,
                    help="多卡 DataParallel，逗号分隔 GPU 索引，如 0,1,2,3")
    ap.add_argument("--mixed", action="store_true",
                    help="启用 DeepXDE 混合精度 (torch.autocast AMP)。与 Hugging Face accelerate 库无关")
    ap.add_argument("--split_mode", choices=("random", "explicit"), default=None,
                    help="数据划分: random=随机留一文件作测试; explicit=用 train_files/test_files")
    ap.add_argument("--train_files", default=None,
                    help="显式训练文件，逗号分隔，如 1.txt,2.txt,3.txt,4.txt")
    ap.add_argument("--test_files", default=None,
                    help="显式测试文件，逗号分隔，如 5.txt")
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
    c["temp_only"]         = bool(g("model", "temp_only", default=False))

    c["epochs"]         = args.epochs     if args.epochs     is not None else g("training", "epochs", default=300)
    c["batch_size"]     = args.batch_size if args.batch_size is not None else g("training", "batch_size", default=2048)
    c["lr"]             = args.lr         if args.lr         is not None else g("training", "lr", default=1e-3)
    c["weight_decay"]   = g("training", "weight_decay", default=1e-5)
    c["physics_warmup"] = g("training", "physics_warmup", default=30)
    c["log_interval"]   = g("training", "log_interval", default=10)
    c["lbfgs_iters"]    = args.lbfgs_iters if args.lbfgs_iters is not None else g("training", "lbfgs_iters", default=0)
    c["val_bad_threshold"] = float(g("training", "val_bad_threshold", default=5.0))

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

    c["out_dir"] = args.out_dir or g("output", "out_dir", default="outputs_pinn")
    c["force_cpu"] = bool(args.cpu) or bool(g("training", "force_cpu", default=False))
    c["mixed_precision"] = bool(args.mixed) or bool(g("training", "mixed_precision", default=False))

    def _parse_gpu_ids(s: str | list | None) -> list[int] | None:
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
    c["lbfgs_iters"] = _coerce_int(c["lbfgs_iters"], "training.lbfgs_iters")
    c["hidden_dim"] = _coerce_int(c["hidden_dim"], "model.hidden_dim")
    c["n_hidden"] = _coerce_int(c["n_hidden"], "model.n_hidden")
    c["thermal_dim"] = _coerce_int(c["thermal_dim"], "model.thermal_dim")
    c["hysteresis_hidden"] = _coerce_int(c["hysteresis_hidden"], "model.hysteresis_hidden")
    return c


# -----------------------------------------------------------------------
#  数据加载
# -----------------------------------------------------------------------
def load_data(cfg: dict):
    if cfg.get("data_dir"):
        X_train, y_train, X_test, y_test, _ = load_xy_train_test_from_dir(cfg)
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
    if cfg.get("mixed_precision"):
        dde.config.set_default_float("mixed")
        if cfg.get("force_cpu"):
            print(
                "[WARN] mixed_precision 已开启但 force_cpu=True；"
                "AMP 在 CPU 上收益有限，建议 GPU 训练时使用。"
            )
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

    # ---- 设备 (GPU: 将 NumPy batch 与模型对齐到同一 device，见 deepxde model.py 补丁) ----
    gpu_ids = [int(x) for x in cfg["gpu_ids"]]
    use_dp = (
        not cfg["force_cpu"]
        and torch.cuda.is_available()
        and len(gpu_ids) > 1
    )
    if cfg["force_cpu"]:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        for gid in gpu_ids:
            if gid < 0 or gid >= torch.cuda.device_count():
                raise ValueError(
                    f"无效 GPU id={gid}，当前可见设备数={torch.cuda.device_count()}")
        primary = int(gpu_ids[0])
        torch.cuda.set_device(primary)
        device = torch.device(f"cuda:{primary}")
    else:
        device = torch.device("cpu")
        use_dp = False
        print(
            "[DEVICE] 未检测到 CUDA（torch.cuda.is_available()=False）。"
            "若已安装 NVIDIA 驱动，请安装带 CUDA 的 PyTorch: "
            "https://pytorch.org/get-started/locally/"
        )

    # ---- Network ----
    net = IMUNet(cfg, x_mean, x_std, y_std)
    if cfg["weight_decay"] > 0:
        net.regularizer = ("l2", cfg["weight_decay"])
    net = net.to(device)
    if use_dp:
        net = nn.DataParallel(net, device_ids=gpu_ids)

    if device.type == "cuda":
        pname = torch.cuda.get_device_name(device)
        cap = torch.cuda.get_device_capability(device)
        if use_dp:
            print(
                f"[DEVICE] DataParallel  gpu_ids={gpu_ids}  主卡 {device}  {pname}  (cap {cap})"
            )
        else:
            print(f"[DEVICE] {device}  {pname}  (cap {cap})")
    else:
        print("[DEVICE] cpu")

    n_params = unwrap_parallel_net(net).num_trainable_parameters()
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
    print(f"  use_dTdt={cfg['use_dTdt']}  use_hysteresis={cfg['use_hysteresis']}  "
          f"temp_only={cfg.get('temp_only', False)}")

    # ---- 外部可学习物理参数 (dde.Variable) ----
    log_k_E   = dde.Variable(math.log(60e-6))
    log_alpha = dde.Variable(math.log(2.6e-6))
    log_da    = dde.Variable(math.log(5e-6))
    ext_vars  = [log_k_E, log_alpha, log_da]
    ext_vars = [t.detach().to(device).requires_grad_(True) for t in ext_vars]

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

    val_metrics, val_metric_names = make_delta_val_metrics(cfg["val_bad_threshold"])

    model.compile(
        "adam",
        lr=cfg["lr"],
        loss="MSE",
        loss_weights=loss_weights,
        external_trainable_variables=ext_vars,
        decay=("cosine", total_iters, cfg["lr"] * 0.01),
        metrics=val_metrics,
    )

    # ---- Callbacks ----
    # 每次启动独立 run_*，查看时指向 tb 根目录即可对比多次训练（与参考仓库一致）
    tb_root = os.path.join(out_dir, "tb")
    tb_dir = os.path.join(tb_root, f"run_{time.strftime('%Y%m%d_%H%M%S')}")
    callbacks = [
        PhysicsWarmup(loss_weights, warmup_iters),
        ProgressBar(total_iters, desc="Adam"),
        TensorBoardCallback(
            tb_dir,
            iters_per_epoch=iters_per_epoch,
            val_metric_names=val_metric_names,
        ),
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
    print(f"[TB] 本次日志目录: {tb_dir}")
    print(f"[TB] 启动查看: tensorboard --logdir {tb_root}")

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
            metrics=val_metrics,
        )
        lbfgs_display = max(1, lbfgs_iters // 10)
        losshistory, train_state = model.train(
            display_every=lbfgs_display,
            callbacks=[
                ProgressBar(lbfgs_iters, desc="L-BFGS"),
                TensorBoardCallback(
                    tb_dir,
                    iters_per_epoch=0,
                    val_metric_names=val_metric_names,
                ),
                dde.callbacks.VariableValue(
                    ext_vars, period=lbfgs_display,
                    precision=8,
                ),
            ],
        )

    elapsed = time.time() - t0
    print(f"\n[DONE] elapsed={elapsed:.1f}s")
    print(f"[TB] tensorboard --logdir {tb_root}  （本次 run: {tb_dir}）")

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
            "temp_only": bool(cfg.get("temp_only", False)),
            "acc_scale": cfg["acc_scale"],
            "gyro_scale": cfg["gyro_scale"],
            "temp_scale": cfg["temp_scale"],
            "val_bad_threshold": float(cfg["val_bad_threshold"]),
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
            "gpu_ids": cfg["gpu_ids"],
            "split_mode": cfg.get("split_mode"),
            "mixed_precision": bool(cfg.get("mixed_precision")),
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
    export_onnx(unwrap_parallel_net(model.net), onnx_path)


if __name__ == "__main__":
    main()
