"""
DeepXDE-based PINN for IMU 6-axis temperature drift correction (v3-dde).

相比手动 PyTorch 训练循环的简化:
  - dde.nn.FNN   → 替代手写 MLP
  - dde.grad     → 替代手动 autograd 循环
  - dde.Model    → 替代手动 epoch/batch 循环
  - dde.Variable → 可学习物理参数 (反问题框架)
  - dde.callbacks → ModelCheckpoint / EarlyStopping / VariableValue
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Sequence

import numpy as np

# ---------- DeepXDE 路径 & 后端 ----------
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DDE_DIR = os.path.join(_REPO, "deepxde")
if _DDE_DIR not in sys.path:
    sys.path.insert(0, _DDE_DIR)
os.environ.setdefault("DDE_BACKEND", "pytorch")

import deepxde as dde  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# ---------------------------------------------------------------------------
ACC_SCALE = 2048.0
GYRO_SCALE = 16.0
TEMP_SCALE = 256.0
IDEAL6 = np.array([0.0, 0.0, 2048.0, 0.0, 0.0, 0.0], dtype=np.float64)


def unwrap_parallel_net(net: nn.Module) -> nn.Module:
    """``DataParallel`` 包装时返回内部 ``module``，否则原样返回。"""
    return net.module if hasattr(net, "module") else net


# ---------------------------------------------------------------------------
#  IMUNet — 多分支 PINN
# ---------------------------------------------------------------------------
class IMUNet(dde.nn.NN):
    """Multi-branch PINN for 6-axis IMU temperature drift.

    Input : (N, 7+) = [acc_raw(3), gyro_raw(3), T_raw(1), Tdot_raw(1, opt)]
    Output: (N, 6)  = delta6 (LSB)

    temp_only
      若 True，残差分支不读取加计/陀螺（raw6_norm 置零），输出仅由温度(及 Ṫ)经物理/热/残差决定；
      数据格式仍为 7/8 列，需单独训练 checkpoint。

    Branches
      physics  — three-factor polynomial (acc g / gyro °/s → ×scale → LSB)
      thermal  — dde.nn.FNN (T_norm → features)
      residual — dde.nn.FNN ([raw6_norm, thermal] → delta)
      hysteresis (opt) — GRU
    """

    def __init__(self, cfg: dict,
                 x_mean: np.ndarray, x_std: np.ndarray,
                 y_std: np.ndarray):
        super().__init__()
        hidden   = cfg.get("hidden_dim", 64)
        n_hidden = cfg.get("n_hidden", 4)
        th_dim   = cfg.get("thermal_dim", 16)
        act      = cfg.get("act", "tanh")
        self.use_dTdt       = cfg.get("use_dTdt", True)
        self.use_hysteresis = cfg.get("use_hysteresis", False)
        self.temp_only      = bool(cfg.get("temp_only", False))

        self.register_buffer("acc_scale",  torch.tensor(float(cfg.get("acc_scale", ACC_SCALE))))
        self.register_buffer("gyro_scale", torch.tensor(float(cfg.get("gyro_scale", GYRO_SCALE))))
        self.register_buffer("temp_scale", torch.tensor(float(cfg.get("temp_scale", TEMP_SCALE))))

        self.register_buffer("x_mean", torch.tensor(x_mean[:7], dtype=torch.float32))
        self.register_buffer("x_std",  torch.tensor(x_std[:7],  dtype=torch.float32))
        self.register_buffer("y_std",  torch.tensor(y_std,       dtype=torch.float32))

        # ---- Physics branch: acc (g) / gyro (°/s) ----
        self.T_ref = nn.Parameter(torch.tensor(25.0))
        for prefix, n in [("acc", 3), ("gyro", 3)]:
            for name in ["c0", "c1", "c2", "c3", "d1", "d2", "e1"]:
                self.register_parameter(
                    f"{prefix}_{name}", nn.Parameter(torch.zeros(n)))

        # ---- Thermal branch (dde.nn.FNN) ----
        self.thermal_net = dde.nn.FNN(
            [1, th_dim, th_dim, th_dim], act, "Glorot normal")

        # ---- Hysteresis branch (opt) ----
        if self.use_hysteresis:
            hh = cfg.get("hysteresis_hidden", 16)
            self.hyst_gru = nn.GRUCell(2, hh)
            self.hyst_out = nn.Linear(hh, 6)
            self._hyst_h = hh

        # ---- Residual branch (dde.nn.FNN) ----
        self.residual_net = dde.nn.FNN(
            [6 + th_dim] + [hidden] * n_hidden + [6], act, "Glorot normal")

    # ---- helpers ----
    def _poly3(self, prefix: str, dT, Tdot_c):
        c0 = getattr(self, f"{prefix}_c0")
        c1 = getattr(self, f"{prefix}_c1")
        c2 = getattr(self, f"{prefix}_c2")
        c3 = getattr(self, f"{prefix}_c3")
        out = c0 + c1 * dT + c2 * dT ** 2 + c3 * dT ** 3
        if self.use_dTdt:
            d1 = getattr(self, f"{prefix}_d1")
            d2 = getattr(self, f"{prefix}_d2")
            e1 = getattr(self, f"{prefix}_e1")
            out = out + d1 * Tdot_c + d2 * Tdot_c ** 2 + e1 * dT * Tdot_c
        return out

    def forward_physics_hysteresis(self, inputs):
        """物理分支 + 滞回分支（与 forward 前半段一致）。供损失里用 outputs 分解残差。"""
        raw6 = inputs[:, :6]
        T_raw = inputs[:, 6:7]
        N = inputs.shape[0]

        T_c = T_raw / self.temp_scale
        dT = T_c - self.T_ref

        if self.use_dTdt:
            Tdot_c = inputs[:, 7:8] / self.temp_scale
        else:
            Tdot_c = torch.zeros_like(T_raw)

        acc_g = self._poly3("acc", dT, Tdot_c)
        gyro_dps = self._poly3("gyro", dT, Tdot_c)
        physics = torch.cat(
            [acc_g * self.acc_scale, gyro_dps * self.gyro_scale], dim=1
        )

        hysteresis = torch.zeros(N, 6, device=inputs.device, dtype=inputs.dtype)
        if self.use_hysteresis:
            h0 = torch.zeros(
                N, self._hyst_h, device=inputs.device, dtype=inputs.dtype
            )
            h1 = self.hyst_gru(torch.cat([dT, Tdot_c], dim=1), h0)
            hysteresis = self.hyst_out(h1)
        return physics, hysteresis

    def forward(self, inputs, *, return_parts: bool = False):
        physics, hysteresis = self.forward_physics_hysteresis(inputs)

        raw6 = inputs[:, :6]
        T_raw = inputs[:, 6:7]
        T_norm = (T_raw - self.x_mean[6:7]) / self.x_std[6:7]
        thermal = self.thermal_net(T_norm)

        raw6_norm = (raw6 - self.x_mean[:6]) / self.x_std[:6]
        if self.temp_only:
            raw6_norm = torch.zeros_like(raw6_norm)
        residual = self.residual_net(
            torch.cat([raw6_norm, thermal], dim=1)) * self.y_std

        self._physics_part = physics
        self._residual_part = residual

        delta = physics + hysteresis + residual
        if return_parts:
            return delta, physics, hysteresis, residual, None
        return delta

    @torch.no_grad()
    def predict_delta6(
        self,
        X_np: np.ndarray,
        Tdot_np: np.ndarray | None = None,
        chunk: int = 50000,
    ) -> np.ndarray:
        """NumPy 推理。``use_dTdt`` 时需 7 列 X + ``Tdot_np`` 拼成 8 列与训练一致。"""
        dev = next(self.parameters()).device
        if self.use_dTdt:
            if Tdot_np is None:
                Tdot_np = np.zeros((X_np.shape[0], 1), dtype=np.float32)
            X_full = np.hstack(
                [X_np.astype(np.float32), Tdot_np.astype(np.float32)]
            )
        else:
            X_full = X_np
        N = X_full.shape[0]
        out = np.zeros((N, 6), dtype=np.float64)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            xt = torch.from_numpy(X_full[s:e].astype(np.float32)).to(dev)
            out[s:e] = self.forward(xt).cpu().numpy().astype(np.float64)
        return out


# ---------------------------------------------------------------------------
#  IMUPINNData — DataSet + 8 项物理损失
# ---------------------------------------------------------------------------
class IMUPINNData(dde.data.DataSet):
    """DataSet that augments the data MSE with 8 physics-informed loss terms.

    losses() 返回 9 个标量张量:
      [L_data, L_heat_smooth, L_physics_prior, L_stiffness_mono,
       L_acc_tdb, L_three_factor, L_gyro_smooth, L_residual_small, L_grad_smooth]
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train=X_train, y_train=y_train,
                         X_test=X_test,   y_test=y_test)

    def train_next_batch(self, batch_size=None):
        if batch_size is None or batch_size >= len(self.train_x):
            return self.train_x, self.train_y
        idx = np.random.choice(len(self.train_x), batch_size, replace=False)
        return self.train_x[idx], self.train_y[idx]

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        net = unwrap_parallel_net(model.net)
        asc, gsc, tsc = net.acc_scale, net.gyro_scale, net.temp_scale
        dev = outputs.device

        # ========= 0) 数据损失 =========
        L_data = loss_fn(targets, outputs)

        # ========= 1) 热传导光滑: ∂²δ/∂T_℃² =========
        d2_sq = []
        for i in range(6):
            sc = asc if i < 3 else gsc
            d2 = dde.grad.hessian(outputs, inputs, component=i, i=6, j=6)
            d2_sq.append((d2 * tsc ** 2 / sc) ** 2)
        L_smooth = torch.mean(torch.cat(d2_sq, dim=1))

        # ========= 2) 物理参数先验 =========
        ev = model.external_trainable_variables
        if len(ev) >= 3:
            k_E, alpha, da = [torch.exp(v) for v in ev[:3]]
            L_prior = (((k_E - 60e-6) / 60e-6) ** 2
                       + ((alpha - 2.6e-6) / 2.6e-6) ** 2
                       + ((da - 5e-6) / 5e-6) ** 2)
        else:
            L_prior = torch.tensor(0.0, device=dev)

        # ========= 3) 加速度计刚度单调 =========
        mono = []
        for i in range(3):
            dy = dde.grad.jacobian(outputs, inputs, i=i, j=6)
            mono.append(F.relu(-dy * tsc / asc))
        L_mono = torch.mean(torch.cat(mono, dim=1))

        # ========= 4) TDB ∝ δα·ΔT =========
        T_c = inputs[:, 6:7] / tsc
        dT = T_c - net.T_ref
        da_val = torch.exp(ev[2]) if len(ev) >= 3 else torch.tensor(5e-6, device=dev)
        tdb_phys = da_val * dT
        c1_dT = net.acc_c1.unsqueeze(0) * dT
        sc_n = c1_dT.detach().abs().mean().clamp(min=1e-10)
        L_tdb = ((c1_dT - tdb_phys.expand_as(c1_dT)) / sc_n).pow(2).mean()

        # ========= 5) 三因子: 动态 << 静态 =========
        if net.use_dTdt:
            s_a = net.acc_c1.abs()  + net.acc_c2.abs()  + 1e-8
            d_a = net.acc_d1.abs()  + net.acc_d2.abs()
            s_g = net.gyro_c1.abs() + net.gyro_c2.abs() + 1e-8
            d_g = net.gyro_d1.abs() + net.gyro_d2.abs()
            L_3f = F.relu(d_a - s_a).mean() + F.relu(d_g - s_g).mean()
        else:
            L_3f = torch.tensor(0.0, device=dev)

        # ========= 6) 陀螺光滑 =========
        gd2 = []
        for i in range(3, 6):
            d2 = dde.grad.hessian(outputs, inputs, component=i, i=6, j=6)
            gd2.append((d2 * tsc ** 2 / gsc) ** 2)
        L_gyro = torch.mean(torch.cat(gd2, dim=1))

        # ========= 7) 残差幅度 =========
        # DataParallel 时 forward 在子模块副本上执行，主 module 上无 _residual_part；用输出分解
        physics, hysteresis = net.forward_physics_hysteresis(inputs)
        res = outputs - physics - hysteresis
        L_res = ((res[:, :3] / asc) ** 2).mean() + ((res[:, 3:] / gsc) ** 2).mean()

        # ========= 8) 残差梯度光滑 =========
        rd = []
        for i in range(6):
            sc = asc if i < 3 else gsc
            g = torch.autograd.grad(
                (res[:, i] / sc).sum(), inputs,
                create_graph=True, retain_graph=True,
            )[0][:, 6:7]
            rd.append((g * tsc) ** 2)
        L_grad = torch.mean(torch.cat(rd, dim=1))

        return [L_data, L_smooth, L_prior, L_mono,
                L_tdb, L_3f, L_gyro, L_res, L_grad]

    def losses_test(self, targets, outputs, loss_fn, inputs, model, aux=None):
        """测试阶段只算数据损失 + 简单物理项 (跳过耗时的高阶梯度)。"""
        net = unwrap_parallel_net(model.net)
        asc, gsc = net.acc_scale, net.gyro_scale
        dev = outputs.device
        zero = torch.tensor(0.0, device=dev)

        L_data = loss_fn(targets, outputs)

        ev = model.external_trainable_variables
        if len(ev) >= 3:
            k_E, alpha, da = [torch.exp(v) for v in ev[:3]]
            L_prior = (((k_E - 60e-6) / 60e-6) ** 2
                       + ((alpha - 2.6e-6) / 2.6e-6) ** 2
                       + ((da - 5e-6) / 5e-6) ** 2)
        else:
            L_prior = zero

        physics, hysteresis = net.forward_physics_hysteresis(inputs)
        res = outputs - physics - hysteresis
        L_res = ((res[:, :3] / asc) ** 2).mean() + ((res[:, 3:] / gsc) ** 2).mean()

        return [L_data, zero, L_prior, zero,
                zero, zero, zero, L_res, zero]


# ---------------------------------------------------------------------------
#  PhysicsWarmup 回调
# ---------------------------------------------------------------------------
class PhysicsWarmup(dde.callbacks.Callback):
    """前 warmup_iters 步线性增大物理损失权重 (第 0 项 = 数据权重不变)。"""

    def __init__(self, base_weights: list[float], warmup_iters: int):
        super().__init__()
        self.base = list(base_weights)
        self.warmup = max(1, warmup_iters)

    def on_epoch_begin(self):
        step = self.model.train_state.step
        s = min(1.0, step / self.warmup)
        w = [self.base[0]] + [wi * s for wi in self.base[1:]]
        dev = next(self.model.net.parameters()).device
        self.model.loss_weights = torch.as_tensor(w, dtype=torch.float32, device=dev)


# ---------------------------------------------------------------------------
#  进度条 + ETA 回调
# ---------------------------------------------------------------------------
class ProgressBar(dde.callbacks.Callback):
    """tqdm 进度条，显示训练进度、当前 loss 和 ETA。"""

    def __init__(self, total_iters: int, desc: str = "Adam"):
        super().__init__()
        self._total = total_iters
        self._desc = desc
        self._pbar = None
        self._phase_done = 0

    def on_train_begin(self):
        from tqdm import tqdm
        self._start_step = self.model.train_state.step
        self._phase_done = 0
        self._pbar = tqdm(
            total=self._total, unit="it", desc=self._desc,
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                       "[{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )

    def on_epoch_end(self):
        current = self.model.train_state.step - self._start_step
        advance = current - self._phase_done
        if advance > 0:
            self._pbar.update(advance)
            self._phase_done = current

        ts = self.model.train_state
        if ts.loss_train is not None:
            data_l = ts.loss_train[0]
            total_l = sum(ts.loss_train)
            self._pbar.set_postfix_str(
                f"data={data_l:.3e} total={total_l:.3e}")

    def on_train_end(self):
        if self._pbar:
            self._pbar.close()


# ---------------------------------------------------------------------------
#  TensorBoard 回调
# ---------------------------------------------------------------------------
_LOSS_NAMES = [
    "data", "heat_smooth", "physics_prior", "stiffness_mono",
    "acc_tdb", "three_factor", "gyro_smooth", "residual_small", "grad_smooth",
]


class TensorBoardCallback(dde.callbacks.Callback):
    """参考 git 版 PINN-IMU-Optimization（Monsterl-ite 风格）的 TensorBoard：

    - **train/***  每个优化步写一次（从 ``model._step_losses`` 读取，由 ``Model._compile_pytorch`` 在
      ``train_step`` 的 closure 里写入）
    - **val/***    验证步：``losses_test`` 分项；若 ``model.compile(metrics=...)`` 则另有 6 轴
      ``rmse_*`` / ``bad_*``、``rmse_total``、``bad_total``（与 ``training.val_bad_threshold`` 一致）。

    横坐标 = ``train_state.step``（优化步）。需配合 ``deepxde`` 中已打补丁的 ``_step_losses``。
    """

    def __init__(
        self,
        log_dir: str,
        iters_per_epoch: int = 0,
        val_metric_names: Sequence[str] | None = None,
    ):
        super().__init__()
        self._log_dir = log_dir
        self._writer = None
        self._ipe = max(0, int(iters_per_epoch))
        self._lh_seen = 0
        self._val_metric_names = tuple(val_metric_names) if val_metric_names else ()

    def _write_train(self, step: int) -> None:
        losses = getattr(self.model, "_step_losses", None)
        if losses is None or self._writer is None:
            return
        for i, val in enumerate(losses):
            name = _LOSS_NAMES[i] if i < len(_LOSS_NAMES) else f"term_{i}"
            self._writer.add_scalar(f"train/{name}", float(val), step)
        self._writer.add_scalar("train/total", float(sum(losses)), step)
        if self._ipe > 0:
            self._writer.add_scalar("meta/epoch", float(step) / float(self._ipe), step)
        lr = None
        if hasattr(self.model, "opt") and hasattr(self.model.opt, "param_groups"):
            lr = self.model.opt.param_groups[0].get("lr")
        if lr is not None:
            self._writer.add_scalar("train/lr", float(lr), step)

    def _write_val(self, step: int) -> None:
        ts = self.model.train_state
        if ts.loss_test is None or self._writer is None:
            return
        for i, val in enumerate(ts.loss_test):
            name = _LOSS_NAMES[i] if i < len(_LOSS_NAMES) else f"term_{i}"
            self._writer.add_scalar(f"val/{name}", float(val), step)
        self._writer.add_scalar("val/total", float(sum(ts.loss_test)), step)

    def _write_val_metrics(self, step: int) -> None:
        if not self._val_metric_names or self._writer is None:
            return
        ts = self.model.train_state
        if not ts.metrics_test:
            return
        step = int(step)
        for name, val in zip(self._val_metric_names, ts.metrics_test):
            self._writer.add_scalar(f"val/{name}", float(val), step)

    def on_train_begin(self):
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(self._log_dir, exist_ok=True)
        self._writer = SummaryWriter(self._log_dir, flush_secs=5)
        self._lh_seen = 0
        n = len(self.model.losshistory.steps)
        if n > 0:
            self._lh_seen = n
            st = int(self.model.train_state.step)
            self._write_val(st)
            self._write_val_metrics(st)
            self._writer.flush()

    def on_epoch_end(self):
        if self._writer is None:
            return
        step = int(self.model.train_state.step)
        self._write_train(step)
        n = len(self.model.losshistory.steps)
        if n > self._lh_seen:
            self._lh_seen = n
            self._write_val(step)
            self._write_val_metrics(step)
        self._writer.flush()

    def on_train_end(self):
        if self._writer:
            self._writer.flush()
            self._writer.close()
            self._writer = None


# ---------------------------------------------------------------------------
#  保存 / 加载 (兼容 eval_pinn.py / plot 脚本)
# ---------------------------------------------------------------------------
def save_dde_checkpoint(model: dde.Model,
                        ext_vars: list,
                        meta: dict[str, Any],
                        path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    raw = unwrap_parallel_net(model.net)
    torch.save({
        "model_state": raw.state_dict(),
        "ext_vars": [float(v.data) for v in ext_vars],
        "meta": meta,
    }, path)


def load_dde_checkpoint(
    path: str, cfg: dict, device: str = "cpu",
) -> tuple[IMUNet, list[torch.nn.Parameter], dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    hp = meta.get("hyperparams", {})

    x_mean = np.array(hp["x_mean"], dtype=np.float32) if "x_mean" in hp else np.zeros(7, dtype=np.float32)
    x_std  = np.array(hp["x_std"],  dtype=np.float32) if "x_std"  in hp else np.ones(7,  dtype=np.float32)
    y_std  = np.array(hp["y_std"],  dtype=np.float32) if "y_std"  in hp else np.ones(6,  dtype=np.float32)

    merged = {**cfg, **hp}
    net = IMUNet(merged, x_mean, x_std, y_std)
    net.load_state_dict(ckpt["model_state"])
    net.to(device)
    net.eval()

    ext_vars = []
    for val in ckpt.get("ext_vars", []):
        ext_vars.append(torch.nn.Parameter(torch.tensor(val)))
    return net, ext_vars, meta


def load_imunet_from_dde_model_save(
    path: str,
    cfg: dict,
    device: str = "cpu",
) -> tuple[IMUNet, dict[str, Any]]:
    """从 DeepXDE ``Model.save()`` 的 ``dde_ckpt-*.pt`` 恢复 ``IMUNet``。

    文件通常仅含 ``model_state_dict`` / ``optimizer_state_dict``，无 ``meta``。
    ``x_mean`` / ``x_std`` / ``y_std`` 以权重里的 buffer 为准（与训练一致）。
    返回 ``(net, cfg_used)``：若 ``train_config`` 与权重中是否含迟滞分支不一致，``cfg_used`` 以权重为准。
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict")
    if sd is None:
        raise ValueError(
            f"期望 DeepXDE Model.save 格式（含 model_state_dict）: {path}"
        )
    new_sd = {}
    for k, v in sd.items():
        nk = k[7:] if k.startswith("module.") else k
        new_sd[nk] = v
    # 迟滞分支可选：仅以权重为准，避免 train_config 与保存时不一致（如后来改过 YAML）
    has_hyst = any(k.startswith("hyst_gru") or k.startswith("hyst_out") for k in new_sd)
    cfg_m = dict(cfg)
    want_hyst = bool(cfg_m.get("use_hysteresis", False))
    if want_hyst != has_hyst:
        cfg_m["use_hysteresis"] = has_hyst
        print(
            f"[WARN] train_config use_hysteresis={want_hyst} 与 checkpoint 不一致，"
            f"已改为 use_hysteresis={has_hyst}（以权重为准）。"
        )
    x_mean = np.zeros(7, dtype=np.float32)
    x_std = np.ones(7, dtype=np.float32)
    y_std = np.ones(6, dtype=np.float32)
    net = IMUNet(cfg_m, x_mean, x_std, y_std)
    net.load_state_dict(new_sd, strict=True)
    net.to(device)
    net.eval()
    return net, cfg_m


def load_eval_checkpoint(
    model_path: str,
    train_config_path: str | None,
    device: str = "cpu",
) -> tuple[Any, dict[str, Any]]:
    """从 ``pinn_model_best.pt`` 或 DeepXDE ``dde_ckpt-*.pt`` 加载模型与 meta。"""
    from core.pinn_model import load_pinn_checkpoint

    blob = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(blob, dict) and "meta" in blob and "model_state" in blob:
        return load_pinn_checkpoint(model_path, device)
    if isinstance(blob, dict) and "model_state_dict" in blob:
        tc = train_config_path
        if not tc:
            d = os.path.dirname(os.path.abspath(model_path))
            tc = os.path.join(d if d else ".", "train_config.json")
        if not os.path.isfile(tc):
            raise FileNotFoundError(
                f"DeepXDE 中间 checkpoint 需 train_config.json（或指定路径），未找到: {tc}"
            )
        with open(tc, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        model, cfg_used = load_imunet_from_dde_model_save(model_path, cfg, device)
        return model, {"hyperparams": cfg_used}
    raise ValueError(
        f"无法识别 checkpoint: {model_path}（需 pinn_model_best 或 dde_ckpt 的 model_state_dict）"
    )


# ---------------------------------------------------------------------------
#  ONNX 导出
# ---------------------------------------------------------------------------
def export_onnx(
    net: IMUNet,
    out_path: str,
    opset: int = 17,
    verify: bool = True,
) -> str:
    """将 IMUNet 导出为 ONNX 格式，用于后续量化 / 边缘部署。

    Parameters
    ----------
    net : IMUNet
        训练好的网络 (会自动切换到 eval 模式)。
    out_path : str
        输出 .onnx 文件路径。
    opset : int
        ONNX opset 版本 (默认 17，兼容主流量化工具)。
    verify : bool
        若为 True 且安装了 ``onnx`` 包，则用 onnx.checker 校验。

    Returns
    -------
    str
        实际保存路径。
    """
    net.eval()
    net.cpu()

    in_dim = 8 if net.use_dTdt else 7
    dummy = torch.randn(1, in_dim)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    torch.onnx.export(
        net,
        (dummy,),
        out_path,
        input_names=["imu_input"],
        output_names=["delta6"],
        dynamic_axes={
            "imu_input": {0: "batch"},
            "delta6":    {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    size_kb = os.path.getsize(out_path) / 1024
    print(f"[ONNX] 导出成功: {out_path}  ({size_kb:.1f} KB)")
    print(f"       输入: imu_input  shape=(batch, {in_dim})"
          f"{'  (temp_only: 残差不使用 raw6)' if getattr(net, 'temp_only', False) else ''}")
    print(f"       输出: delta6     shape=(batch, 6)")
    print(f"       opset={opset}")

    if verify:
        try:
            import onnx
            model_onnx = onnx.load(out_path)
            onnx.checker.check_model(model_onnx)
            print("       onnx.checker: PASS")
        except ImportError:
            print("       (跳过 onnx.checker: 未安装 onnx 包)")
        except Exception as e:
            print(f"       onnx.checker: FAIL - {e}")

    return out_path
