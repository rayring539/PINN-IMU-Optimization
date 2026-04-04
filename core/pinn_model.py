"""
PINN (Physics-Informed Neural Network) —— IMU 6 轴温度漂移修正 v3。

数据列定义:
  col 0-2  加速度计 (LSB)   物理量 = LSB / 2048      单位 g
  col 3-5  陀螺仪   (LSB)   物理量 = LSB / 16        单位 °/s
  col 6    温度     (LSB)   物理量 = LSB / 256       单位 ℃

物理约束源自 IMU_Temperature_Drift_PINN_Framework.md：
  Layer 1  热传导 → 温度场平滑
  Layer 2  热弹性 → 杨氏模量 / CTE / 封装应力
  Layer 3-A 加速度计 → TDB / TDSF / 三因子模型 / 迟滞
  Layer 3-B 陀螺仪   → 阻尼各向异性 / 品质因子

网络架构:
  physics_branch    : ΔT, Ṫ → 三因子参数化漂移 (acc 3-dim in g, gyro 3-dim in °/s → 拼合为 LSB)
  thermal_branch    : T_norm → MLP → 热特征
  hysteresis_branch : GRU 隐状态 → 迟滞修正 (可选)
  residual_branch   : [raw6_norm, thermal_feat] → MLP → 残差修正 (6-dim LSB)
  output            : delta6 = physics + hysteresis + residual  (LSB)
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
#  传感器标度因子
# ---------------------------------------------------------------------------
ACC_SCALE  = 2048.0   # 加速度: 物理值(g) = 寄存器值 / 2048
GYRO_SCALE = 16.0     # 角速度: 物理值(°/s) = 寄存器值 / 16
TEMP_SCALE = 256.0    # 温度:   物理值(℃)  = 寄存器值 / 256

IDEAL6 = np.array([0.0, 0.0, 2048.0, 0.0, 0.0, 0.0], dtype=np.float64)


# ---------------------------------------------------------------------------
#  基础构件
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 n_layers: int = 3, act: str = "tanh"):
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        act_cls = {"tanh": nn.Tanh, "gelu": nn.GELU, "relu": nn.ReLU}[act]
        for _ in range(n_layers):
            layers += [nn.Linear(dim, hidden), act_cls()]
            dim = hidden
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
#  PINN 主模型
# ---------------------------------------------------------------------------
class PINN_IMU(nn.Module):
    """Physics-Informed Neural Network for 6-axis IMU temperature drift (v3)."""

    def __init__(
        self,
        hidden_dim: int = 64,
        n_hidden: int = 4,
        thermal_dim: int = 16,
        act: str = "tanh",
        use_dTdt: bool = True,
        use_hysteresis: bool = False,
        hysteresis_hidden: int = 16,
        acc_scale: float = ACC_SCALE,
        gyro_scale: float = GYRO_SCALE,
        temp_scale: float = TEMP_SCALE,
        x_mean: np.ndarray | None = None,
        x_std: np.ndarray | None = None,
        y_mean: np.ndarray | None = None,
        y_std: np.ndarray | None = None,
    ):
        super().__init__()
        self.use_dTdt = use_dTdt
        self.use_hysteresis = use_hysteresis

        # ---- 标度因子 ----
        self.register_buffer("acc_scale",  torch.tensor(acc_scale,  dtype=torch.float32))
        self.register_buffer("gyro_scale", torch.tensor(gyro_scale, dtype=torch.float32))
        self.register_buffer("temp_scale", torch.tensor(temp_scale, dtype=torch.float32))

        # ---- 归一化参数 ----
        _z7 = np.zeros(7, dtype=np.float32)
        _o7 = np.ones(7, dtype=np.float32)
        _z6 = np.zeros(6, dtype=np.float32)
        _o6 = np.ones(6, dtype=np.float32)
        self.register_buffer("x_mean", torch.from_numpy(
            x_mean.astype(np.float32) if x_mean is not None else _z7))
        self.register_buffer("x_std", torch.from_numpy(
            x_std.astype(np.float32) if x_std is not None else _o7))
        self.register_buffer("y_mean", torch.from_numpy(
            y_mean.astype(np.float32) if y_mean is not None else _z6))
        self.register_buffer("y_std", torch.from_numpy(
            y_std.astype(np.float32) if y_std is not None else _o6))

        # ==================================================================
        #  Physics Branch — 三因子参数化 (Section 4.5.3)
        #  加速度计 (g) / 陀螺仪 (°/s) 分开参数化，输出时 × 标度 → LSB
        #    B = c0 + c1·ΔT + c2·ΔT² + c3·ΔT³
        #      + d1·Ṫ_℃ + d2·Ṫ_℃² + e1·ΔT·Ṫ_℃
        # ==================================================================
        self.T_ref = nn.Parameter(torch.tensor(25.0))  # ℃

        # 加速度计 (物理单位 g)
        self.acc_c0 = nn.Parameter(torch.zeros(3))
        self.acc_c1 = nn.Parameter(torch.zeros(3))
        self.acc_c2 = nn.Parameter(torch.zeros(3))
        self.acc_c3 = nn.Parameter(torch.zeros(3))
        self.acc_d1 = nn.Parameter(torch.zeros(3))
        self.acc_d2 = nn.Parameter(torch.zeros(3))
        self.acc_e1 = nn.Parameter(torch.zeros(3))

        # 陀螺仪 (物理单位 °/s)
        self.gyro_c0 = nn.Parameter(torch.zeros(3))
        self.gyro_c1 = nn.Parameter(torch.zeros(3))
        self.gyro_c2 = nn.Parameter(torch.zeros(3))
        self.gyro_c3 = nn.Parameter(torch.zeros(3))
        self.gyro_d1 = nn.Parameter(torch.zeros(3))
        self.gyro_d2 = nn.Parameter(torch.zeros(3))
        self.gyro_e1 = nn.Parameter(torch.zeros(3))

        # ---- 可学习物理参数 (Section 7.3) ----
        self.log_k_E = nn.Parameter(torch.tensor(math.log(60e-6)))
        self.log_alpha = nn.Parameter(torch.tensor(math.log(2.6e-6)))
        self.log_delta_alpha = nn.Parameter(torch.tensor(math.log(5e-6)))

        # ---- Thermal Branch ----
        self.thermal_net = MLP(1, hidden_dim // 2, thermal_dim,
                               n_layers=2, act=act)

        # ---- Hysteresis Branch (可选) ----
        if use_hysteresis:
            self.hyst_gru = nn.GRUCell(2, hysteresis_hidden)
            self.hyst_out = nn.Linear(hysteresis_hidden, 6)
            self.hysteresis_hidden_dim = hysteresis_hidden
        else:
            self.hysteresis_hidden_dim = 0

        # ---- Residual Branch ----
        self.residual_net = MLP(6 + thermal_dim, hidden_dim, 6,
                                n_layers=n_hidden, act=act)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def raw_T_to_celsius(self, T_raw: torch.Tensor) -> torch.Tensor:
        """T_℃ = T_raw / 256"""
        return T_raw / self.temp_scale

    def _three_factor(self, c0, c1, c2, c3, d1, d2, e1,
                      dT, Tdot_c):
        """三因子多项式 (静态 + 动态 + 交叉)，返回物理单位值。"""
        out = c0 + c1 * dT + c2 * dT ** 2 + c3 * dT ** 3
        if self.use_dTdt:
            out = out + d1 * Tdot_c + d2 * Tdot_c ** 2 + e1 * dT * Tdot_c
        return out

    # ---- forward ----
    def forward(
        self,
        x: torch.Tensor,
        Tdot: torch.Tensor | None = None,
        h_state: torch.Tensor | None = None,
        *,
        return_parts: bool = False,
    ):
        """
        x     : (N, 7)  [acc_raw(3), gyro_raw(3), T_raw(1)]  全部寄存器值
        Tdot  : (N, 1)  dT_raw/dt (可选)
        """
        raw6  = x[:, :6]
        T_raw = x[:, 6:7]
        N = x.shape[0]

        # ---- 温度 (℃) ----
        T_c = self.raw_T_to_celsius(T_raw)                       # (N,1)
        dT  = T_c - self.T_ref                                    # (N,1)

        # Ṫ 转换为 ℃/sample
        Tdot_c = (Tdot / self.temp_scale) if Tdot is not None else torch.zeros_like(T_raw)

        # ---- 物理分支 ----
        acc_phys_g   = self._three_factor(
            self.acc_c0, self.acc_c1, self.acc_c2, self.acc_c3,
            self.acc_d1, self.acc_d2, self.acc_e1, dT, Tdot_c)    # (N,3) in g

        gyro_phys_dps = self._three_factor(
            self.gyro_c0, self.gyro_c1, self.gyro_c2, self.gyro_c3,
            self.gyro_d1, self.gyro_d2, self.gyro_e1, dT, Tdot_c) # (N,3) in °/s

        # → LSB
        physics = torch.cat([
            acc_phys_g * self.acc_scale,
            gyro_phys_dps * self.gyro_scale,
        ], dim=1)                                                  # (N,6)

        # ---- 迟滞分支 ----
        hysteresis = torch.zeros(N, 6, device=x.device, dtype=x.dtype)
        h_out = h_state
        if self.use_hysteresis:
            gru_in = torch.cat([dT, Tdot_c], dim=1)
            if h_state is None:
                h_state = torch.zeros(N, self.hysteresis_hidden_dim,
                                      device=x.device, dtype=x.dtype)
            h_out = self.hyst_gru(gru_in, h_state)
            hysteresis = self.hyst_out(h_out)

        # ---- 热特征分支 ----
        T_norm = (T_raw - self.x_mean[6:7]) / self.x_std[6:7]
        thermal_feat = self.thermal_net(T_norm)

        # ---- 残差分支 ----
        raw6_norm = (raw6 - self.x_mean[:6]) / self.x_std[:6]
        res_input = torch.cat([raw6_norm, thermal_feat], dim=1)
        residual  = self.residual_net(res_input) * self.y_std

        # ---- 合成 (LSB) ----
        delta6 = physics + hysteresis + residual

        if return_parts:
            return delta6, physics, hysteresis, residual, h_out
        return delta6

    # ---- numpy 推理 ----
    @torch.no_grad()
    def predict_delta6(self, X_raw_np: np.ndarray,
                       Tdot_np: np.ndarray | None = None,
                       chunk: int = 50000) -> np.ndarray:
        dev = next(self.parameters()).device
        N = X_raw_np.shape[0]
        out = np.zeros((N, 6), dtype=np.float64)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            xt = torch.from_numpy(X_raw_np[s:e].astype(np.float32)).to(dev)
            td = None
            if Tdot_np is not None:
                td = torch.from_numpy(Tdot_np[s:e].astype(np.float32)).to(dev)
            out[s:e] = self.forward(xt, Tdot=td).cpu().numpy().astype(np.float64)
        return out


# ---------------------------------------------------------------------------
#  物理损失 v3 — 分传感器、物理量纲一致
# ---------------------------------------------------------------------------
class PINNPhysicsLoss:
    """
    8 类物理约束损失。在各传感器物理单位空间中计算。

    加速度计 (col 0-2): 物理单位 g = LSB / 2048
    陀螺仪   (col 3-5): 物理单位 °/s = LSB / 16
    温度               : 物理单位 ℃  = LSB / 256
    """

    @staticmethod
    def compute(
        model: PINN_IMU,
        x: torch.Tensor,
        delta_pred: torch.Tensor,
        T_raw_grad: torch.Tensor,
        physics: torch.Tensor,
        residual: torch.Tensor,
        Tdot: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        dev = delta_pred.device
        asc = model.acc_scale
        gsc = model.gyro_scale
        tsc = model.temp_scale

        # ---------- 转到物理单位 ----------
        acc_pred_g   = delta_pred[:, :3] / asc        # (N,3) g
        gyro_pred_dps = delta_pred[:, 3:] / gsc       # (N,3) °/s

        # ================= Layer 1: 热传导光滑 (物理单位) =================
        # 对 T_raw 求导, 再 ×temp_scale 转为 ∂/∂T_℃
        # ∂(phys_unit)/∂T_℃ = (∂(LSB)/∂T_raw) × temp_scale / sensor_scale
        acc_d1_g: list[torch.Tensor] = []
        for i in range(3):
            g1 = torch.autograd.grad(
                acc_pred_g[:, i].sum(), T_raw_grad,
                create_graph=True, retain_graph=True)[0]
            acc_d1_g.append(g1 * tsc)                  # → g / ℃
        acc_d1_cat = torch.cat(acc_d1_g, dim=1)

        gyro_d1_dps: list[torch.Tensor] = []
        for i in range(3):
            g1 = torch.autograd.grad(
                gyro_pred_dps[:, i].sum(), T_raw_grad,
                create_graph=True, retain_graph=True)[0]
            gyro_d1_dps.append(g1 * tsc)               # → (°/s) / ℃
        gyro_d1_cat = torch.cat(gyro_d1_dps, dim=1)

        # 二阶导
        acc_d2_g: list[torch.Tensor] = []
        for i in range(3):
            g2 = torch.autograd.grad(
                acc_d1_cat[:, i].sum(), T_raw_grad,
                create_graph=True, retain_graph=True)[0]
            acc_d2_g.append(g2 * tsc)                  # → g / ℃²
        acc_d2_cat = torch.cat(acc_d2_g, dim=1)

        gyro_d2_dps: list[torch.Tensor] = []
        for i in range(3):
            g2 = torch.autograd.grad(
                gyro_d1_cat[:, i].sum(), T_raw_grad,
                create_graph=True, retain_graph=True)[0]
            gyro_d2_dps.append(g2 * tsc)
        gyro_d2_cat = torch.cat(gyro_d2_dps, dim=1)

        losses["L_heat_smooth"] = (acc_d2_cat ** 2).mean() + (gyro_d2_cat ** 2).mean()

        # ================= Layer 2: 物理参数先验 =================
        k_E = torch.exp(model.log_k_E)
        alpha = torch.exp(model.log_alpha)
        delta_alpha = torch.exp(model.log_delta_alpha)

        losses["L_physics_prior"] = (
            ((k_E - 60e-6) / 60e-6) ** 2
            + ((alpha - 2.6e-6) / 2.6e-6) ** 2
            + ((delta_alpha - 5e-6) / 5e-6) ** 2
        )

        # ================= Layer 2: 刚度单调下降 =================
        # 加速度计: K↓ → SF↑ → 偏置正向漂移 → ∂(acc_g)/∂T_℃ ≥ 0
        losses["L_stiffness_mono"] = F.relu(-acc_d1_cat).mean()

        # ================= Layer 3-A: TDB 约束 (g 单位) =================
        T_c = model.raw_T_to_celsius(T_raw_grad)
        dT = T_c - model.T_ref                              # (N,1) ℃
        # 物理期望: TDB ∝ δα · ΔT (量纲: 无量纲 → 需乘几何因子, 此处做相对一致性)
        tdb_phys = delta_alpha * dT                          # (N,1)
        # 物理分支中加速度计线性项 (g 单位)
        acc_c1_dT = model.acc_c1.unsqueeze(0) * dT           # (N,3) g
        scale_norm = acc_c1_dT.detach().abs().mean().clamp(min=1e-10)
        losses["L_acc_tdb"] = (
            (acc_c1_dT - tdb_phys.expand_as(acc_c1_dT)) / scale_norm
        ).pow(2).mean()

        # ================= Layer 3-A: 三因子约束 =================
        if model.use_dTdt:
            static_acc  = model.acc_c1.abs()  + model.acc_c2.abs()  + 1e-8
            dynamic_acc = model.acc_d1.abs()  + model.acc_d2.abs()
            static_gyro  = model.gyro_c1.abs() + model.gyro_c2.abs() + 1e-8
            dynamic_gyro = model.gyro_d1.abs() + model.gyro_d2.abs()
            losses["L_three_factor"] = (
                F.relu(dynamic_acc - static_acc).mean()
                + F.relu(dynamic_gyro - static_gyro).mean()
            )
        else:
            losses["L_three_factor"] = torch.tensor(0.0, device=dev)

        # ================= Layer 3-B: 陀螺光滑 (°/s 单位) =================
        losses["L_gyro_smooth"] = (gyro_d2_cat ** 2).mean()

        # ================= 通用: 残差 ∥·∥² (物理单位) =================
        res_acc_g   = residual[:, :3] / asc
        res_gyro_dps = residual[:, 3:] / gsc
        losses["L_residual_small"] = (res_acc_g ** 2).mean() + (res_gyro_dps ** 2).mean()

        # ================= 通用: 残差梯度光滑 =================
        res_d1: list[torch.Tensor] = []
        for i in range(6):
            sc = asc if i < 3 else gsc
            g1 = torch.autograd.grad(
                (residual[:, i] / sc).sum(), T_raw_grad,
                create_graph=True, retain_graph=True)[0]
            res_d1.append(g1 * tsc)
        res_d1_cat = torch.cat(res_d1, dim=1)
        losses["L_grad_smooth"] = (res_d1_cat ** 2).mean()

        return losses


# ---------------------------------------------------------------------------
#  保存 / 加载
# ---------------------------------------------------------------------------
def save_pinn_checkpoint(model: PINN_IMU, meta: dict[str, Any], path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    raw = model.module if hasattr(model, "module") else model
    torch.save({"model_state": raw.state_dict(), "meta": meta}, path)


def load_pinn_checkpoint(
    path: str, device: str = "cpu"
) -> tuple[PINN_IMU, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    hp = meta.get("hyperparams", {})

    model = PINN_IMU(
        hidden_dim=hp.get("hidden_dim", 64),
        n_hidden=hp.get("n_hidden", 4),
        thermal_dim=hp.get("thermal_dim", 16),
        act=hp.get("act", "tanh"),
        use_dTdt=hp.get("use_dTdt", True),
        use_hysteresis=hp.get("use_hysteresis", False),
        hysteresis_hidden=hp.get("hysteresis_hidden", 16),
        acc_scale=hp.get("acc_scale", ACC_SCALE),
        gyro_scale=hp.get("gyro_scale", GYRO_SCALE),
        temp_scale=hp.get("temp_scale", TEMP_SCALE),
        x_mean=np.array(hp["x_mean"], dtype=np.float32) if "x_mean" in hp else None,
        x_std=np.array(hp["x_std"], dtype=np.float32)   if "x_std"  in hp else None,
        y_mean=np.array(hp["y_mean"], dtype=np.float32) if "y_mean" in hp else None,
        y_std=np.array(hp["y_std"], dtype=np.float32)   if "y_std"  in hp else None,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, meta
