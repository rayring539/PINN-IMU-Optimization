"""
绘制 PINN 修正后的 6 轴 vs 温度曲线。
可同时叠加物理分支 / 残差分支各自的贡献以便分析。

用法:
  python tools/plot_pinn_results.py \
      --model_path outputs_pinn/pinn_model_best.pt \
      --split_meta outputs_pinn/train_test_split.json \
      --out_png outputs_pinn/pinn_corrected_6d.png
"""
import argparse
import json
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import matplotlib.pyplot as plt
import numpy as np
import torch

from core.pinn_model import ACC_SCALE, GYRO_SCALE, TEMP_SCALE, load_pinn_checkpoint


def load_data_7col(path: str, n_lines: int | None = None):
    xs = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if n_lines is not None and n_lines >= 0 and i >= n_lines:
                break
            line = line.strip()
            if not line:
                continue
            arr = np.fromstring(line, sep="\t")
            if arr.size != 7:
                arr = np.fromstring(line, sep=" ")
            if arr.size != 7:
                continue
            xs.append(arr.astype(np.float64))
    return np.stack(xs, axis=0).astype(np.float64)


def bin_mean(x: np.ndarray, y: np.ndarray):
    bins = np.unique(x)
    ym = np.array([float(np.mean(y[x == b])) if np.any(x == b) else np.nan for b in bins])
    return bins, ym


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path",
                    default=os.path.join(_REPO, "outputs_pinn", "pinn_model_best.pt"))
    ap.add_argument("--split_meta",
                    default=os.path.join(_REPO, "outputs_pinn", "train_test_split.json"))
    ap.add_argument("--test_data_path", default=None)
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--N_used", type=int, default=300000)
    ap.add_argument("--out_png",
                    default=os.path.join(_REPO, "outputs_pinn", "pinn_corrected_6d.png"))
    ap.add_argument("--show_components", type=int, default=1,
                    help="1: 同时画 raw / corrected / physics-only 三条线")
    args = ap.parse_args()

    # 确定绘图数据
    if args.test_data_path:
        plot_path = args.test_data_path
    elif args.data_path:
        plot_path = args.data_path
    else:
        with open(args.split_meta, "r", encoding="utf-8") as f:
            plot_path = json.load(f)["test_file"]

    n_lines = None if args.N_used < 0 else args.N_used
    X = load_data_7col(plot_path, n_lines=n_lines)
    raw6 = X[:, :6]
    T_raw = X[:, 6]
    ideal6 = np.array([0.0, 0.0, 2048.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # 加载 PINN
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_pinn_checkpoint(args.model_path, device=device)
    model.eval()

    # dT/dt
    use_dTdt = meta.get("hyperparams", {}).get("use_dTdt", False)
    Tdot_np = None
    if use_dTdt:
        dT = np.zeros_like(T_raw)
        dT[1:-1] = (T_raw[2:] - T_raw[:-2]) / 2.0
        if len(T_raw) > 1:
            dT[0] = T_raw[1] - T_raw[0]
            dT[-1] = T_raw[-1] - T_raw[-2]
        Tdot_np = dT.reshape(-1, 1)

    delta_pred = model.predict_delta6(X, Tdot_np=Tdot_np)
    corrected6 = raw6 + delta_pred

    # 物理 / 残差分量
    phys_np, res_np = None, None
    if args.show_components:
        with torch.no_grad():
            xt = torch.from_numpy(X.astype(np.float32)).to(device)
            td = torch.from_numpy(Tdot_np.astype(np.float32)).to(device) if Tdot_np is not None else None
            _, phys_t, _hyst_t, res_t, _ = model(xt, Tdot=td, return_parts=True)
            phys_np = phys_t.cpu().numpy().astype(np.float64)
            res_np = res_t.cpu().numpy().astype(np.float64)

    # 温度轴 → ℃
    hp = meta.get("hyperparams", {})
    tsc = hp.get("temp_scale", TEMP_SCALE)
    asc = hp.get("acc_scale", ACC_SCALE)
    gsc = hp.get("gyro_scale", GYRO_SCALE)

    T_c = T_raw / tsc
    T_bin = np.rint(T_c).astype(np.int64)

    names = ["ax (g)", "ay (g)", "az (g)", "gx (°/s)", "gy (°/s)", "gz (°/s)"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    axes = axes.flatten()

    # 转换为物理单位显示
    scales = [asc]*3 + [gsc]*3
    ideal_phys = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # g / °/s

    for dim in range(6):
        ax = axes[dim]
        sc = scales[dim]

        bx, by_raw = bin_mean(T_bin, raw6[:, dim] / sc)
        ax.plot(bx, by_raw, linewidth=0.8, alpha=0.5, label="raw", color="gray")

        _, by_corr = bin_mean(T_bin, corrected6[:, dim] / sc)
        ax.plot(bx, by_corr, marker="o", markersize=2, linewidth=1.2, label="corrected")

        if phys_np is not None:
            corr_phys = (raw6[:, dim] + phys_np[:, dim]) / sc
            _, by_phys = bin_mean(T_bin, corr_phys)
            ax.plot(bx, by_phys, linewidth=1.0, linestyle="--", alpha=0.7,
                    label="physics only", color="green")

        ax.axhline(ideal_phys[dim], linewidth=0.8, linestyle=":", color="red")
        ax.set_title(names[dim])
        unit = "g" if dim < 3 else "°/s"
        ax.set_ylabel(unit)
        ax.grid(True, alpha=0.25)
        if dim == 0:
            ax.legend(fontsize=7)

    axes[-2].set_xlabel("Temperature (℃)")
    axes[-1].set_xlabel("Temperature (℃)")
    fig.suptitle("PINN corrected 6D vs Temperature", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    fig.savefig(args.out_png, dpi=160)
    print(f"[INFO] saved figure: {args.out_png}")


if __name__ == "__main__":
    main()
