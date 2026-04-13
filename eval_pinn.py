"""
PINN 评测入口 v3 —— 与 eval_blend.py 输出格式对齐，支持 dT/dt。
同时输出 LSB 和物理单位 (g / °/s) 指标。

用法:
  python eval_pinn.py --model_path outputs_pinn/pinn_model_best.pt
  python eval_pinn.py --model_path outputs_pinn/dde_ckpt-3150.pt
        # dde 中间权重需同目录 train_config.json，或 --train_config 指定
  python eval_pinn.py --model_path outputs_pinn/pinn_model_best.pt --test_data_path data/1.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

from core.data_pipeline import compute_dTdt, parse_data_file
from core.imu_data_io import test_path_from_split_meta
from core.pinn_metrics import summarize_delta_metrics
from core.pinn_dde import load_eval_checkpoint
from core.pinn_model import ACC_SCALE, GYRO_SCALE, TEMP_SCALE, PINN_IMU


def mae_rmse(a: np.ndarray, b: np.ndarray):
    err = b - a
    return np.mean(np.abs(err), axis=0).tolist(), np.sqrt(np.mean(err ** 2, axis=0)).tolist()


def coverage(ct: np.ndarray, cp: np.ndarray, eps: float):
    return np.mean(np.abs(cp - ct) <= eps, axis=0).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path",
                    default=os.path.join("outputs_pinn", "pinn_model_best.pt"))
    ap.add_argument("--split_meta",
                    default=os.path.join("outputs_pinn", "train_test_split.json"))
    ap.add_argument("--test_data_path", default=None)
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--N_used", type=int, default=-1)
    ap.add_argument("--eps", type=float, default=5.0,
                    help="coverage 阈值：各轴 |error|≤eps 的样本比例（与 bad 互补视角）")
    ap.add_argument("--bad_x", type=float, default=None,
                    help="bad 阈值 (LSB)：任一分量 |error|>bad_x 计为坏样本；默认与 --eps 相同")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--train_config",
        default=None,
        help="与 dde_ckpt-*.pt 配套；默认 <model_path 所在目录>/train_config.json",
    )
    args = ap.parse_args()

    # 确定测试文件
    if args.test_data_path:
        test_path = args.test_data_path
        split_note = "explicit_test_data_path"
    elif args.data_path:
        test_path = args.data_path
        split_note = "legacy_single_file"
    else:
        with open(args.split_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        test_path, test_meta_note = test_path_from_split_meta(meta)
        split_note = f"split_meta test={test_meta_note}"

    if not os.path.isfile(args.model_path):
        print(
            f"[ERROR] 找不到权重文件: {os.path.abspath(args.model_path)}\n"
            "        请使用 pinn_model_best.pt，或 DeepXDE 的 dde_ckpt-*.pt（需同目录 train_config.json），\n"
            "        并检查 --model_path / 训练配置里的 output.out_dir 是否一致。",
            file=sys.stderr,
        )
        sys.exit(1)

    # 先加载模型，再读测试数据（避免大文件读完才发现 checkpoint 不存在）
    model, ckpt_meta = load_eval_checkpoint(
        args.model_path, args.train_config, args.device
    )

    n_lines = None if args.N_used < 0 else args.N_used
    X_test, y_test = parse_data_file(test_path, n_lines=n_lines)
    Tdot_test = compute_dTdt(X_test[:, 6])
    use_dTdt = ckpt_meta.get("hyperparams", {}).get("use_dTdt", False)

    delta_pred = model.predict_delta6(
        X_test,
        Tdot_np=Tdot_test if use_dTdt else None,
    )

    hp = ckpt_meta.get("hyperparams", {})
    asc = hp.get("acc_scale", ACC_SCALE)
    gsc = hp.get("gyro_scale", GYRO_SCALE)

    raw6 = X_test[:, :6]
    corr_true = raw6 + y_test
    corr_pred = raw6 + delta_pred

    mae_vals, rmse_vals = mae_rmse(corr_true, corr_pred)
    cov_vals = coverage(corr_true, corr_pred, args.eps)

    bad_x = float(args.bad_x) if args.bad_x is not None else float(args.eps)
    # δ 空间：6 轴各自 RMSE/bad + 总 RMSE + 总 bad（与 train_pinn_dde metrics 一致）
    delta_lsb = summarize_delta_metrics(y_test, delta_pred, bad_x)
    # 物理单位指标 (g for acc, °/s for gyro)
    err_lsb = corr_pred - corr_true
    err_phys = np.copy(err_lsb)
    err_phys[:, :3] /= asc
    err_phys[:, 3:] /= gsc
    mae_phys = np.mean(np.abs(err_phys), axis=0).tolist()
    rmse_phys = np.sqrt(np.mean(err_phys ** 2, axis=0)).tolist()

    # 与 bad_x(LSB) 等价的物理阈值：|e_lsb|>bad_x ⇔ |e_phys|>bad_x/scale（逐轴）
    thr_phys = np.array(
        [bad_x / float(asc)] * 3 + [bad_x / float(gsc)] * 3, dtype=np.float64
    )
    ap = np.abs(err_phys)
    bad_phys_per_dim = np.mean(ap > thr_phys, axis=0).tolist()
    cov_phys_per_dim = np.mean(ap <= thr_phys, axis=0).tolist()
    bad_phys_total = float(np.mean(np.any(ap > thr_phys, axis=1)))
    bad_phys_element = float(np.mean(ap > thr_phys))

    # δ 在物理单位下的误差与「校正后读数」物理误差相同（δ_err == corr_err），单独列出便于对照 YAML 里 sensor 缩放
    delta_err_phys_mae = mae_phys
    delta_err_phys_rmse = rmse_phys
    eps_phys_hint = {
        "per_axis_if_uniform_eps_lsb": {
            "acc_axes_g": float(args.eps) / float(asc),
            "gyro_axes_dps": float(args.eps) / float(gsc),
        },
        "note": "将同一 eps(LSB) 换算为加计/陀螺各自物理量纲；bad_x 同理除以对应 scale。",
    }

    # 分支贡献分析
    model.eval()
    dev = args.device
    with torch.no_grad():
        xt = torch.from_numpy(X_test.astype(np.float32)).to(dev)
        td = torch.from_numpy(Tdot_test.astype(np.float32)).to(dev) if use_dTdt else None
        if isinstance(model, PINN_IMU):
            _, phys, hyst, res, _ = model(xt, Tdot=td, return_parts=True)
        else:
            if use_dTdt and td is not None:
                x_in = torch.cat([xt, td], dim=1)
            else:
                x_in = xt
            _, phys, hyst, res, _ = model(x_in, return_parts=True)
        phys_mag = float(phys.abs().mean())
        hyst_mag = float(hyst.abs().mean())
        res_mag  = float(res.abs().mean())

    dim_labels = ["ax(g)", "ay(g)", "az(g)", "gx(°/s)", "gy(°/s)", "gz(°/s)"]

    result = {
        "model_path": args.model_path,
        "split_type": split_note,
        "test_data_path": test_path,
        "N_used": int(args.N_used),
        "N_test": int(len(X_test)),
        "eps": float(args.eps),
        "bad_x": bad_x,
        "sensor_scales": {"acc": asc, "gyro": gsc},
        "delta_lsb": delta_lsb,
        "metrics_corrected_lsb": {
            "mae_per_dim": mae_vals,
            "rmse_per_dim": rmse_vals,
        },
        "metrics_corrected_physical": {
            "labels": dim_labels,
            "mae_per_dim": mae_phys,
            "rmse_per_dim": rmse_phys,
        },
        "delta_physical": {
            "labels": dim_labels,
            "mae_per_dim": delta_err_phys_mae,
            "rmse_per_dim": delta_err_phys_rmse,
            "bad_at_threshold_of_bad_x_lsb": {
                "thresholds_per_axis": thr_phys.tolist(),
                "threshold_units": dim_labels,
                "bad_per_dim": bad_phys_per_dim,
                "bad_total": bad_phys_total,
                "bad_element_rate": bad_phys_element,
                "coverage_per_dim": cov_phys_per_dim,
                "note": (
                    "与 delta_lsb 中 bad/coverage 数值等价，仅把阈值写成 g 与 °/s；"
                    "例如 bad_x=5 LSB 对应加计 |e|<5/2048 g、陀螺 |e|<5/16 °/s。"
                ),
            },
            "note": "δ_pred−δ_true 经 acc_scale/gyro_scale 换算；与 metrics_corrected_physical 数值相同。",
        },
        "eps_physical_hint": eps_phys_hint,
        "coverage_per_dim": cov_vals,
        "branch_mean_abs": {
            "physics": phys_mag,
            "hysteresis": hyst_mag,
            "residual": res_mag,
        },
        "note": (
            "PINN v3: acc/gyro/temp 标度见 sensor_scales；"
            "delta_lsb 中 bad/coverage 在 LSB；"
            "delta_physical.bad_at_threshold_of_bad_x_lsb 为同一判据在物理阈值下的复述（比例应与 delta_lsb 一致）。"
        ),
    }

    print("[INFO] eval result:\n" + json.dumps(result, ensure_ascii=False, indent=2))

    safe = os.path.splitext(os.path.basename(test_path))[0]
    model_stem = os.path.splitext(os.path.basename(args.model_path))[0]
    out_path = os.path.join(
        os.path.dirname(args.model_path) or "outputs_pinn",
        f"pinn_eval_{model_stem}_bad{bad_x}_eps{args.eps}_test_{safe}.json",
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[INFO] saved: {out_path}")


if __name__ == "__main__":
    main()
