"""
从已保存的 PINN checkpoint (.pt) 导出 ONNX 模型。

用法:
  python tools/export_onnx.py --ckpt D:/IMU_output/pinn_model_best.pt
  python tools/export_onnx.py --ckpt D:/IMU_output/pinn_model_best.pt --out model.onnx --opset 13
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_DDE = os.path.join(_REPO, "deepxde")
if _DDE not in sys.path:
    sys.path.insert(0, _DDE)
os.environ.setdefault("DDE_BACKEND", "pytorch")

from core.pinn_dde import load_dde_checkpoint, export_onnx  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="PINN checkpoint → ONNX")
    ap.add_argument("--ckpt", required=True, help="checkpoint .pt 文件路径")
    ap.add_argument("--out", default=None,
                    help="输出 .onnx 路径 (默认: 与 ckpt 同目录)")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset 版本")
    args = ap.parse_args()

    net, ext_vars, meta = load_dde_checkpoint(args.ckpt, cfg={})
    hp = meta.get("hyperparams", {})

    n_params = sum(p.numel() for p in net.parameters())
    print(f"[CKPT] {args.ckpt}")
    print(f"       params={n_params:,d}  ({n_params * 4 / 1024:.1f} KB float32)")
    print(f"       use_dTdt={hp.get('use_dTdt', True)}  "
          f"use_hysteresis={hp.get('use_hysteresis', False)}")

    if args.out is None:
        out_dir = os.path.dirname(args.ckpt) or "."
        args.out = os.path.join(out_dir, "pinn_model.onnx")

    export_onnx(net, args.out, opset=args.opset)


if __name__ == "__main__":
    main()
