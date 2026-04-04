"""
多文件数据目录：列出目录下 1.txt, 2.txt, ... 等，随机选一个作测试集，其余作训练集。
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any

import numpy as np

DEFAULT_DATA_DIR = r"D:\IMU_data"


def list_sorted_txt_in_dir(data_dir: str) -> list[str]:
    data_dir = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    paths = glob.glob(os.path.join(data_dir, "*.txt"))
    if not paths:
        raise FileNotFoundError(f"目录下无 .txt 文件: {data_dir}")

    def sort_key(p: str) -> tuple[int, Any]:
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            return (0, int(stem))
        except ValueError:
            return (1, stem.lower())

    paths.sort(key=sort_key)
    return paths


def split_train_test_by_random_file(data_dir: str, seed: int) -> tuple[list[str], str, dict[str, Any]]:
    paths = list_sorted_txt_in_dir(data_dir)
    if len(paths) < 2:
        raise RuntimeError(f"至少需要 2 个 .txt 才能划分训练/测试，当前: {len(paths)}")

    rng = np.random.default_rng(seed)
    test_idx = int(rng.integers(0, len(paths)))
    test_path = paths[test_idx]
    train_paths = [p for i, p in enumerate(paths) if i != test_idx]

    meta = {
        "data_dir": os.path.abspath(data_dir),
        "test_file": os.path.abspath(test_path),
        "test_basename": os.path.basename(test_path),
        "test_index": test_idx,
        "train_files": [os.path.abspath(p) for p in train_paths],
        "train_basenames": [os.path.basename(p) for p in train_paths],
        "seed": int(seed),
        "n_files": len(paths),
    }
    return train_paths, test_path, meta


def save_split_meta(meta: dict[str, Any], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
