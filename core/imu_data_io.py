"""
多文件数据目录：支持随机划分或显式指定训练/测试文件列表。
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any

import numpy as np

# 无 --config / 未在 YAML 中写 data_dir 时的默认目录（可按本机修改）
DEFAULT_DATA_DIR = "/home/leiyilin/IMU_data"

# 默认显式划分：1–4 训练，5 测试（与常见扫温实验一致）
DEFAULT_TRAIN_FILES = ["1.txt", "2.txt", "3.txt", "4.txt"]
DEFAULT_TEST_FILES = ["5.txt"]


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


def resolve_txt_path(data_dir: str, basename: str) -> str:
    """将相对文件名解析为绝对路径并校验存在。"""
    basename = str(basename).strip()
    if not basename:
        raise ValueError("文件名为空")
    if os.path.isabs(basename):
        p = basename
    else:
        p = os.path.join(os.path.abspath(data_dir), basename)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"数据文件不存在: {p}")
    return p


def split_train_test_explicit(
    data_dir: str,
    train_basenames: list[str],
    test_basenames: list[str],
) -> tuple[list[str], list[str], dict[str, Any]]:
    """按显式文件列表划分训练集 / 测试集（测试可为多个文件）。"""
    if not train_basenames:
        raise ValueError("train_files 不能为空")
    if not test_basenames:
        raise ValueError("test_files 不能为空")
    train_paths = [resolve_txt_path(data_dir, b) for b in train_basenames]
    test_paths = [resolve_txt_path(data_dir, b) for b in test_basenames]
    norm = lambda s: os.path.normcase(os.path.abspath(s))
    if set(map(norm, train_paths)) & set(map(norm, test_paths)):
        raise ValueError("训练集与测试集不能包含相同文件")
    meta: dict[str, Any] = {
        "split_mode": "explicit",
        "data_dir": os.path.abspath(data_dir),
        "train_files": [os.path.abspath(p) for p in train_paths],
        "test_files": [os.path.abspath(p) for p in test_paths],
        "train_basenames": list(train_basenames),
        "test_basenames": list(test_basenames),
    }
    return train_paths, test_paths, meta


def parse_file_list_arg(s: str | None) -> list[str] | None:
    """解析 CLI 逗号分隔列表，如 '1.txt,2.txt,3.txt'。"""
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def split_train_test_by_random_file(data_dir: str, seed: int) -> tuple[list[str], str, dict[str, Any]]:
    paths = list_sorted_txt_in_dir(data_dir)
    if len(paths) < 2:
        raise RuntimeError(f"至少需要 2 个 .txt 才能划分训练/测试，当前: {len(paths)}")

    rng = np.random.default_rng(seed)
    test_idx = int(rng.integers(0, len(paths)))
    test_path = paths[test_idx]
    train_paths = [p for i, p in enumerate(paths) if i != test_idx]

    meta = {
        "split_mode": "random",
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


def test_path_from_split_meta(meta: dict[str, Any]) -> tuple[str, str]:
    """从 ``train_test_split.json`` 解析测试文件路径与展示用说明（与 eval 脚本一致）。"""
    if "test_file" in meta:
        return meta["test_file"], str(meta.get("test_basename", ""))
    tfs = meta.get("test_files") or []
    if not tfs:
        raise KeyError(
            "train_test_split.json 中缺少 test_file 或 test_files，请检查训练是否写出划分文件"
        )
    test_path = tfs[0]
    tbn = meta.get("test_basenames") or []
    note_suffix = tbn[0] if tbn else os.path.basename(test_path)
    if len(tfs) > 1:
        note_suffix += (
            f" (test_files 共 {len(tfs)} 个，本脚本默认只评测第一个；可用 --test_data_path 指定)"
        )
    return test_path, note_suffix


def save_split_meta(meta: dict[str, Any], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_xy_train_test_from_dir(cfg: dict):
    """从 ``data_dir`` 读取并拼接训练/测试集。

    ``cfg`` 需包含: ``data_dir``, ``out_dir``, ``split_mode``, ``train_files``,
    ``test_files`` (explicit 时), ``seed`` (random 时), ``N_used``。

    Returns
    -------
    X_train, y_train, X_test, y_test, split_meta
    """
    from core.data_pipeline import parse_data_file

    data_dir = cfg["data_dir"]
    os.makedirs(cfg["out_dir"], exist_ok=True)
    split_json = os.path.join(cfg["out_dir"], "train_test_split.json")

    if cfg.get("split_mode") == "explicit":
        train_paths, test_paths, split_meta = split_train_test_explicit(
            data_dir, cfg["train_files"], cfg["test_files"])
        save_split_meta(split_meta, split_json)
        print(
            f"[SPLIT] explicit  train={split_meta['train_basenames']}  "
            f"test={split_meta['test_basenames']}")
    else:
        train_paths, test_path, split_meta = split_train_test_by_random_file(
            data_dir, seed=cfg["seed"])
        test_paths = [test_path]
        save_split_meta(split_meta, split_json)
        print(
            f"[SPLIT] random  test={split_meta['test_basename']}  "
            f"train={split_meta['train_basenames']}")

    xs, ys, total = [], [], 0
    n_limit = cfg["N_used"]
    for fp in train_paths:
        remain = None if n_limit < 0 else max(0, n_limit - total)
        if remain is not None and remain <= 0:
            break
        xf, yf = parse_data_file(fp, n_lines=remain)
        xs.append(xf)
        ys.append(yf)
        total += len(xf)
    X_train, y_train = np.concatenate(xs), np.concatenate(ys)

    test_n = None if cfg["N_used"] < 0 else cfg["N_used"]
    tx, ty = [], []
    for fp in test_paths:
        X, y = parse_data_file(fp, n_lines=test_n)
        tx.append(X)
        ty.append(y)
    X_test, y_test = np.concatenate(tx), np.concatenate(ty)
    return X_train, y_train, X_test, y_test, split_meta
