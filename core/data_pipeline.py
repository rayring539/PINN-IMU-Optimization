"""7 列数据读取与按温度 bin 的分层划分。"""
import os
import time

import numpy as np

IDEAL6 = np.array([0.0, 0.0, 2048.0, 0.0, 0.0, 0.0], dtype=np.float64)


def compute_dTdt(T: np.ndarray) -> np.ndarray:
    """温度列 T_raw（与 ``parse_data_file`` 第 7 列一致）的差分，形状 (N,1)。"""
    dT = np.zeros_like(T)
    dT[1:-1] = (T[2:] - T[:-2]) / 2.0
    dT[0] = T[1] - T[0] if len(T) > 1 else 0.0
    dT[-1] = T[-1] - T[-2] if len(T) > 1 else 0.0
    return dT.reshape(-1, 1)


def parse_data_file(data_path: str, n_lines: int | None = None):
    """
    Read 7-col numeric file (no header).
    X (N,7) = [ax,ay,az,gx,gy,gz,T], y (N,6) = delta6 = ideal6 - raw6.

    Uses np.loadtxt (C-based) for speed; falls back to line-by-line
    if the file contains malformed rows.
    """
    basename = os.path.basename(data_path)
    size_mb = os.path.getsize(data_path) / (1024 * 1024)
    max_rows = n_lines if n_lines is not None and n_lines >= 0 else None

    print(f"  loading {basename} ({size_mb:.0f} MB) ...", end="", flush=True)
    t0 = time.time()

    try:
        data = np.loadtxt(data_path, max_rows=max_rows, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 7:
            raise ValueError(f"need 7+ cols, got {data.shape[1]}")
        X_raw = data[:, :7].copy()
    except (ValueError, IndexError):
        X_raw = _parse_data_file_slow(data_path, max_rows)

    y = IDEAL6 - X_raw[:, :6]
    elapsed = time.time() - t0
    print(f" {len(X_raw):,d} rows  ({elapsed:.1f}s)")
    return X_raw, y


def _parse_data_file_slow(data_path: str, max_rows: int | None) -> np.ndarray:
    """Line-by-line fallback for files with inconsistent formatting."""
    rows = []
    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            arr = np.fromstring(line, sep="\t")
            if arr.size < 7:
                arr = np.fromstring(line, sep=" ")
            if arr.size < 7:
                continue
            rows.append(arr[:7])
    return np.array(rows, dtype=np.float64)


def stratified_split_by_temp_bin_round_int(X_raw: np.ndarray, y: np.ndarray, test_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    T = X_raw[:, 6]
    T_bin = np.rint(T).astype(np.int64)

    test_mask = np.zeros(X_raw.shape[0], dtype=bool)
    for tb in np.unique(T_bin):
        idx = np.where(T_bin == tb)[0]
        if idx.size == 0:
            continue
        n_test = int(np.floor(test_ratio * idx.size))
        if idx.size >= 5:
            n_test = max(n_test, 1)
        else:
            n_test = min(max(n_test, 1), idx.size - 1) if idx.size > 1 else 0
        if n_test <= 0:
            continue
        chosen = rng.choice(idx, size=n_test, replace=False)
        test_mask[chosen] = True

    train_mask = ~test_mask
    X_train_raw, y_train = X_raw[train_mask], y[train_mask]
    X_test, y_test = X_raw[test_mask], y[test_mask]
    return X_train_raw, y_train, X_test, y_test


def split_train_val_random_rows(
    X_raw: np.ndarray, y: np.ndarray, val_ratio: float, seed: int
):
    """从同一段数据中随机划分验证集（行级打乱）。"""
    rng = np.random.default_rng(seed)
    n = X_raw.shape[0]
    n_val = max(1, int(round(n * float(val_ratio))))
    if n_val >= n:
        n_val = max(1, n - 1)
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    return X_raw[tr_idx], y[tr_idx], X_raw[val_idx], y[val_idx]


def split_train_val_temporal_tail(X_raw: np.ndarray, y: np.ndarray, val_ratio: float):
    """按拼接后的行序取末尾 ``val_ratio`` 为验证集（假设时间单调；多文件按 train_files 顺序拼接）。"""
    n = X_raw.shape[0]
    n_val = max(1, int(round(n * float(val_ratio))))
    if n_val >= n:
        n_val = max(1, n // 5)
    return X_raw[:-n_val], y[:-n_val], X_raw[-n_val:], y[-n_val:]
