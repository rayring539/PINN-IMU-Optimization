"""早期自包含稀疏核试验脚本（与 core 并行存在，便于对照）。"""
import json
import os
import sys
from dataclasses import dataclass

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np


@dataclass
class KernelModel:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    Z_scaled: np.ndarray
    sigma: float
    W: np.ndarray


def rbf_kernel_phi(X_scaled: np.ndarray, Z_scaled: np.ndarray, sigma: float) -> np.ndarray:
    X2 = np.sum(X_scaled * X_scaled, axis=1, keepdims=True)
    Z2 = np.sum(Z_scaled * Z_scaled, axis=1, keepdims=True).T
    d2 = X2 + Z2 - 2.0 * (X_scaled @ Z_scaled.T)
    d2 = np.maximum(d2, 0.0)
    return np.exp(-d2 / (2.0 * sigma * sigma))


def compute_delta6(raw6: np.ndarray) -> np.ndarray:
    ideal = np.array([0.0, 0.0, 2048.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return ideal[None, :] - raw6


def read_first_n_lines(path: str, n_lines: int) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= n_lines:
                break
            line = line.strip()
            if not line:
                continue
            arr = np.fromstring(line, sep="\t")
            if arr.size != 7:
                arr = np.fromstring(line, sep=" ")
            if arr.size != 7:
                continue
            raw6 = arr[:6].astype(np.float64)
            T = float(arr[6])
            xs.append(np.concatenate([raw6, np.array([T], dtype=np.float64)]))
            ys.append(compute_delta6(raw6[None, :])[0])

    X_raw = np.stack(xs, axis=0).astype(np.float64)
    y = np.stack(ys, axis=0).astype(np.float64)
    return X_raw, y


def estimate_sigma_from_inducing(X_scaled: np.ndarray, Z_scaled: np.ndarray, rng: np.random.Generator) -> float:
    n = X_scaled.shape[0]
    k = min(2000, n)
    idx = rng.choice(n, size=k, replace=False) if n > k else np.arange(n)
    X_sub = X_scaled[idx]

    X2 = np.sum(X_sub * X_sub, axis=1, keepdims=True)
    Z2 = np.sum(Z_scaled * Z_scaled, axis=1, keepdims=True).T
    d2 = X2 + Z2 - 2.0 * (X_sub @ Z_scaled.T)
    d2 = np.maximum(d2, 0.0)
    median_d2 = float(np.median(d2))
    sigma = float(np.sqrt(median_d2 / 2.0 + 1e-12))
    sigma = max(sigma, 1e-3)
    return sigma


def fit_sparse_kernel_rbf(
    X_raw: np.ndarray,
    y: np.ndarray,
    M: int = 12,
    lam: float = 1e-3,
    sigma: float | None = None,
    seed: int = 0,
    train_n: int | None = None,
) -> tuple[KernelModel, dict]:
    rng = np.random.default_rng(seed)
    N = X_raw.shape[0]
    if train_n is None:
        train_n = N
    train_n = min(train_n, N)

    X_train_raw = X_raw[:train_n]
    y_train = y[:train_n]

    x_mean = X_train_raw.mean(axis=0)
    x_std = X_train_raw.std(axis=0) + 1e-12
    X_train = (X_train_raw - x_mean) / x_std

    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0) + 1e-12
    y_train_scaled = (y_train - y_mean) / y_std

    M = min(M, train_n)
    Z_idx = rng.choice(train_n, size=M, replace=False) if train_n > M else np.arange(M)
    Z_scaled = X_train[Z_idx]

    if sigma is None:
        sigma = estimate_sigma_from_inducing(X_train, Z_scaled, rng)

    A = np.zeros((M, M), dtype=np.float64)
    B = np.zeros((M, y_train_scaled.shape[1]), dtype=np.float64)

    chunk = 50000
    for start in range(0, train_n, chunk):
        end = min(start + chunk, train_n)
        Xc = X_train[start:end]
        yc = y_train_scaled[start:end]
        Phi = rbf_kernel_phi(Xc, Z_scaled, sigma=sigma)
        A += Phi.T @ Phi
        B += Phi.T @ yc

    A_reg = A + lam * np.eye(M, dtype=np.float64)
    W = np.linalg.solve(A_reg, B)

    model = KernelModel(
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        Z_scaled=Z_scaled,
        sigma=float(sigma),
        W=W,
    )
    meta = {"train_n": int(train_n), "M": int(M), "lam": float(lam), "sigma": float(sigma)}
    return model, meta


def predict_sparse_kernel(model: KernelModel, X_raw: np.ndarray, chunk: int = 50000) -> np.ndarray:
    X_scaled = (X_raw - model.x_mean[None, :]) / model.x_std[None, :]
    N = X_scaled.shape[0]
    out_scaled = np.zeros((N, 6), dtype=np.float64)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        Phi = rbf_kernel_phi(X_scaled[start:end], model.Z_scaled, sigma=model.sigma)
        out_scaled[start:end] = Phi @ model.W
    out = out_scaled * model.y_std[None, :] + model.y_mean[None, :]
    return out


def mae_rmse(a: np.ndarray, b: np.ndarray) -> dict:
    err = a - b
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err * err, axis=0))
    return {"mae_per_dim": mae.tolist(), "rmse_per_dim": rmse.tolist()}


def main():
    data_path = os.path.join(_REPO, "data", "1.txt")
    out_dir = os.path.join(_REPO, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    N_used = 300000
    test_ratio = 0.2

    M = 12
    lam = 1e-3
    seed = 0

    print(f"[INFO] read first {N_used} lines from {data_path}")
    X_raw, y = read_first_n_lines(data_path, n_lines=N_used)
    assert X_raw.shape[1] == 7 and y.shape[1] == 6

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

    print(f"[INFO] split: train={len(X_train_raw)} test={len(X_test)} used={len(X_raw)}")

    model, meta = fit_sparse_kernel_rbf(
        X_raw=X_train_raw,
        y=y_train,
        M=M,
        lam=lam,
        sigma=None,
        seed=seed,
        train_n=len(X_train_raw),
    )
    meta = dict(meta)
    meta.update({"test_ratio": test_ratio, "split_type": "temp_bin_stratified_round_int"})
    print(f"[INFO] meta: {meta}")

    y_pred = predict_sparse_kernel(model, X_test)
    metrics = mae_rmse(y_test, y_pred)
    print("[INFO] metrics:", json.dumps(metrics, ensure_ascii=False, indent=2))

    export = {
        "input_dim": 7,
        "output_dim": 6,
        "M": meta["M"],
        "sigma": meta["sigma"],
        "lam": meta["lam"],
        "x_mean": model.x_mean.tolist(),
        "x_std": model.x_std.tolist(),
        "y_mean": model.y_mean.tolist(),
        "y_std": model.y_std.tolist(),
        "Z_scaled": model.Z_scaled.tolist(),
        "W": model.W.tolist(),
        "note": "Predict y=delta6 (ideal6-raw6). MCU: x_scaled=(x-x_mean)/x_std; phi=RBF(x_scaled,Z_scaled); y_scaled=phi@W; y=y_scaled*y_std+y_mean; raw6_corrected=raw6+y.",
    }
    export_path = os.path.join(out_dir, "scheme_c_sparse_kernel_model_stratified_tempbin.json")
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False)
    print(f"[INFO] exported: {export_path}")


if __name__ == "__main__":
    main()
