"""稀疏 RBF 核岭回归拟合（诱导点）。"""
import time

import numpy as np

from .sparse_kernel_model import SparseKernelRbfModel, rbf_kernel_phi


def estimate_sigma_from_inducing(X_scaled: np.ndarray, Z_scaled: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    n = X_scaled.shape[0]
    m = Z_scaled.shape[0]
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
    show_eta: bool = True,
):
    rng = np.random.default_rng(seed)
    N = X_raw.shape[0]

    x_mean = X_raw.mean(axis=0)
    x_std = X_raw.std(axis=0) + 1e-12
    X_train = (X_raw - x_mean) / x_std

    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0) + 1e-12
    y_scaled = (y - y_mean) / y_std

    train_n = N
    M = min(M, train_n)
    Z_idx = rng.choice(train_n, size=M, replace=False) if train_n > M else np.arange(M)
    Z_scaled = X_train[Z_idx]

    if sigma is None:
        sigma = estimate_sigma_from_inducing(X_train, Z_scaled, seed)

    A = np.zeros((M, M), dtype=np.float64)
    B = np.zeros((M, 6), dtype=np.float64)

    chunk = 50000
    t0 = time.time()
    starts = list(range(0, train_n, chunk))
    n_chunks = len(starts)
    for ci, start in enumerate(starts):
        end = min(start + chunk, train_n)
        Xc = X_train[start:end]
        yc = y_scaled[start:end]
        Phi = rbf_kernel_phi(Xc, Z_scaled, sigma=float(sigma))
        A += Phi.T @ Phi
        B += Phi.T @ yc

        if show_eta and n_chunks > 1:
            elapsed = time.time() - t0
            progress = (ci + 1) / n_chunks
            eta = (elapsed / progress) - elapsed if progress > 0 else float("inf")
            print(f"[ETA] chunk {ci+1}/{n_chunks} elapsed={elapsed:.1f}s eta={eta:.1f}s")

    A_reg = A + lam * np.eye(M, dtype=np.float64)
    W = np.linalg.solve(A_reg, B)

    model = SparseKernelRbfModel(
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        Z_scaled=Z_scaled,
        sigma=float(sigma),
        W=W,
    )
    meta = {"M": int(M), "lam": float(lam), "sigma": float(sigma)}
    return model, meta
