"""内存版 blend 训练：二次多项式拟合与 raw→摄氏度线性映射。"""
import numpy as np

from .blend_model import QuadraticPoly6


def map_raw_to_c(T_raw: np.ndarray, raw_min: float, raw_max: float, t_min_c: float, t_max_c: float) -> np.ndarray:
    if abs(raw_max - raw_min) < 1e-12:
        return np.full_like(T_raw, (t_min_c + t_max_c) / 2.0, dtype=np.float64)
    scale = (t_max_c - t_min_c) / (raw_max - raw_min)
    return t_min_c + (T_raw - raw_min) * scale


def fit_quadratic_poly6(T_c: np.ndarray, delta6: np.ndarray, lam: float = 1e-3) -> QuadraticPoly6:
    X = np.stack([T_c * T_c, T_c, np.ones_like(T_c)], axis=1).astype(np.float64)
    XtX = X.T @ X
    I = np.eye(3, dtype=np.float64)
    beta = np.linalg.solve(XtX + lam * I, X.T @ delta6)
    return QuadraticPoly6(a2=beta[0, :], a1=beta[1, :], a0=beta[2, :])
