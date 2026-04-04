import json
from dataclasses import dataclass

import numpy as np


@dataclass
class SparseKernelRbfModel:
    # Feature scaling for 7D input: x_scaled = (x_raw - x_mean) / x_std
    x_mean: np.ndarray  # (7,)
    x_std: np.ndarray   # (7,)

    # Target scaling for 6D output: y_scaled = (y - y_mean) / y_std
    y_mean: np.ndarray  # (6,)
    y_std: np.ndarray   # (6,)

    # Inducing points in scaled feature space: Z_scaled (M, 7)
    Z_scaled: np.ndarray  # (M, 7)

    # RBF width in scaled space
    sigma: float

    # Basis weights: W (M, 6) such that y_scaled_pred = phi(x,Z) @ W
    W: np.ndarray  # (M, 6)

    def predict_delta6(self, X_raw: np.ndarray, chunk: int = 50000) -> np.ndarray:
        """
        X_raw: (N,7) where last dim is temperature, first 6 dims are raw6.
        Returns:
          delta6_pred: (N,6) in original LSB units.
        """
        X_scaled = (X_raw - self.x_mean[None, :]) / self.x_std[None, :]
        N = X_scaled.shape[0]
        out_scaled = np.zeros((N, 6), dtype=np.float64)

        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            Phi = rbf_kernel_phi(X_scaled[start:end], self.Z_scaled, sigma=self.sigma)  # (Nc,M)
            out_scaled[start:end] = Phi @ self.W  # (Nc,6)

        out = out_scaled * self.y_std[None, :] + self.y_mean[None, :]
        return out


def rbf_kernel_phi(X_scaled: np.ndarray, Z_scaled: np.ndarray, sigma: float) -> np.ndarray:
    """
    X_scaled: (N, D)
    Z_scaled: (M, D)
    returns phi: (N, M), where phi[n,m] = exp(-||x-z||^2 / (2*sigma^2))
    """
    X2 = np.sum(X_scaled * X_scaled, axis=1, keepdims=True)  # (N,1)
    Z2 = np.sum(Z_scaled * Z_scaled, axis=1, keepdims=True).T  # (1,M)
    d2 = X2 + Z2 - 2.0 * (X_scaled @ Z_scaled.T)  # (N,M)
    d2 = np.maximum(d2, 0.0)
    return np.exp(-d2 / (2.0 * sigma * sigma))


def load_model_json(model_path: str) -> SparseKernelRbfModel:
    with open(model_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    return SparseKernelRbfModel(
        x_mean=np.array(obj["x_mean"], dtype=np.float64),
        x_std=np.array(obj["x_std"], dtype=np.float64),
        y_mean=np.array(obj["y_mean"], dtype=np.float64),
        y_std=np.array(obj["y_std"], dtype=np.float64),
        Z_scaled=np.array(obj["Z_scaled"], dtype=np.float64),
        sigma=float(obj["sigma"]),
        W=np.array(obj["W"], dtype=np.float64),
    )


def export_model_json(model: SparseKernelRbfModel, meta: dict, export_path: str) -> None:
    payload = {
        "input_dim": 7,
        "output_dim": 6,
        "sigma": float(model.sigma),
        "x_mean": model.x_mean.tolist(),
        "x_std": model.x_std.tolist(),
        "y_mean": model.y_mean.tolist(),
        "y_std": model.y_std.tolist(),
        "Z_scaled": model.Z_scaled.tolist(),
        "W": model.W.tolist(),
        **meta,
    }
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
