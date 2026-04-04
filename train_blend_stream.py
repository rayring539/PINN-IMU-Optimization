import argparse
import json
import os
import time

import numpy as np

from core.imu_data_io import DEFAULT_DATA_DIR, save_split_meta, split_train_test_by_random_file
from core.sparse_kernel_model import SparseKernelRbfModel, export_model_json, rbf_kernel_phi


IDEAL6 = np.array([0.0, 0.0, 2048.0, 0.0, 0.0, 0.0], dtype=np.float64)


def map_raw_to_c(T_raw: np.ndarray, raw_min: float, raw_max: float, t_min_c: float, t_max_c: float) -> np.ndarray:
    if abs(raw_max - raw_min) < 1e-12:
        return np.full_like(T_raw, (t_min_c + t_max_c) / 2.0, dtype=np.float64)
    scale = (t_max_c - t_min_c) / (raw_max - raw_min)
    return t_min_c + (T_raw - raw_min) * scale


def update_welford(mean: np.ndarray, m2: np.ndarray, count: int, x: np.ndarray):
    count_new = count + 1
    delta = x - mean
    mean = mean + delta / count_new
    delta2 = x - mean
    m2 = m2 + delta * delta2
    return mean, m2, count_new


def reservoir_add(reservoir: list[np.ndarray], x: np.ndarray, n_seen: int, k: int, rng: np.random.Generator):
    if len(reservoir) < k:
        reservoir.append(x.copy())
        return
    j = int(rng.integers(0, n_seen + 1))
    if j < k:
        reservoir[j] = x.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        default=DEFAULT_DATA_DIR,
        help=f"多文件模式：目录下 1.txt,2.txt,... 随机一个作测试，其余训练。默认 {DEFAULT_DATA_DIR}",
    )
    ap.add_argument(
        "--data_path",
        default=None,
        help="单文件模式（兼容旧流程）：指定单个 txt 时忽略 data_dir，仍按温度 bin 内 k 抽样划分 train/test",
    )
    ap.add_argument("--N_limit", type=int, default=-1, help="-1 means all train lines; otherwise max train rows (多文件为训练集总行数上限)")
    ap.add_argument("--test_ratio", type=float, default=0.2, help="仅单文件模式有效：近似测试比例 -> k=round(1/ratio)")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--M", type=int, default=12, help="inducing points")
    ap.add_argument("--lam_kernel", type=float, default=1e-3)

    ap.add_argument("--poly_lam", type=float, default=1e-3)
    ap.add_argument("--T_start_c", type=float, default=55.0)
    ap.add_argument("--margin_c", type=float, default=3.0)

    ap.add_argument("--t_min_c", type=float, default=-35.0)
    ap.add_argument("--t_max_c", type=float, default=65.0)

    ap.add_argument("--clip_p_low", type=float, default=1.0)
    ap.add_argument("--clip_p_high", type=float, default=99.0)
    ap.add_argument("--quantile_reservoir", type=int, default=10000)

    ap.add_argument("--sigma_sample_size", type=int, default=2000)
    ap.add_argument("--chunk_train_rows", type=int, default=50000)
    ap.add_argument("--show_eta", type=int, default=1)

    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    multi_file = args.data_path is None or str(args.data_path).strip() == ""
    if multi_file:
        train_paths, test_path, split_meta = split_train_test_by_random_file(args.data_dir, args.seed)
        split_json = os.path.join(args.out_dir, "train_test_split.json")
        save_split_meta(split_meta, split_json)
        print(f"[SPLIT] test_file={split_meta['test_basename']} (随机 1 个), train={split_meta['train_basenames']}")
        print(f"[SPLIT] saved: {split_json}")
        k = None
    else:
        train_paths = [args.data_path]
        test_path = None
        split_meta = None
        k = max(1, int(round(1.0 / args.test_ratio)))

    rng = np.random.default_rng(args.seed)

    x_mean = np.zeros(7, dtype=np.float64)
    x_m2 = np.zeros(7, dtype=np.float64)
    x_count = 0

    y_mean = np.zeros(6, dtype=np.float64)
    y_m2 = np.zeros(6, dtype=np.float64)
    y_count = 0

    Z_reservoir = []
    X_sigma_reservoir = []
    Q_reservoir = []
    train_seen = 0

    bin_counts = {}
    raw_min = None
    raw_max = None

    t_start = time.time()

    line_idx = 0
    if multi_file:
        stop_train = False
        for fp in train_paths:
            if stop_train:
                break
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if args.N_limit >= 0 and train_seen >= args.N_limit:
                        stop_train = True
                        break
                    line_idx += 1
                    s = line.strip()
                    if not s:
                        continue
                    arr = np.fromstring(s, sep="\t")
                    if arr.size != 7:
                        arr = np.fromstring(s, sep=" ")
                    if arr.size != 7:
                        continue

                    raw6 = arr[:6].astype(np.float64)
                    T_raw = float(arr[6])

                    X_row = np.concatenate([raw6, np.array([T_raw], dtype=np.float64)])
                    y_row = IDEAL6 - raw6

                    x_mean, x_m2, x_count = update_welford(x_mean, x_m2, x_count, X_row)
                    y_mean, y_m2, y_count = update_welford(y_mean, y_m2, y_count, y_row)

                    if raw_min is None or T_raw < raw_min:
                        raw_min = T_raw
                    if raw_max is None or T_raw > raw_max:
                        raw_max = T_raw

                    reservoir_add(Z_reservoir, X_row, train_seen, args.M, rng)
                    reservoir_add(X_sigma_reservoir, X_row, train_seen, args.sigma_sample_size, rng)
                    reservoir_add(Q_reservoir, y_row, train_seen, args.quantile_reservoir, rng)

                    train_seen += 1
    else:
        with open(args.data_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if args.N_limit >= 0 and line_idx >= args.N_limit:
                    break
                line_idx += 1
                s = line.strip()
                if not s:
                    continue
                arr = np.fromstring(s, sep="\t")
                if arr.size != 7:
                    arr = np.fromstring(s, sep=" ")
                if arr.size != 7:
                    continue

                raw6 = arr[:6].astype(np.float64)
                T_raw = float(arr[6])
                tb = int(np.rint(T_raw))

                c = bin_counts.get(tb, 0)
                in_test = (c % k == 0)
                bin_counts[tb] = c + 1

                if in_test:
                    continue

                X_row = np.concatenate([raw6, np.array([T_raw], dtype=np.float64)])
                y_row = IDEAL6 - raw6

                x_mean, x_m2, x_count = update_welford(x_mean, x_m2, x_count, X_row)
                y_mean, y_m2, y_count = update_welford(y_mean, y_m2, y_count, y_row)

                if raw_min is None or T_raw < raw_min:
                    raw_min = T_raw
                if raw_max is None or T_raw > raw_max:
                    raw_max = T_raw

                reservoir_add(Z_reservoir, X_row, train_seen, args.M, rng)
                reservoir_add(X_sigma_reservoir, X_row, train_seen, args.sigma_sample_size, rng)
                reservoir_add(Q_reservoir, y_row, train_seen, args.quantile_reservoir, rng)

                train_seen += 1

    if train_seen < max(args.M, 10):
        raise RuntimeError(f"train_seen too small: {train_seen}. Check split rule / N_limit.")

    x_std = np.sqrt(x_m2 / max(1, x_count - 1)) + 1e-12
    y_std = np.sqrt(y_m2 / max(1, y_count - 1)) + 1e-12

    Z_raw = np.stack(Z_reservoir, axis=0).astype(np.float64)
    Z_scaled = (Z_raw - x_mean[None, :]) / x_std[None, :]
    X_sigma_raw = np.stack(X_sigma_reservoir, axis=0).astype(np.float64)
    X_sigma_scaled = (X_sigma_raw - x_mean[None, :]) / x_std[None, :]

    Ks = X_sigma_scaled.shape[0]
    M = Z_scaled.shape[0]
    X2 = np.sum(X_sigma_scaled * X_sigma_scaled, axis=1, keepdims=True)
    Z2 = np.sum(Z_scaled * Z_scaled, axis=1, keepdims=True).T
    d2 = X2 + Z2 - 2.0 * (X_sigma_scaled @ Z_scaled.T)
    d2 = np.maximum(d2, 0.0)
    sigma_med = float(np.median(d2))
    sigma = float(np.sqrt(sigma_med / 2.0 + 1e-12))
    sigma = max(sigma, 1e-3)

    Q = np.stack(Q_reservoir, axis=0) if len(Q_reservoir) > 0 else np.zeros((1, 6), dtype=np.float64)
    clip_low = np.percentile(Q, args.clip_p_low, axis=0).astype(np.float64)
    clip_high = np.percentile(Q, args.clip_p_high, axis=0).astype(np.float64)

    print(
        f"[PASS1 done] train_seen={train_seen} "
        f"x_mean[0]={x_mean[0]:.3f} T_raw_range=[{raw_min:.3f},{raw_max:.3f}] "
        f"sigma={sigma:.4f} elapsed={time.time()-t_start:.1f}s"
    )

    A = np.zeros((M, M), dtype=np.float64)
    B = np.zeros((M, 6), dtype=np.float64)

    S00 = S01 = S02 = 0.0
    S11 = S12 = S22 = 0.0
    R0 = np.zeros(6, dtype=np.float64)
    R1 = np.zeros(6, dtype=np.float64)
    R2 = np.zeros(6, dtype=np.float64)

    bin_counts = {}
    train_seen2 = 0
    processed_rows = 0
    t0 = time.time()

    X_chunk: list = []
    y_chunk: list = []
    T_chunk_raw: list = []

    def flush_chunk():
        nonlocal A, B
        nonlocal S00, S01, S02, S11, S12, S22, R0, R1, R2, train_seen2
        if not X_chunk:
            return

        Xc_raw = np.stack(X_chunk, axis=0).astype(np.float64)
        yc_raw = np.stack(y_chunk, axis=0).astype(np.float64)
        Tc_raw = np.array(T_chunk_raw, dtype=np.float64)

        Xc_scaled = (Xc_raw - x_mean[None, :]) / x_std[None, :]
        yc_scaled = (yc_raw - y_mean[None, :]) / y_std[None, :]
        Phi = rbf_kernel_phi(Xc_scaled, Z_scaled, sigma=sigma)
        A += Phi.T @ Phi
        B += Phi.T @ yc_scaled

        Tc = map_raw_to_c(Tc_raw, raw_min=float(raw_min), raw_max=float(raw_max), t_min_c=args.t_min_c, t_max_c=args.t_max_c)
        t2 = Tc * Tc
        t3 = t2 * Tc
        t4 = t2 * t2

        S00 += float(np.sum(t4))
        S01 += float(np.sum(t3))
        S02 += float(np.sum(t2))
        S11 += float(np.sum(t2))
        S12 += float(np.sum(Tc))
        S22 += float(len(Tc))

        R0 += (t2[:, None] * yc_raw).sum(axis=0)
        R1 += (Tc[:, None] * yc_raw).sum(axis=0)
        R2 += yc_raw.sum(axis=0)

        if args.show_eta and processed_rows > 0 and (train_seen2 % (args.chunk_train_rows * 2) == 0):
            elapsed = time.time() - t0
            rate = train_seen2 / max(1e-9, elapsed)
            print(f"[PASS2] train_seen={train_seen2} elapsed={elapsed:.1f}s rate={rate:.1f} rows/s")

        X_chunk.clear()
        y_chunk.clear()
        T_chunk_raw.clear()

    def process_line_pass2(s: str) -> None:
        nonlocal train_seen2, processed_rows
        s = s.strip()
        if not s:
            return
        processed_rows += 1

        arr = np.fromstring(s, sep="\t")
        if arr.size != 7:
            arr = np.fromstring(s, sep=" ")
        if arr.size != 7:
            return

        raw6 = arr[:6].astype(np.float64)
        T_raw = float(arr[6])

        if multi_file:
            pass
        else:
            tb = int(np.rint(T_raw))
            c = bin_counts.get(tb, 0)
            in_test = (c % k == 0)
            bin_counts[tb] = c + 1
            if in_test:
                return

        X_row = np.concatenate([raw6, np.array([T_raw], dtype=np.float64)])
        y_row = IDEAL6 - raw6

        X_chunk.append(X_row)
        y_chunk.append(y_row)
        T_chunk_raw.append(T_raw)
        train_seen2 += 1

        if len(X_chunk) >= args.chunk_train_rows:
            flush_chunk()

    if multi_file:
        stop2 = False
        for fp in train_paths:
            if stop2:
                break
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if args.N_limit >= 0 and train_seen2 >= args.N_limit:
                        stop2 = True
                        break
                    process_line_pass2(line)
    else:
        with open(args.data_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if args.N_limit >= 0 and processed_rows >= args.N_limit:
                    break
                process_line_pass2(line)

    flush_chunk()

    print(f"[PASS2 done] train_seen2={train_seen2} elapsed={time.time()-t0:.1f}s")

    A_reg = A + args.lam_kernel * np.eye(M, dtype=np.float64)
    W = np.linalg.solve(A_reg, B)

    kernel_model = SparseKernelRbfModel(
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        Z_scaled=Z_scaled,
        sigma=float(sigma),
        W=W,
    )

    kernel_out = os.path.join(args.out_dir, f"scheme_c_kernel_model_blend_stream_M{args.M}.json")
    split_type_kernel = (
        "random_one_txt_test_others_train"
        if multi_file
        else f"temp_bin_round_int_every_k k={k}"
    )
    export_model_json(
        model=kernel_model,
        meta={
            "split_type": split_type_kernel,
            "test_ratio_approx": float(args.test_ratio) if not multi_file else None,
            "N_train": int(train_seen2),
            "N_limit": int(args.N_limit),
            "M": int(args.M),
            "lam_kernel": float(args.lam_kernel),
            "sigma": float(sigma),
            "seed": int(args.seed),
            "test_file": split_meta["test_basename"] if multi_file and split_meta else None,
        },
        export_path=kernel_out,
    )

    XtX = np.array(
        [
            [S00, S01, S02],
            [S01, S11, S12],
            [S02, S12, S22],
        ],
        dtype=np.float64,
    )
    XtX_reg = XtX + args.poly_lam * np.eye(3, dtype=np.float64)
    XtY = np.stack([R0, R1, R2], axis=0)
    beta = np.linalg.solve(XtX_reg, XtY)
    a2 = beta[0, :]
    a1 = beta[1, :]
    a0 = beta[2, :]

    blend_out = os.path.join(args.out_dir, "scheme_c_blend_model_stream.json")
    blend_payload = {
        "kernel_model_path": kernel_out,
        "poly6_a2": a2.tolist(),
        "poly6_a1": a1.tolist(),
        "poly6_a0": a0.tolist(),
        "raw_min": float(raw_min),
        "raw_max": float(raw_max),
        "t_min_c": float(args.t_min_c),
        "t_max_c": float(args.t_max_c),
        "T_start_c": float(args.T_start_c),
        "margin_c": float(args.margin_c),
        "clip_low": clip_low.tolist(),
        "clip_high": clip_high.tolist(),
        "meta": {
            "split_type": split_type_kernel,
            "test_ratio_approx": float(args.test_ratio) if not multi_file else None,
            "N_train": int(train_seen2),
            "N_limit": int(args.N_limit),
            "M": int(args.M),
            "lam_kernel": float(args.lam_kernel),
            "poly_lam": float(args.poly_lam),
            "seed": int(args.seed),
            "clip_p_low": float(args.clip_p_low),
            "clip_p_high": float(args.clip_p_high),
            "quantile_reservoir": int(args.quantile_reservoir),
            "sigma_sample_size": int(args.sigma_sample_size),
            "train_test_split": split_meta if multi_file else None,
        },
    }

    with open(blend_out, "w", encoding="utf-8") as f:
        json.dump(blend_payload, f, ensure_ascii=False)

    print(f"[DONE] kernel_model={kernel_out}")
    print(f"[DONE] blend_model={blend_out}")


if __name__ == "__main__":
    main()
