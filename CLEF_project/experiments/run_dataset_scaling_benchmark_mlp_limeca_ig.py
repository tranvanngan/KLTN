"""run_dataset_scaling_benchmark_mlp_limeca_ig.py
=================================================
Benchmark CLEF runtime + (approx) memory scaling when the dataset size grows.

Requested configuration:
  - Methods: LIME-CA and Integrated Gradients
  - Model: MLP (MLP-AE)

Dataset scaling:
  - Create synthetic scaled dataset from the base CLEF dataset by:
      * tiling/resampling base samples to reach target N
      * adding small jitter noise in feature space (features are already
        min-max normalised to [0,1]) to avoid exact duplicates.

Targets:
  - 1x / 5x / 10x / 20x of the base dataset size
  - optionally also "--targets" to try 50k/100k/1M directly.

Outputs:
  results/tables/table_dataset_scaling_benchmark_mlp_limeca_ig.csv
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import gc
import tracemalloc

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data_prep
import models
import explainers as ex
import metrics


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
LOG_PATH = os.path.join(RESULTS_DIR, "logs", "run_log.txt")


def log(msg: str):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def scale_dataset_tile_jitter(X: np.ndarray, target_n: int, factor_label: str,
                               jitter_std: float = 0.01, seed: int = 42) -> np.ndarray:
    """Create X_big with length target_n from base X.

    - Resample base indices with replacement.
    - Add small Gaussian jitter to all 6 features.
    - Clip to [0,1].

    X is assumed already min-max normalised.
    """
    rng = np.random.default_rng(seed)
    n_base = X.shape[0]
    if target_n <= n_base:
        # For completeness; still allow downscaling.
        idx = rng.choice(np.arange(n_base), size=target_n, replace=False)
    else:
        idx = rng.choice(np.arange(n_base), size=target_n, replace=True)

    X_big = X[idx].copy()

    # Jitter: keep it small to preserve distribution while avoiding
    # identical samples which can lead to degenerate similarity filtering.
    noise = rng.normal(loc=0.0, scale=jitter_std, size=X_big.shape)
    X_big = np.clip(X_big + noise, 0.0, 1.0)

    # Avoid potential pathological all-zero/constant columns due to clipping.
    # (Shouldn't happen with jitter_std > 0.)
    return X_big.astype(np.float32)


def approximate_memory_peak_mb() -> float:
    # Called after tracemalloc snapshots; user-friendly wrapper.
    current, peak = tracemalloc.get_traced_memory()
    return float(peak) / (1024 * 1024)


def _mlp_checkpoint_path(device: str, target_n: int, seed: int, jitter_std: float) -> str:
    chk_dir = os.path.join(RESULTS_DIR, "checkpoints")
    os.makedirs(chk_dir, exist_ok=True)
    # Use jitter_std with safe string format to avoid filesystem issues.
    jitter_str = f"{jitter_std:.6f}".replace('.', 'p')
    fname = f"mlp_weights_{device}_n{target_n}_seed{seed}_j{jitter_str}.npz"
    return os.path.join(chk_dir, fname)


def run_one_config(device: str, X_base: np.ndarray, target_n: int, topk: int,
                    seed: int, jitter_std: float,
                    method_list: list[str],
                    checkpoint_every: int = 1) -> list[dict]:

    """Run MLP detector + LIME-CA / IG explanations on X_big.

    Returns rows to be written to CSV.
    """
    # Scale dataset

    factor_label = "custom"
    if target_n == X_base.shape[0]:
        factor_label = "1x"

    X = scale_dataset_tile_jitter(
        X_base, target_n=target_n, factor_label=factor_label,
        jitter_std=jitter_std, seed=seed,
    )

    # Fit MLP detector (with checkpoint caching to improve reproducibility)
    chk_path = _mlp_checkpoint_path(device=device, target_n=target_n, seed=seed, jitter_std=jitter_std)
    if os.path.exists(chk_path):
        # Load cached weights into a fresh MLP model instance
        m = models.MLPAEModel()
        data = np.load(chk_path, allow_pickle=True)
        m._weights = data["weights"].tolist()
        m._activations = data["activations"].tolist()
        # threshold depends on scores computed on this X (but weights deterministic now)
        m.fit_threshold(X)
    else:
        fitted = models.build_and_fit_selected(X, model_names=["MLP"])
        m = fitted["MLP"]
        # Save weights for later runs
        np.savez_compressed(
            chk_path,
            weights=np.array(m._weights, dtype=object),
            activations=np.array(m._activations, dtype=object),
        )


    scores = m.score(X)
    normal_idx = np.where(scores <= m.threshold)[0]
    top_idx = m.top_k_indices(X, topk)

    score_fn = m.score
    err_model = m.reconstruction_error_model

    # Methods
    rows: list[dict] = []

    # Limit how many anomalies we explain in benchmark mode to avoid long
    # runtimes on large synthetic datasets.
    # Default: full topk.
    max_anomalies = min(topk, 5)
    top_idx_bench = top_idx[:max_anomalies]

    for mi, method in enumerate(method_list):

        # Reproducibility per method

        base_seed = int(abs(hash((device, target_n, method, seed))) % 100_000)

       # 🌟 SỬA TÊN THAM SỐ TỪ seed_ THÀNH seed ĐỂ PHÙ HỢP VỚI src/metrics.py
        if method == "LIME-CA":
            def explain_fn(score_fn_, X_, idx, seed, **kw):  # Đổi seed_ thành seed
                return ex.lime_ca_context_reuse_explain(
                    score_fn_, X_, idx,
                    num_samples=150, 
                    tau=0.5,
                    seed=seed        # Đổi seed_ thành seed
                )
        elif method == "IntegratedGradients":
            def explain_fn(score_fn_, X_, idx, seed, **kw):
                # Integrated Gradients requires a TF model whose forward pass is
                # the scalar anomaly score. When we load cached numpy weights,
                # `err_model` may be None; in that case, fall back to a TF model
                # created in the normal way.
                b_vector = np.mean(X_[normal_idx], axis=0)
                import integrated_gradients_impl as ig_impl

                nonlocal err_model
                score_model = err_model
                if score_model is None:
                    # Build TF model with deterministic weights loaded into Dense layers.
                    # We can reuse the same dense weights stored in `m`.
                    import tensorflow as tf
                    tf.random.set_seed(42)
                    # Ensure a fresh err_model exists.
                    temp_model = models.MLPAEModel()
                    # Copy weights/activations from cached model.
                    temp_model._weights = m._weights
                    temp_model._activations = m._activations
                    _, score_model = temp_model._build()  # create tf models
                    err_model = score_model

                return ig_impl.integrated_gradients_explain(
                    score_model=score_model,
                    X=X_,
                    anomaly_idx=idx,
                    baseline=b_vector,
                    steps=50,
                    seed=seed,
                )

        else:
            raise ValueError(method)

        # Measure runtime+memory.
        # Kept intentionally low so that 50k/100k/1M can finish.
        # If you need more stable statistics, increase RUNS/REPS.
        RUNS = 1
        REPS = 1


        # tracemalloc measures Python allocations; TensorFlow/native memory
        # might not be fully reflected, but it's still useful for relative scaling.
        tracemalloc.start()
        gc.collect()
        t0 = time.time()
        res = metrics.evaluate_combo(
            score_fn,
            X,
            top_idx_bench,
            normal_idx,
            explain_fn,
            runs=RUNS,
            reps=REPS,
            seed_offset=base_seed,
        )
        elapsed = time.time() - t0
        mem_peak_mb = approximate_memory_peak_mb()
        tracemalloc.stop()

        rows.append({
            "device": device,
            "method": method,
            "model": "MLP",
            "target_n": int(target_n),
            "normal_pool_size": int(len(normal_idx)),
            "topk": int(topk),
            "jitter_std": float(jitter_std),
            "seed": int(seed),
            "runtime_wallclock_s": float(elapsed),
            "memory_peak_mb": float(mem_peak_mb),
            # From metrics.evaluate_combo
            "stability_mean": res["stability_mean"],
            "fidelity_mean": res["fidelity_mean"],
            "consistency_mean": res["consistency_mean"],
            "sensitivity_mean": res["sensitivity_mean"],
            "runtime_mean_s": res["runtime_mean"],
            "runtime_std_s": res["runtime_std"],
            "n_calls_mean": res["n_calls_mean"],
        })

        # free some memory before next method
        gc.collect()

    return rows


def main():
    # Reproducibility: lock TF/NumPy/py seeds as early as possible.
    os.environ['PYTHONHASHSEED'] = '42'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # oneDNN may introduce tiny numeric differences affecting explanation outputs.
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    import random
    import numpy as np

    np.random.seed(42)
    random.seed(42)

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="fridge", choices=["fridge", "printer"])
    parser.add_argument("--topk", type=int, default=20)

    parser.add_argument("--factors", type=int, nargs="*", default=[1, 5, 10, 20])
    parser.add_argument("--targets", type=int, nargs="*", default=[],
                        help="Direct target sample sizes, e.g. --targets 50000 100000 1000000")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jitter_std", type=float, default=0.01)

    args = parser.parse_args()

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "logs"), exist_ok=True)

    X_base_df = data_prep.load_device_dataframe(args.device)
    X_base = X_base_df[data_prep.FEATURE_COLUMNS].values.astype(np.float32)

    base_n = X_base.shape[0]
    targets: list[int] = []

    for f in args.factors:
        targets.append(int(base_n * f))
    for t in args.targets:
        targets.append(int(t))

    # Deduplicate while keeping order
    seen = set()
    targets_ordered = []
    for t in targets:
        if t not in seen:
            seen.add(t)
            targets_ordered.append(t)

    method_list = ["LIME-CA", "IntegratedGradients"]

    out_path = os.path.join(
        TABLES_DIR,
        "table_dataset_scaling_benchmark_mlp_limeca_ig.csv",
    )

    all_rows: list[dict] = []
    for target_n in targets_ordered:
        log(f"=== Benchmark device={args.device} target_n={target_n} base_n={base_n} ===")

        rows = run_one_config(
            device=args.device,
            X_base=X_base,
            target_n=target_n,
            topk=args.topk,
            seed=args.seed,
            jitter_std=args.jitter_std,
            method_list=method_list,
        )
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(out_path, index=False)

    log(f"DONE wrote {out_path}")


if __name__ == "__main__":
    main()

