"""
run_complexity_scaling.py
==========================
Empirical runtime-scaling benchmark to accompany the theoretical Big-O
complexity table (table_complexity_theoretical.csv).

For the Fridge / IsolationForest combination (representative of the
sklearn-based detectors -- LOF/OCSVM follow the same pattern, MLP is
covered separately for DeepSHAP), we vary the key complexity parameter of
each method and record mean wall-clock runtime per explanation over the
top-20 anomalies (1 repetition each, 5 random seeds):

  - SHAP      : KernelSHAP coalition samples M in {16, 32, 64, 128, 256}
                (background size B fixed at 50)
  - LIME-std  : perturbation samples N in {50, 100, 150, 300, 600}
  - LIME-CA   : same N grid (context-similarity filter overhead included)
  - DeepSHAP  : context-aware background size B' in {10, 20, 50, 100, 150}
                (MLP-AE / Fridge)

Output: results/tables/table_complexity_scaling.csv
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data_prep
import models
import explainers as ex

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

TOPK = 20
N_SEEDS = 5


def bench_shap(score_fn, X, top_idx, normal_idx, M_values):
    rows = []
    for M in M_values:
        times = []
        for seed in range(N_SEEDS):
            for idx in top_idx:
                _, n_calls, el = ex.shap_explain(
                    score_fn, X, int(idx), normal_idx, k=50, tau=0.5, nsamples=M, seed=seed
                )
                times.append(el)
        rows.append({"method": "SHAP", "param_name": "nsamples (M)", "param_value": M,
                      "runtime_ms_mean": np.mean(times) * 1000, "runtime_ms_std": np.std(times) * 1000,
                      "n_calls": n_calls})
    return rows


def bench_lime(score_fn, X, top_idx, fn, label, N_values):
    rows = []
    for N in N_values:
        times = []
        for seed in range(N_SEEDS):
            for idx in top_idx:
                _, n_calls, el = fn(score_fn, X, int(idx), num_samples=N, seed=seed)
                times.append(el)
        rows.append({"method": label, "param_name": "num_samples (N)", "param_value": N,
                      "runtime_ms_mean": np.mean(times) * 1000, "runtime_ms_std": np.std(times) * 1000,
                      "n_calls": n_calls})
    return rows


def bench_deepshap(err_model, X, top_idx, normal_idx, B_values):
    rows = []
    for B in B_values:
        times = []
        for seed in range(N_SEEDS):
            for idx in top_idx:
                _, n_calls, el = ex.deepshap_explain(
                    err_model, X, int(idx), normal_idx, k=B, tau=0.5, seed=seed
                )
                times.append(el)
        rows.append({"method": "DeepSHAP", "param_name": "background size (B')", "param_value": B,
                      "runtime_ms_mean": np.mean(times) * 1000, "runtime_ms_std": np.std(times) * 1000,
                      "n_calls": n_calls})
    return rows


def main():
    df = data_prep.load_device_dataframe("fridge")
    X = df[data_prep.FEATURE_COLUMNS].values.astype(np.float32)

    fitted = models.build_and_fit_all(X)
    m_if = fitted["IF"]
    scores = m_if.score(X)
    normal_idx = np.where(scores <= m_if.threshold)[0]
    top_idx = m_if.top_k_indices(X, TOPK)

    m_mlp = fitted["MLP"]
    mlp_scores = m_mlp.score(X)
    mlp_normal_idx = np.where(mlp_scores <= m_mlp.threshold)[0]
    mlp_top_idx = m_mlp.top_k_indices(X, TOPK)

    rows = []
    t0 = time.time()
    rows += bench_shap(m_if.score, X, top_idx, normal_idx, [16, 32, 64, 128, 256])
    print(f"SHAP scaling done ({time.time()-t0:.1f}s)")

    t0 = time.time()
    rows += bench_lime(m_if.score, X, top_idx, ex.lime_explain, "LIME-std", [50, 100, 150, 300, 600])
    print(f"LIME-std scaling done ({time.time()-t0:.1f}s)")

    t0 = time.time()
    rows += bench_lime(m_if.score, X, top_idx, ex.lime_ca_explain, "LIME-CA", [50, 100, 150, 300, 600])
    print(f"LIME-CA scaling done ({time.time()-t0:.1f}s)")

    t0 = time.time()
    rows += bench_deepshap(m_mlp.reconstruction_error_model, X, mlp_top_idx, mlp_normal_idx,
                            [10, 20, 50, 100, 150])
    print(f"DeepSHAP scaling done ({time.time()-t0:.1f}s)")

    # Integrated Gradients scaling (MLP-AE only)
    # Key complexity parameter: number of IG steps.
    t0 = time.time()
    ig_steps_values = [10, 20, 50, 100, 200]
    rows += []
    for steps in ig_steps_values:
        times = []
        last_n_calls = None
        for seed in range(N_SEEDS):
            for idx in mlp_top_idx:
                _, n_calls, el = ex.integrated_gradients_explain(
                    m_mlp.reconstruction_error_model,
                    X,
                    int(idx),
                    baseline=X[int(idx)],
                    steps=steps,
                    seed=seed,
                )
                last_n_calls = n_calls
                times.append(el)
        # Use last_n_calls as representative (should be deterministic for IG)
        rows.append({
            "method": "IntegratedGradients",
            "param_name": "steps",
            "param_value": steps,
            "runtime_ms_mean": np.mean(times) * 1000,
            "runtime_ms_std": np.std(times) * 1000,
            "n_calls": last_n_calls if last_n_calls is not None else 0,
        })

    print(f"IntegratedGradients scaling done ({time.time()-t0:.1f}s)")


    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLES_DIR, "table_complexity_scaling.csv"), index=False)
    print(out)


if __name__ == "__main__":
    main()
