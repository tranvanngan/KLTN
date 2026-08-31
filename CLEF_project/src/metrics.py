"""
metrics.py
==========
Evaluation metrics for explanation quality (Section 3.6), plus the new
runtime / computational-cost instrumentation requested for the trade-off
analysis.

For each (device, model, method) combination, `evaluate_combo` runs R
independent repetitions ("runs"). Within each run:

  - For each of the top-20 anomalies, the explanation is computed `reps`
    times (different seeds) -> per-anomaly Stability (CV of |phi| across
    the `reps` repetitions, averaged over features). The first repetition
    is used as the "main" explanation for that anomaly.
  - Fidelity (MSE of the local linear surrogate vs. the black-box score on
    the 50 nearest neighbours) and Sensitivity (|score change| after
    replacing the top-3 |phi| features with their normal-population median)
    are computed from the main explanation.
  - Consistency = mean pairwise cosine similarity of the 20 main
    explanation vectors.
  - Runtime = mean wall-clock time per explanation call in this run;
    n_calls = mean number of black-box evaluations per explanation call.

The R per-run values are then summarised as mean +/- std.
"""

from __future__ import annotations

import numpy as np

from explainers import fidelity_mse


def mean_pairwise_cosine(vectors: np.ndarray) -> float:
    n = vectors.shape[0]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    unit = vectors / norms
    sims = unit @ unit.T
    iu = np.triu_indices(n, k=1)
    return float(np.mean(sims[iu]))


def sensitivity_metric(score_fn, X, idx, phi, normal_idx, top_n=3):
    x0 = X[idx].copy()
    f_x0 = float(score_fn(x0[None, :])[0])
    order = np.argsort(np.abs(phi))[::-1][:top_n]
    medians = np.median(X[normal_idx], axis=0)
    x_pert = x0.copy()
    x_pert[order] = medians[order]
    f_pert = float(score_fn(x_pert[None, :])[0])
    return abs(f_x0 - f_pert)


def evaluate_combo(
    score_fn,
    X,
    top_idx: np.ndarray,
    normal_idx: np.ndarray,
    explain_fn,
    runs: int = 10,
    reps: int = 3,
    seed_offset: int = 0,
    explain_kwargs: dict | None = None,
) -> dict:
    explain_kwargs = explain_kwargs or {}
    per_run = {"stability": [], "fidelity": [], "consistency": [], "sensitivity": [],
               "runtime": [], "n_calls": []}

    for run in range(runs):
        explanation_vectors = []
        run_stab, run_fid, run_sens, run_time, run_calls = [], [], [], [], []

        for rank, idx in enumerate(top_idx):
            phis = []
            for rep in range(reps):
                seed = seed_offset + run * 10_000 + rank * 100 + rep
                phi, n_calls, elapsed = explain_fn(score_fn, X, int(idx), seed=seed, **explain_kwargs)
                phis.append(phi)
                run_time.append(elapsed)
                run_calls.append(n_calls)
            phis = np.array(phis)
            main_phi = phis[0]
            explanation_vectors.append(main_phi)

            # Stability definition: CV of |phi| across `reps`.
            # If `reps==1`, std=0 => stability is uninformative (0).
            # We handle this by returning NaN so the caller can filter/avoid
            # misleading zeros in FAST mode or degenerate settings.
            if reps <= 1:
                run_stab.append(float('nan'))
            else:
                mean_abs = np.mean(np.abs(phis), axis=0)
                std_abs = np.std(np.abs(phis), axis=0)
                cv = np.divide(
                    std_abs,
                    mean_abs,
                    out=np.zeros_like(mean_abs),
                    where=mean_abs > 1e-12,
                )
                run_stab.append(float(np.mean(cv)))


            run_fid.append(fidelity_mse(score_fn, X, int(idx), main_phi))
            run_sens.append(sensitivity_metric(score_fn, X, int(idx), main_phi, normal_idx))

        per_run["consistency"].append(mean_pairwise_cosine(np.array(explanation_vectors)))
        per_run["stability"].append(float(np.mean(run_stab)))
        per_run["fidelity"].append(float(np.mean(run_fid)))
        per_run["sensitivity"].append(float(np.mean(run_sens)))
        per_run["runtime"].append(float(np.mean(run_time)))
        per_run["n_calls"].append(float(np.mean(run_calls)))

    summary = {}
    for k, v in per_run.items():
        v = np.array(v)
        summary[f"{k}_mean"] = float(np.mean(v))
        summary[f"{k}_std"] = float(np.std(v))
    summary["_per_run"] = per_run
    return summary
