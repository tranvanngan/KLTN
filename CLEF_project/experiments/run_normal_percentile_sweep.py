"""
run_normal_percentile_sweep.py
================================
Sensitivity study for the definition of the "normal pool" used by
LIME-CA / context-aware SHAP.

Goal (paper-friendly):
- Sweep anomaly-score threshold percentiles P in {50,70,80,90,95}
- For each detector model in {IF, LOF, OCSVM, MLP} on *fridge*:
    - define normal_idx = {i | score(X_i) <= threshold_P}
    - explain the top-20 anomalies with LIME-std and LIME-CA
- Report mean +/- std across runs/reps using existing evaluate_combo().

Output:
  results/tables/table_normal_percentile_limeca.csv
"""

from __future__ import annotations

import os
import sys
import time
from functools import partial

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

# Match main-study defaults (except threshold percentile)
RUNS = 10
REPS = 3
TOPK = 20
LIME_NSAMPLES = 150
TAU = 0.5

PERCENTILES = [50, 70, 80, 90, 95]

DEVICE = "fridge"

MODELS_TO_RUN = ["IF", "LOF", "OCSVM", "MLP"]


LOG_FH = open(LOG_PATH, "a")

def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_FH.write(line + "\n")
    LOG_FH.flush()


def seed_offset(*parts):
    return abs(hash("_".join(str(p) for p in parts))) % 100_000


def get_lime_fns(num_samples: int = LIME_NSAMPLES):
    return {
        "LIME-std": ex.lime_explain,
        "LIME-CA": partial(ex.lime_ca_explain, tau=TAU),
    }


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)

    out_path = os.path.join(TABLES_DIR, "table_normal_percentile_limeca.csv")
    rows = []
    t_start = time.time()

    df = data_prep.load_device_dataframe(DEVICE)
    X = df[data_prep.FEATURE_COLUMNS].values.astype(np.float32)

    # Fit all models once to get consistent detector scores.
    fitted_all = models.build_and_fit_all(X)

    for percentile in PERCENTILES:
        log(f"=== percentile={percentile} | computing normal pool & explanations ===")

        for model_name in MODELS_TO_RUN:
            m = fitted_all[model_name]
            score_fn = m.score

            # recompute threshold for this percentile
            scores = score_fn(X)
            threshold = float(np.percentile(scores, percentile))
            normal_idx = np.where(scores <= threshold)[0]
            top_idx = m.top_k_indices(X, TOPK)

            log(
                f"  model={model_name:6s} threshold={threshold:.6f} | "
                f"normal_pool={len(normal_idx)} topk={len(top_idx)}"
            )

            lime_fns = get_lime_fns(num_samples=LIME_NSAMPLES)
            for method_name, fn in lime_fns.items():
                seedbase = seed_offset("normal_percentile", DEVICE, model_name, method_name, percentile)
                t0 = time.time()
                res = metrics.evaluate_combo(
                    score_fn,
                    X,
                    top_idx,
                    normal_idx,
                    fn,
                    runs=RUNS,
                    reps=REPS,
                    seed_offset=seedbase,
                )
                dt = time.time() - t0

                row = {
                    "device": DEVICE,
                    "percentile": percentile,
                    "model": model_name,
                    "method": method_name,
                    "normal_pool_size": int(len(normal_idx)),
                    "topk": int(TOPK),
                }
                for k, v in res.items():
                    if k != "_per_run":
                        row[k] = v
                    
                rows.append(row)
                log(
                    f"    {method_name:9s} | stab={row['stability_mean']:.4f} "
                    f"fid={row['fidelity_mean']:.6f} cons={row['consistency_mean']:.4f} "
                    f"sens={row['sensitivity_mean']:.4f} "
                    f"runtime={row['runtime_mean']*1000:.2f}ms "
                    f"ncalls={row['n_calls_mean']:.0f} ({dt:.1f}s)"
                )

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    log(f"===== DONE | wrote {out_path} in {time.time() - t_start:.1f}s =====")


if __name__ == "__main__":
    main()

