"""run_context_similarity_sensitivity_test.py
================================================
LIGHTWEIGHT unit-style sensitivity experiment (test-only)
for reviewer request: Context Similarity proof.

Design goals:
- Do NOT rerun the heavy main pipeline.
- Only compare how LIME-CA (context_reuse branch, i.e. base indices) behaves
  under different context similarity metrics.
- Report Consistency / Fidelity / Stability using existing
  metrics.evaluate_combo().

Important:
- You can run this with small settings (default below) so it finishes fast.

Output:
  results/tables/table_context_similarity_sensitivity_test.csv

What it changes vs main study:
- Introduces `context_metric` that affects which base indices are selected
  for LIME-CA (context_reuse sampling).
- Metrics computed are the same as evaluate_combo.

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

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

LOG_FH = open(LOG_PATH, "a")

def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_FH.write(line + "\n")
    LOG_FH.flush()


def seed_offset(*parts):
    import hashlib
    string_to_hash = "_".join(str(p) for p in parts)
    hash_int = int(hashlib.md5(string_to_hash.encode('utf-8')).hexdigest(), 16)
    return hash_int % 100_000


# --------------------------
# Test-only small settings
# --------------------------
DEVICE = "fridge"
MODEL_NAME = "MLP"  # tester cho context similarity (run nhanh hơn pipeline chính)

METHOD_NAME = "LIME-CA"

RUNS = 3
REPS = 2
TOPK = 10
LIME_NSAMPLES = 60

# Context similarity metrics to test
CONTEXT_METRICS = ["cosine", "euclidean", "mahalanobis"]


def get_context_metric_fn(context_metric: str):
    """Return a partial that calls LIME-CA with specified context metric.

    Note: requires code support added in src/explainers.py.
    """
    return partial(
        ex.lime_ca_context_reuse_explain,
        num_samples=LIME_NSAMPLES,
        tau=0.5,  # irrelevant for context_reuse, kept for signature
        context_metric=context_metric,
    )


def main():
    t_start = time.time()
    log("=== START context similarity sensitivity (TEST ONLY) ===")

    out_path = os.path.join(TABLES_DIR, "table_context_similarity_sensitivity_test.csv")
    rows = []

    df = data_prep.load_device_dataframe(DEVICE)
    X = df[data_prep.FEATURE_COLUMNS].values.astype(np.float32)

    fitted = models.build_and_fit_all(X)
    m = fitted[MODEL_NAME]
    scores = m.score(X)
    normal_idx = np.where(scores <= m.threshold)[0]
    top_idx = m.top_k_indices(X, TOPK)
    score_fn = m.score

    for cm in CONTEXT_METRICS:
        fn = get_context_metric_fn(cm)
        log(f"--- context_metric={cm} ---")

        res = metrics.evaluate_combo(
            score_fn,
            X,
            top_idx,
            normal_idx,
            fn,
            runs=RUNS,
            reps=REPS,
            seed_offset=seed_offset("ctxsim_test", DEVICE, MODEL_NAME, cm),
        )

        row = {
            "device": DEVICE,
            "model": MODEL_NAME,
            "method": METHOD_NAME,
            "context_metric": cm,
            "topk": TOPK,
            "runs": RUNS,
            "reps": REPS,
            "lime_num_samples": LIME_NSAMPLES,
        }
        for k, v in res.items():
            if k != "_per_run":
                row[k] = v
        rows.append(row)

        log(
            f"ctxsim={cm}: stab={row['stability_mean']:.4f} "
            f"fid={row['fidelity_mean']:.6f} cons={row['consistency_mean']:.4f} "
            f"sens={row['sensitivity_mean']:.4f}"
        )

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    log(f"=== DONE (wrote {out_path}) in {time.time()-t_start:.1f}s ===")


if __name__ == "__main__":
    main()

