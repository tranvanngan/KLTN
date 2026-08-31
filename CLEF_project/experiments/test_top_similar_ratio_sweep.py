"""test_top_similar_ratio_sweep.py
================================

Sweep the "Top Similar Points" ratio used by LIME-CA context-reuse sampling.

Interpretation in this codebase:
- In src/explainers.py, LIME-CA context-reuse picks base points from the top-k
  most similar points (by cosine similarity of the context vector).
- Historically that k was fixed to 20% of all points.
- This test sweeps k-ratio in [0.10, 0.20, 0.30, 0.40, 0.50] while keeping
  everything else unchanged.

Runs a *light* experiment setup per user request:
- device: fridge
- model: IF (single detector)
- normal pool: computed using the model's default threshold (95th percentile)

Output:
  results/tables/table_top_similar_ratio_limeca.csv
"""

from __future__ import annotations
# ĐẶT BLOCK KHÓA SEED TOÀN CỤC 
import os
import random
import numpy as np

os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)
random.seed(42)

try:
    import tensorflow as tf
    tf.random.set_seed(42)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
except ImportError:
    pass

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

# Keep consistent with existing sensitivity scripts
RUNS = 10
REPS = 3
TOPK = 20
LIME_NSAMPLES = 150
TAU = 0.5

TOP_SIMILAR_RATIOS = [0.10, 0.20, 0.30, 0.40, 0.50]

DEVICE = "fridge"
MODEL_NAME = "IF"

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

def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    out_path = os.path.join(TABLES_DIR, "table_top_similar_ratio_limeca.csv")
    rows: list[dict] = []

    df = data_prep.load_device_dataframe(DEVICE)
    X = df[data_prep.FEATURE_COLUMNS].values.astype(np.float32)

    log(f"=== Fitting model={MODEL_NAME} on device={DEVICE} ===")
    fitted = models.build_and_fit_all(X)
    m = fitted[MODEL_NAME]

    score_fn = m.score
    scores = score_fn(X)
    normal_idx = np.where(scores <= m.threshold)[0]
    top_idx = m.top_k_indices(X, TOPK)

    log(
        f"normal_pool_size={len(normal_idx)} threshold={m.threshold:.6f} topk={len(top_idx)}"
    )

    for ratio in TOP_SIMILAR_RATIOS:
        log(f"=== ratio(top similar)={ratio:.2f} | running LIME-CA ===")

        fn = partial(
            ex.lime_ca_context_reuse_explain,
            num_samples=LIME_NSAMPLES,
            tau=TAU,
            context_reuse_ratio=ratio,
        )

        # Để so khớp pipeline full: seed không phụ thuộc vào ratio.
        # (Pipeline full seed_offset(device, model, method) cũng không phụ thuộc ratio.)
        seedbase = seed_offset(DEVICE, MODEL_NAME, "LIME-CA")

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
            "ratio": ratio,
            "model": MODEL_NAME,
            "method": "LIME-CA",
            "normal_pool_size": int(len(normal_idx)),
            "topk": int(TOPK),
        }
        for k, v in res.items():
            if k != "_per_run":
                row[k] = v

        rows.append(row)

        log(
            "  LIME-CA "
            f"stab={row['stability_mean']:.4f} fid={row['fidelity_mean']:.6f} "
            f"cons={row['consistency_mean']:.4f} sens={row['sensitivity_mean']:.4f} "
            f"runtime={row['runtime_mean']*1000:.2f}ms "
            f"ncalls={row['n_calls_mean']:.0f} ({dt:.1f}s)"
        )

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    log(f"===== DONE | wrote {out_path} =====")


if __name__ == "__main__":
    main()

