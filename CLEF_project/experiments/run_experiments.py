"""
run_experiments.py
===================
Master experiment script for the 2-device (Fridge, Printer) CLEF study.

Produces (all under results/tables/ and results/logs/):
  - per_anomaly_metrics_<device>.csv      : raw per-(model,method) summary
  - table_xai_metrics_<device>.csv        : Tables 2-3 equivalent
  - table_aggregated.csv                  : Table 6 equivalent (median/IQR)
  - table_cross_device_RI.csv             : Table 7 equivalent
  - table_runtime_complexity.csv          : NEW - runtime / #calls trade-off
  - table_ablation.csv                    : Table 8 equivalent (Fridge, IF)
  - table_tau_sensitivity.csv             : Table 9 equivalent (Fridge, IF)
  - table_alignment.csv                   : Table 10 equivalent
  - table_complexity_theoretical.csv      : NEW - Big-O complexity summary
  - run_log.txt                           : progress / timing log
"""

from __future__ import annotations
import os
import random
import numpy as np

os.environ['PYTHONHASHSEED'] = '42'  # Tắt tính năng ngẫu nhiên hóa hash của Python
np.random.seed(42)                  # Khóa seed toàn cục cho NumPy
random.seed(42)                    # Khóa seed toàn cục cho thư viện random

# Khóa seed cho TensorFlow nếu có chạy mô hình MLP
try:
    import tensorflow as tf
    tf.random.set_seed(42)
    os.environ['TF_DETERMINISTIC_OPS'] = '1' # Ép các hàm tính toán của GPU/CPU chạy tuần tự
except ImportError:
    pass

import os
import sys
import json
import time
from functools import partial

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data_prep
import models
import explainers as ex
import metrics
import alignment
import stats_tests

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
LOG_PATH = os.path.join(RESULTS_DIR, "logs", "run_log.txt")

DEVICES = ["fridge", "printer"]
RUNS = 10
REPS = 3
TOPK = 20
SHAP_NSAMPLES = 64
LIME_NSAMPLES = 150
K_BG = 50
TAU = 0.5

LOG_FH = open(LOG_PATH, "a")



def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_FH.write(line + "\n")
    LOG_FH.flush()



#  XÓA HÀM CŨ VÀ THAY BẰNG ĐOẠN NÀY:
def seed_offset(*parts):
    import hashlib
    string_to_hash = "_".join(str(p) for p in parts)
    # Băm MD5 chuỗi ký tự, lấy giá trị hex và chuyển sang số nguyên để làm seed cố định
    hash_int = int(hashlib.md5(string_to_hash.encode('utf-8')).hexdigest(), 16)
    return hash_int % 100_000

def get_method_fns(model_name, m, normal_idx):
    fns = {
        "SHAP": partial(
            ex.shap_explain, normal_idx=normal_idx, k=K_BG, tau=TAU, nsamples=SHAP_NSAMPLES
        ),
        "LIME-std": partial(ex.lime_explain, num_samples=LIME_NSAMPLES),
        "LIME-CA": partial(ex.lime_ca_context_reuse_explain, num_samples=LIME_NSAMPLES),


    }

    if model_name == "MLP":
        err_model = m.reconstruction_error_model

        def deepshap_adapter(score_fn, X, idx, seed, **kw):
            return ex.deepshap_explain(err_model, X, idx, normal_idx, k=K_BG, tau=TAU, seed=seed)

        def ig_adapter(score_fn, X, idx, seed, steps=50, **kw):
            return ex.integrated_gradients_explain(
                err_model,
                X,
                idx,
                baseline=None,
                steps=steps,
                seed=seed,
            )

        fns["DeepSHAP"] = deepshap_adapter
        fns["IntegratedGradients"] = partial(ig_adapter, steps=50)
    return fns



# ---------------------------------------------------------------------------
# 1. Main quantitative comparison (Tables 2-3, 6-7) + runtime (NEW)
# ---------------------------------------------------------------------------
def run_main_comparison():
    out_path = os.path.join(TABLES_DIR, "main_results_raw.csv")
    done = set()
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        done = set(zip(existing.device, existing.model, existing.method))
        log(f"Resuming: {len(done)} (device,model,method) combos already done.")
    else:
        existing = pd.DataFrame()

    rows = []
    for device in DEVICES:
        df = data_prep.load_device_dataframe(device)
        X = df[data_prep.FEATURE_COLUMNS].values.astype(np.float32)
        log(f"=== Device: {device} | fitting models ===")
        fitted = models.build_and_fit_all(X)

        for model_name, m in fitted.items():
            scores = m.score(X)
            normal_idx = np.where(scores <= m.threshold)[0]
            top_idx = m.top_k_indices(X, TOPK)
            score_fn = m.score
            method_fns = get_method_fns(model_name, m, normal_idx)

            for method_name, fn in method_fns.items():
                if (device, model_name, method_name) in done:
                    log(f"  SKIP (already done) {device:8s} {model_name:6s} {method_name:9s}")
                    continue
                t0 = time.time()
                res = metrics.evaluate_combo(
                    score_fn, X, top_idx, normal_idx, fn,
                    runs=RUNS, reps=REPS,
                    seed_offset=seed_offset(device, model_name, method_name),
                )
                dt = time.time() - t0
                row = {"device": device, "model": model_name, "method": method_name}
                for k, v in res.items():
                    if k != "_per_run":
                        row[k] = v

                # Also export per-run raw values so we can do Wilcoxon + Holm later.
                # This makes each (device, model, method) have vectors of length RUNS.
                for metric_key in ["stability", "fidelity", "consistency", "sensitivity"]:
                    perrun_vals = res["_per_run"][metric_key]
                    for r_i, vv in enumerate(perrun_vals):
                        row[f"{metric_key}_run{r_i}"] = float(vv)

                # Optional runtime per-run (not used for Wilcoxon in this request, but useful).
                # Keeping it consistent with other metrics.
                for metric_key in ["runtime", "n_calls"]:
                    perrun_vals = res["_per_run"][metric_key]
                    for r_i, vv in enumerate(perrun_vals):
                        row[f"{metric_key}_run{r_i}"] = float(vv)

                rows.append(row)

                log(
                    f"  {device:8s} {model_name:6s} {method_name:9s} | "
                    f"stab={row['stability_mean']:.4f} fid={row['fidelity_mean']:.6f} "
                    f"cons={row['consistency_mean']:.4f} sens={row['sensitivity_mean']:.4f} "
                    f"runtime={row['runtime_mean']*1000:.2f}ms ncalls={row['n_calls_mean']:.0f} "
                    f"({dt:.1f}s)"
                )
                # Checkpoint after every combo so a restart can resume.
                combo_df = pd.DataFrame([row])
                header = not os.path.exists(out_path)
                combo_df.to_csv(out_path, mode="a", index=False, header=header)

    if os.path.exists(out_path):
        df_all = pd.read_csv(out_path)
    else:
        df_all = pd.DataFrame(rows)
    return df_all


def build_xai_tables(df_all: pd.DataFrame):
    for device in DEVICES:
        sub = df_all[df_all.device == device]
        out = sub[[
            "model", "method", "stability_mean", "stability_std",
            "fidelity_mean", "fidelity_std", "consistency_mean", "consistency_std",
            "sensitivity_mean", "sensitivity_std",
        ]].copy()
        out.to_csv(os.path.join(TABLES_DIR, f"table_xai_metrics_{device}.csv"), index=False)


def build_aggregated_table(df_all: pd.DataFrame):
    rows = []
    for method in ["SHAP", "LIME-std", "LIME-CA", "IntegratedGradients"]:
        sub = df_all[df_all.method == method]

        row = {"method": method}
        for metric_name in ["stability_mean", "fidelity_mean", "consistency_mean", "sensitivity_mean"]:
            vals = sub[metric_name].values
            row[f"{metric_name}_median"] = float(np.median(vals))
            row[f"{metric_name}_q1"] = float(np.percentile(vals, 25))
            row[f"{metric_name}_q3"] = float(np.percentile(vals, 75))
            row[f"{metric_name}_n"] = len(vals)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLES_DIR, "table_aggregated.csv"), index=False)
    return out


def build_cross_device_table(df_all: pd.DataFrame):
    rows = []
    for device in DEVICES:
        sub = df_all[df_all.device == device]
        std_ = sub[sub.method == "LIME-std"].set_index("model")
        ca_ = sub[sub.method == "LIME-CA"].set_index("model")
        models_common = std_.index.intersection(ca_.index)

        def ri_lower_better(a, b):
            return float((a - b) / (np.abs(a) + 1e-12) * 100)

        def ri_higher_better(a, b):
            return float((b - a) / (np.abs(a) + 1e-12) * 100)

        stab_ri, fid_ri, cons_ri, sens_ri = [], [], [], []
        for mod in models_common:
            stab_ri.append(ri_lower_better(std_.loc[mod, "stability_mean"], ca_.loc[mod, "stability_mean"]))
            fid_ri.append(ri_lower_better(std_.loc[mod, "fidelity_mean"], ca_.loc[mod, "fidelity_mean"]))
            cons_ri.append(ri_higher_better(std_.loc[mod, "consistency_mean"], ca_.loc[mod, "consistency_mean"]))
            sens_ri.append(ri_lower_better(std_.loc[mod, "sensitivity_mean"], ca_.loc[mod, "sensitivity_mean"]))

        rows.append({
            "device": device,
            "stability_RI": float(np.mean(stab_ri)),
            "fidelity_RI": float(np.mean(fid_ri)),
            "consistency_RI": float(np.mean(cons_ri)),
            "sensitivity_RI": float(np.mean(sens_ri)),
        })

    out = pd.DataFrame(rows)
    avg = {"device": "Average"}
    for c in ["stability_RI", "fidelity_RI", "consistency_RI", "sensitivity_RI"]:
        avg[c] = float(out[c].mean())
    out = pd.concat([out, pd.DataFrame([avg])], ignore_index=True)
    out.to_csv(os.path.join(TABLES_DIR, "table_cross_device_RI.csv"), index=False)
    return out


def build_runtime_table(df_all: pd.DataFrame):
    rows = []
    for method in df_all.method.unique():
        sub = df_all[df_all.method == method]
        rows.append({
            "method": method,
            "runtime_ms_mean": float(sub.runtime_mean.mean() * 1000),
            "runtime_ms_std": float(sub.runtime_mean.std() * 1000),
            "n_calls_mean": float(sub.n_calls_mean.mean()),
        })
    out = pd.DataFrame(rows)
    base = out.loc[out.method == "LIME-std", "runtime_ms_mean"].values[0]
    out["overhead_vs_LIME_std_pct"] = (out.runtime_ms_mean - base) / base * 100
    out.to_csv(os.path.join(TABLES_DIR, "table_runtime_complexity.csv"), index=False)

    detail = df_all[["device", "model", "method", "runtime_mean", "runtime_std", "n_calls_mean"]].copy()
    detail["runtime_ms_mean"] = detail.runtime_mean * 1000
    detail["runtime_ms_std"] = detail.runtime_std * 1000
    detail.to_csv(os.path.join(TABLES_DIR, "table_runtime_detail.csv"), index=False)
    return out


# ---------------------------------------------------------------------------
# 2. Theoretical complexity table (NEW)
# ---------------------------------------------------------------------------
def run_friedman_nemenyi_main(df_all: pd.DataFrame, alpha: float = 0.05):
    """Friedman + Nemenyi on main pipeline results.

    Blocks: each (device, model) combo.
    Methods: SHAP, LIME-std, LIME-CA.
    Metrics: stability (lower better), fidelity (lower), consistency (higher), sensitivity (lower).

    Outputs:
      - results/tables/main_friedman_nemenyi_by_metric.csv
      - results/tables/main_friedman_nemenyi_pairwise_by_metric.csv
    """

    methods = ["SHAP", "LIME-std", "LIME-CA"]

    block_pairs = (
        df_all[["device", "model"]]
        .drop_duplicates()
        .apply(lambda r: f"{r['device']}||{r['model']}", axis=1)
        .tolist()
    )

    metrics_cfg = [
        ("stability_mean", "stability", False),  # higher_is_better? False => smaller better
        ("fidelity_mean", "fidelity", False),
        ("consistency_mean", "consistency", True),  # higher better
        ("sensitivity_mean", "sensitivity", False),
    ]

    by_metric_rows = []
    pair_rows = []

    for metric_col, metric_name, higher_is_better in metrics_cfg:
        mat = stats_tests.build_friedman_matrix(
            df_all,
            blocks=block_pairs,
            methods=methods,
            metric_col=metric_col,
            higher_is_better=higher_is_better,
            block_cols=("device", "model"),
        )

        fried, nemenyi_pairs, avg_ranks = stats_tests.friedman_nemenyi(
            mat, methods=methods, alpha=alpha
        )

        by_metric_rows.append({
            "metric": metric_name,
            "friedman_statistic": fried.statistic,
            "friedman_pvalue": fried.pvalue,
            "alpha": alpha,
        })

        for pr in nemenyi_pairs:
                # Use Nemenyi Critical Difference (CD) decision instead of p-value threshold.
                # This aligns with typical Nemenyi CD tables/plots.
                sig = "*" if getattr(pr, "reject", False) else "ns"
                pair_rows.append({
                    "metric": metric_name,
                    "method_i": pr.method_i,
                    "method_j": pr.method_j,
                    "avg_rank_i": float(avg_ranks[methods.index(pr.method_i)]),
                    "avg_rank_j": float(avg_ranks[methods.index(pr.method_j)]),
                    "rank_diff": pr.rank_diff,
                    "q_stat": pr.q_stat,
                    "nemenyi_pvalue": pr.pvalue,
                    "critical_diff": float(getattr(pr, "critical_diff", np.nan)),
                    "nemenyi_reject": bool(getattr(pr, "reject", False)),
                    "significance": sig,
                    "alpha": alpha,
                })

    by_metric = pd.DataFrame(by_metric_rows)
    by_metric.to_csv(os.path.join(TABLES_DIR, "main_friedman_nemenyi_by_metric.csv"), index=False)

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(os.path.join(TABLES_DIR, "main_friedman_nemenyi_pairwise_by_metric.csv"), index=False)

    log("Friedman+Nemenyi written to results/tables/*.csv")
    return by_metric, pair_df


def build_theoretical_complexity_table():
    M = SHAP_NSAMPLES
    B = K_BG
    N = LIME_NSAMPLES
    d = len(data_prep.FEATURE_COLUMNS)
    rows = [
        {
            "method": "SHAP (context-aware KernelSHAP)",
            "model_calls_per_explanation": f"O(M x B) = {M} x {B} = {M*B}",
            "extra_overhead": f"O(B x m) cosine similarity for background selection (m={len(ex.CONTEXT_IDX)})",
            "memory": f"O(B x d), d={d}",
            "notes": "M = KernelSHAP coalition samples, B = context-aware background size",
        },
        {
            "method": "LIME-std",
            "model_calls_per_explanation": f"O(N) = {N}",
            "extra_overhead": "O(N x d) Ridge regression fit",
            "memory": f"O(N x d), d={d}",
            "notes": "N = perturbation samples; single black-box batch call",
        },
        {
            "method": "LIME-CA",
            "model_calls_per_explanation": f"O(N) = {N} (amortised)",
            "extra_overhead": f"O((N/p) x m) rejection sampling, p=acceptance rate, m={len(ex.CONTEXT_IDX)}; "
                              f"O(N x d) Ridge fit",
            "memory": f"O(N x d), d={d}",
            "notes": "Extra cost vs. LIME-std = context-similarity filter on (N/p) candidates",
        },
        {
            "method": "DeepSHAP (GradientExplainer, MLP-AE only)",
            "model_calls_per_explanation": f"O(B') forward+backward passes, B'={K_BG}",
            "extra_overhead": "O(B' x m) cosine-similarity background selection",
            "memory": f"O(B' x d), d={d}",
            "notes": "B' = context-aware background reference set for expected gradients",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLES_DIR, "table_complexity_theoretical.csv"), index=False)
    return out


# ---------------------------------------------------------------------------
# 3. Ablation study (Fridge, IF, LIME-CA) - Table 8 equivalent
# ---------------------------------------------------------------------------
def run_ablation():
    df = data_prep.load_device_dataframe("fridge")
    X = df[data_prep.FEATURE_COLUMNS].values.astype(np.float32)
    fitted = models.build_and_fit_all(X)
    m = fitted["IF"]
    scores = m.score(X)
    normal_idx = np.where(scores <= m.threshold)[0]
    top_idx = m.top_k_indices(X, TOPK)
    score_fn = m.score

    cols = data_prep.FEATURE_COLUMNS
    configs = {
        "Full context (hour+temp+occ)": [cols.index(c) for c in ["hour", "temperature", "occupancy"]],
        "No occupancy (hour+temp)": [cols.index(c) for c in ["hour", "temperature"]],
        "No temperature (hour+occ)": [cols.index(c) for c in ["hour", "occupancy"]],
        "Only time (hour)": [cols.index(c) for c in ["hour"]],
    }
    rows = []
    for cfg_name, cidx in configs.items():
        fn = partial(ex.lime_ca_explain, num_samples=LIME_NSAMPLES, tau=TAU, context_idx=cidx)
        res = metrics.evaluate_combo(
            score_fn, X, top_idx, normal_idx, fn, runs=RUNS, reps=REPS,
            seed_offset=seed_offset("ablation", cfg_name),
        )
        rows.append({
            "configuration": cfg_name,
            "consistency": res["consistency_mean"],
            "fidelity_mse": res["fidelity_mean"],
        })
        log(f"  ablation [{cfg_name}] cons={res['consistency_mean']:.4f} fid={res['fidelity_mean']:.6f}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLES_DIR, "table_ablation.csv"), index=False)
    return out


# ---------------------------------------------------------------------------
# 4. Tau sensitivity (Fridge, IF, LIME-CA) - Table 9 equivalent
# ---------------------------------------------------------------------------
def run_tau_sensitivity():
    df = data_prep.load_device_dataframe("fridge")
    X = df[data_prep.FEATURE_COLUMNS].values.astype(np.float32)
    fitted = models.build_and_fit_all(X)
    m = fitted["IF"]
    scores = m.score(X)
    normal_idx = np.where(scores <= m.threshold)[0]
    top_idx = m.top_k_indices(X, TOPK)
    score_fn = m.score

    rows = []
    taus = [0.1, 0.3, 0.5, 0.7, 0.9]
    # Use fixed IF+SHAP seeds to align with main pipeline comparisons.
    for tau in taus:

        fn = partial(ex.shap_explain, normal_idx=normal_idx, k=K_BG, tau=tau, nsamples=SHAP_NSAMPLES)

        res = metrics.evaluate_combo(
            score_fn, X, top_idx, normal_idx, fn, runs=RUNS, reps=REPS,
        # Fix seed so tau-sensitivity (tau=TAU) matches main comparison sampling.
        seed_offset=seed_offset("fridge", "IF", "SHAP"),


        )
        rows.append({
            "tau": tau,
            "consistency": res["consistency_mean"],
            "fidelity_mse": res["fidelity_mean"],
        })
        log(f"  tau={tau} cons={res['consistency_mean']:.4f} fid={res['fidelity_mean']:.6f}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLES_DIR, "table_tau_sensitivity.csv"), index=False)
    return out


# ---------------------------------------------------------------------------
# 5. Occupancy-aware Alignment Score - Table 10 equivalent
# ---------------------------------------------------------------------------
def run_alignment(extra_k: int = 40):
    rows = []
    for device in DEVICES:
        df = data_prep.load_device_dataframe(device)
        X = df[data_prep.FEATURE_COLUMNS].values.astype(np.float32)
        fitted = models.build_and_fit_all(X)
        for model_name, m in fitted.items():
            scores = m.score(X)
            normal_idx = np.where(scores <= m.threshold)[0]
            top_idx = m.top_k_indices(X, extra_k)
            phis = []
            for rank, idx in enumerate(top_idx):
                phi, _, _ = ex.shap_explain(
                    m.score, X, int(idx), normal_idx, k=K_BG, tau=TAU,
                    nsamples=SHAP_NSAMPLES, seed=seed_offset(device, model_name, "align", rank),
                )
                phis.append(phi)
            phis = np.array(phis)
            res = alignment.alignment_analysis(X, top_idx, phis)
            res["device"] = device
            res["model"] = model_name
            rows.append(res)
            log(
                f"  alignment {device:8s} {model_name:6s} | "
                f"human={res['human_mean']:.4f}({res['human_n']}) "
                f"device={res['device_mean']:.4f}({res['device_n']}) p={res['p_value']}"
            )
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLES_DIR, "table_alignment.csv"), index=False)
    return out


if __name__ == "__main__":
    t_start = time.time()
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    log(f"===== STARTING RUN (stage={stage}) =====")

    if stage in ("all", "main"):
        df_all = run_main_comparison()
    if stage in ("all", "tables"):
        df_all = pd.read_csv(os.path.join(TABLES_DIR, "main_results_raw.csv"))

        # main_results_raw.csv có thể bị ghi trùng do chạy lại nhiều lần trước đó.

        # Để tính bảng ổn định, gộp các dòng trùng theo (device, model, method).
        # main_results_raw.csv có thể bị ghi trùng do chạy lại nhiều lần.
        # Để tính bảng ổn định, gộp các dòng trùng theo (device, model, method).
        # Với các cột per-run (ví dụ stability_run0..), ta cũng trung bình theo cùng index run.
        # Với các cột mean/std ta cũng trung bình tương tự.
        exclude_cols = {"device", "model", "method"}
        metric_cols = [c for c in df_all.columns if c not in exclude_cols]
        metric_cols = [c for c in metric_cols if df_all[c].dtype != "object"]

        df_all = (
            df_all
            .groupby(["device", "model", "method"], as_index=False)[metric_cols]
            .mean()
        )


        build_xai_tables(df_all)
        build_aggregated_table(df_all)
        build_cross_device_table(df_all)
        build_runtime_table(df_all)
        build_theoretical_complexity_table()

        # NEW: Friedman test + Nemenyi post-hoc on pipeline main results
        run_friedman_nemenyi_main(df_all)

        # NEW: Wilcoxon + Holm post-hoc on pipeline main results
        # Note: if you want to run only this statistical block, execute stage="tables".
        wilcoxon_out_path = os.path.join(TABLES_DIR, "main_wilcoxon_holm_pairwise_by_metric.csv")
        try:
            wilcoxon_pair_df = stats_tests.wilcoxon_holm_main(df_all, alpha=0.05)
            # Ensure dataframe is materialized before writing
            wilcoxon_pair_df = pd.DataFrame(wilcoxon_pair_df)
            log(f"Wilcoxon+Holm: rows={len(wilcoxon_pair_df)} cols={list(wilcoxon_pair_df.columns)[:8]}...")
            wilcoxon_pair_df.to_csv(wilcoxon_out_path, index=False)
        except Exception as e:
            # Write an empty file so existence can be checked by the caller.
            log(f"Wilcoxon+Holm FAILED: {type(e).__name__}: {e}")
            pd.DataFrame({"error": [f"{type(e).__name__}: {e}"]}).to_csv(wilcoxon_out_path, index=False)





    if stage in ("all", "ablation"):
        run_ablation()
    if stage in ("all", "tau"):
        run_tau_sensitivity()
    if stage in ("all", "alignment"):
        run_alignment()

    log(f"===== DONE (stage={stage}) in {time.time() - t_start:.1f}s =====")
