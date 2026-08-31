"""make_xai_shap_plots.py
======================
Create SHAP-based XAI plots:

  1) SHAP Summary Plot
  2) Local Explanation Comparison (SHAP vs LIME-CA)
  3) SHAP Force Plot
  4) Feature Importance Heatmap

This script is designed to run with the existing CLEF pipeline:
  - data_prep.py provides feature names and preprocessing
  - models.py provides anomaly-score models (IF/LOF/OCSVM/MLP-AE)
  - explainers.py provides SHAP/LIME explanation functions

Outputs are written under:
  results/figures/

Notes:
  - Force plot requires the SHAP JS dependency for interactive HTML.
    We therefore save BOTH:
      * results/figures/*.html  (interactive)
      * results/figures/*.png   (static fallback via matplotlib when possible)
  - KernelSHAP summary plots can be heavy; this script limits the number
    of explained points with --max_points.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from plot_utils import set_time_new_roman_font
import data_prep
import models
import explainers as ex


set_time_new_roman_font()

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

METHOD_COLORS = {
    "SHAP": "#4C72B0",
    "LIME-CA": "#55A868",
    "LIME-std": "#DD8452",
}


@dataclass
class PlotConfig:
    device: str
    model_name: str
    method_local_compare: str = "LIME-CA"  # compared against SHAP
    anomaly_rank: int = 0
    max_points: int = 30
    shap_nsamples: int = 64
    k_bg: int = 50
    tau: float = 0.5
    lime_nsamples: int = 150
    seed: int = 0


def _get_feature_labels() -> list[str]:
    return list(data_prep.FEATURE_COLUMNS)


def _load_X_for_device(device: str):
    df = data_prep.load_device_dataframe(device)
    X = df[data_prep.FEATURE_COLUMNS].values.astype(np.float32)
    return X


def _fit_model(X: np.ndarray, model_name: str):
    fitted = models.build_and_fit_selected(X, model_names=[model_name])
    return fitted[model_name]


def _select_anomalies(m, X: np.ndarray, topk: int = 50):
    scores = m.score(X)
    top_idx = m.top_k_indices(X, k=topk)
    normal_idx = np.where(scores <= m.threshold)[0]
    return top_idx, normal_idx


# ---------------------------------------------------------------------------
# 1) SHAP Summary Plot
# ---------------------------------------------------------------------------

def plot_shap_summary(cfg: PlotConfig):
    """SHAP Summary Plot (bar + beeswarm) saved as PNG.

    For KernelSHAP, we compute shap values for multiple points, but limited
    to cfg.max_points for runtime.
    """

    import shap

    X = _load_X_for_device(cfg.device)
    m = _fit_model(X, cfg.model_name)
    top_idx, normal_idx = _select_anomalies(m, X, topk=max(cfg.max_points, 10))
    score_fn = m.score

    # Choose subset of anomaly points
    explain_indices = top_idx[: cfg.max_points]

    # context-aware SHAP background via explainers.shap_explain internals
    # We reuse that by calling shap.KernelExplainer ourselves with a shared
    # background to make summary consistent.
    #
    # We'll mimic context_aware_background from explainers.py:
    bg_idx = ex.context_aware_background(
        X,
        anomaly_idx=int(explain_indices[0]),
        normal_idx=normal_idx,
        k=cfg.k_bg,
        tau=cfg.tau,
        rng=np.random.default_rng(cfg.seed),
    )
    background = X[bg_idx]

    def f(Z):
        return score_fn(Z)

    explainer = shap.KernelExplainer(f, background, seed=cfg.seed)
    shap_values = explainer.shap_values(
        X[explain_indices], nsamples=cfg.shap_nsamples, silent=True
    )

    # SHAP can return list[ndarray] for multi-output; here it's scalar output
    # so normalize to a (n_points, n_features) array.
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)

    feature_names = _get_feature_labels()

    # beeswarm (summary_plot) returns matplotlib fig/ax but also draws directly.
    # Save a clean PNG via shap's built-in summary_plot.
    plt.figure(figsize=(7.2, 4.8))
    shap.summary_plot(
        shap_values,
        X[explain_indices],
        feature_names=feature_names,
        show=False,
        plot_type="dot",
        max_display=len(feature_names),
    )
    out_path = os.path.join(FIG_DIR, f"fig_shap_summary_{cfg.device}_{cfg.model_name}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    # bar version (mean |SHAP|)
    plt.figure(figsize=(7.2, 4.8))
    shap.summary_plot(
        shap_values,
        X[explain_indices],
        feature_names=feature_names,
        show=False,
        plot_type="bar",
        max_display=len(feature_names),
    )
    out_path2 = os.path.join(
        FIG_DIR, f"fig_shap_summary_bar_{cfg.device}_{cfg.model_name}.png"
    )
    plt.tight_layout()
    plt.savefig(out_path2, dpi=200, bbox_inches="tight")
    plt.close()

    return {
        "shap_values": shap_values,
        "explain_indices": explain_indices,
        "background": background,
    }


# ---------------------------------------------------------------------------
# 2) Local Explanation Comparison
# ---------------------------------------------------------------------------

def _compute_local_phi_for_method(
    method_name: str,
    cfg: PlotConfig,
    X: np.ndarray,
    m,
    anomaly_idx: int,
    normal_idx: np.ndarray,
    seed: int,
):
    score_fn = m.score

    if method_name == "SHAP":
        phi, _, _ = ex.shap_explain(
            score_fn,
            X,
            anomaly_idx,
            normal_idx,
            k=cfg.k_bg,
            tau=cfg.tau,
            nsamples=cfg.shap_nsamples,
            seed=seed,
        )
        return phi

    if method_name == "LIME-std":
        phi, _, _ = ex.lime_explain(
            score_fn,
            X,
            anomaly_idx,
            num_samples=cfg.lime_nsamples,
            sigma=0.1,
            seed=seed,
        )
        return phi

    if method_name == "LIME-CA":
        phi, _, _ = ex.lime_ca_explain(
            score_fn,
            X,
            anomaly_idx,
            num_samples=cfg.lime_nsamples,
            sigma=0.1,
            tau=cfg.tau,
            seed=seed,
        )
        return phi

    if method_name == "DeepSHAP":
        if cfg.model_name != "MLP":
            raise ValueError("DeepSHAP chỉ áp dụng cho model_name=MLP")
        err_model = m.reconstruction_error_model
        phi, _, _ = ex.deepshap_explain(
            err_model,
            X,
            anomaly_idx,
            normal_idx,
            k=cfg.k_bg,
            tau=cfg.tau,
            seed=seed,
        )
        return phi

    raise ValueError(f"Unknown method: {method_name}")


def plot_local_comparison(cfg: PlotConfig):
    import shap

    X = _load_X_for_device(cfg.device)
    m = _fit_model(X, cfg.model_name)
    top_idx, normal_idx = _select_anomalies(m, X, topk=max(cfg.max_points, 10))

    # choose anomaly
    anomaly_idx = int(top_idx[cfg.anomaly_rank])

    # Demo-only local explanation comparison (Fridge/IF only):
    # show SHAP vs LIME-CA vs LIME-std.
    # (Bạn yêu cầu giữ LIME-std, nên mình thêm lại.)
    methods = ["SHAP", "LIME-std", "LIME-CA"]



    phis = {}
    for i, method in enumerate(methods):
        phis[method] = _compute_local_phi_for_method(
            method,
            cfg,
            X,
            m,
            anomaly_idx,
            normal_idx,
            seed=cfg.seed + i * 100,
        )

    feature_names = _get_feature_labels()

    # Trend-style visualization instead of many bars.
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    y = np.arange(len(feature_names))
    width = 0.25

    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width

    for j, method in enumerate(methods):
        phi = np.asarray(phis[method])
        ax.barh(
            y + offsets[j],
            phi,
            height=width,
            color=METHOD_COLORS.get(method, "#333333"),
            alpha=0.75,
            label=method,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(feature_names)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Local attribution (phi)")
    ax.set_title(
        f"Local Explanation Comparison (anomaly idx={anomaly_idx})\n"
        f"Device={cfg.device}, Model={cfg.model_name}"
    )
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()

    out_path = os.path.join(
        FIG_DIR,
        f"fig_local_compare_{cfg.device}_{cfg.model_name}_rank{cfg.anomaly_rank}.png",
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {"anomaly_idx": anomaly_idx, "phis": phis}


# ---------------------------------------------------------------------------
# 3) SHAP Force Plot
# ---------------------------------------------------------------------------

def plot_shap_force(cfg: PlotConfig):
    import shap

    X = _load_X_for_device(cfg.device)
    m = _fit_model(X, cfg.model_name)
    top_idx, normal_idx = _select_anomalies(m, X, topk=max(cfg.max_points, 10))

    explain_indices = top_idx[:1]
    anomaly_idx = int(explain_indices[0])

    bg_idx = ex.context_aware_background(
        X,
        anomaly_idx=anomaly_idx,
        normal_idx=normal_idx,
        k=cfg.k_bg,
        tau=cfg.tau,
        rng=np.random.default_rng(cfg.seed),
    )
    background = X[bg_idx]

    def f(Z):
        return m.score(Z)

    explainer = shap.KernelExplainer(f, background, seed=cfg.seed)

    shap_values = explainer.shap_values(
        X[anomaly_idx : anomaly_idx + 1], nsamples=cfg.shap_nsamples, silent=True
    )
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values).reshape(-1)

    feature_names = _get_feature_labels()

    force = shap.force_plot(
        explainer.expected_value,
        shap_values,
        X[anomaly_idx],
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )

    out_png = os.path.join(
        FIG_DIR,
        f"fig_shap_force_{cfg.device}_{cfg.model_name}_rank{cfg.anomaly_rank}.png",
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    # interactive HTML (best effort)
    try:
        force_html = shap.force_plot(
            explainer.expected_value,
            shap_values,
            X[anomaly_idx],
            feature_names=feature_names,
        )
        out_html = os.path.join(
            FIG_DIR,
            f"fig_shap_force_{cfg.device}_{cfg.model_name}_rank{cfg.anomaly_rank}.html",
        )
        shap.save_html(out_html, force_html)
    except Exception:
        pass

    return {"anomaly_idx": anomaly_idx, "shap_values": shap_values}


# ---------------------------------------------------------------------------
# 4) Feature Importance Heatmap
# ---------------------------------------------------------------------------

def plot_feature_importance_heatmap(cfg: PlotConfig):
    X = _load_X_for_device(cfg.device)
    m = _fit_model(X, cfg.model_name)
    top_idx, normal_idx = _select_anomalies(m, X, topk=max(cfg.max_points, 10))

    anomaly_indices = top_idx[: min(cfg.max_points, len(top_idx))]

    feature_names = _get_feature_labels()

    methods = ["SHAP", "LIME-std", "LIME-CA"]
    mat = []
    for method in methods:
        rows = []
        for r, anomaly_idx in enumerate(anomaly_indices):
            phi = _compute_local_phi_for_method(
                method,
                cfg,
                X,
                m,
                int(anomaly_idx),
                normal_idx,
                seed=cfg.seed + r * 37,
            )
            rows.append(np.abs(np.asarray(phi)))
        mat.append(np.mean(np.stack(rows, axis=0), axis=0))  # avg over anomalies

    mat = np.stack(mat, axis=0)  # (methods, features)

    fig, ax = plt.subplots(figsize=(8.8, 3.6))
    vmax = float(np.max(mat) + 1e-12)
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)

    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title(
        f"Feature Importance Heatmap (mean |attribution|)\n"
        f"Device={cfg.device}, Model={cfg.model_name}"
    )

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("mean |phi|")

    fig.tight_layout()
    out_path = os.path.join(
        FIG_DIR,
        f"fig_feature_importance_heatmap_{cfg.device}_{cfg.model_name}.png",
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {"methods": methods, "feature_names": feature_names, "mat": mat}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method_local_compare", type=str, default="LIME-CA", choices=["LIME-std", "LIME-CA"])
    parser.add_argument("--anomaly_rank", type=int, default=0)
    parser.add_argument("--max_points", type=int, default=30)
    parser.add_argument("--shap_nsamples", type=int, default=64)
    parser.add_argument("--k_bg", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--lime_nsamples", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Demo-only: fixed fridge + IF
    cfg = PlotConfig(
        device="fridge",
        model_name="IF",
        method_local_compare=args.method_local_compare,
        anomaly_rank=args.anomaly_rank,
        max_points=args.max_points,
        shap_nsamples=args.shap_nsamples,
        k_bg=args.k_bg,
        tau=args.tau,
        lime_nsamples=args.lime_nsamples,
        seed=args.seed,
    )

    plot_shap_summary(cfg)
    plot_local_comparison(cfg)
    plot_shap_force(cfg)
    plot_feature_importance_heatmap(cfg)
    print("Done. Figures written to", FIG_DIR)



if __name__ == "__main__":
    main()

