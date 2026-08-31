""" 
Standalone script for Fig 8: Cross-Model Comparison (Table 6-style).

Creates 1 figure with 4 subplots (Stability/Fidelity/Consistency/Sensitivity).
Data is aggregated from:
  - results/tables/table_xai_metrics_fridge.csv
  - results/tables/table_xai_metrics_printer.csv

For each method, we aggregate across all (4 models x 2 devices) combinations:
  n = 8 per method per metric.

DeepSHAP is excluded (only MLP-AE applicable).
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from plot_utils import set_time_new_roman_font

set_time_new_roman_font()



RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

METHOD_COLORS = {
    "SHAP": "#4C72B0",
    "LIME-std": "#DD8452",
    "LIME-CA": "#55A868",
    # "DeepSHAP": "#C44E52",  # excluded
}


def _load_device_tables() -> pd.DataFrame:
    df_f = pd.read_csv(os.path.join(TABLES_DIR, "table_xai_metrics_fridge.csv"))
    df_p = pd.read_csv(os.path.join(TABLES_DIR, "table_xai_metrics_printer.csv"))
    df_f["device"] = "fridge"
    df_p["device"] = "printer"
    return pd.concat([df_f, df_p], ignore_index=True)


def _aggregate_samples(df_all: pd.DataFrame, method: str, metric_base: str) -> np.ndarray:
    """Return raw per (model,device) values (n=8) for boxplot."""
    metric_mean_col = f"{metric_base}_mean"
    sub = df_all[
        (df_all["method"] == method)
        & (df_all["model"].isin(["IF", "LOF", "OCSVM", "MLP"]))
    ]
    # We want 4 models x 2 devices => 8 samples. Each row corresponds to a (device, model, method).
    vals = sub[metric_mean_col].values.astype(float)
    return vals


def fig_cross_model_comparison():
    df_all = _load_device_tables()

    # Match Table 6 metrics:
    # Stability (CV) ↓ : stability_mean lower better
    # Fidelity (MSE) ↓ : fidelity_mean lower better
    # Consistency ↑ : consistency_mean higher better
    # Sensitivity ↓ : sensitivity_mean lower better
    metric_bases = ["stability", "fidelity", "consistency", "sensitivity"]
    titles = [
        "Stability (CV, ↓ lower better)",
        "Fidelity (MSE, ↓ lower better)",
        "Consistency (cosine, ↑ higher better)",
        "Sensitivity (|score change|, ↓ lower better)",
    ]

    methods = ["SHAP", "LIME-std", "LIME-CA"]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0), sharey=False)
    axes = axes.flatten()

    for ax, metric_base, title in zip(axes, metric_bases, titles):
        # Collect per-method samples for boxplot
        data = [_aggregate_samples(df_all, m, metric_base) for m in methods]

        bp = ax.boxplot(
            data,
            patch_artist=True,
            showfliers=False,
            widths=0.6,
        )
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels(methods)

        # Color each box
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(METHOD_COLORS[m])
            patch.set_alpha(0.55)
            patch.set_edgecolor("black")

        # Color median/whiskers
        for median_line in bp["medians"]:
            median_line.set_color("black")
            median_line.set_linewidth(2)

        for whisker in bp["whiskers"]:
            whisker.set_color("black")

        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)

        # Use symlog for fidelity like existing fig_xai_metrics
        if metric_base == "fidelity":
            ax.set_yscale("symlog", linthresh=1e-3)

        # Add median labels (optional but helpful)
        for i, vals in enumerate(data, start=1):
            if len(vals) == 0:
                continue
            med = float(np.median(vals))
            ax.text(i, med, f"{med:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        "Cross-Model Comparison (4 models × 2 devices, n=8)\n"
        "Aggregated Table 6-style metrics;\nDeepSHAP excluded",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])

    out_path = os.path.join(FIG_DIR, "fig8_cross_model_comparison.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_cross_model_comparison()
    print("Saved Fig 8 to", os.path.join(FIG_DIR, "fig8_cross_model_comparison.png"))

