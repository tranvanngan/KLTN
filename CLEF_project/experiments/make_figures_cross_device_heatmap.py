"""Standalone script for Cross-Device Analysis heatmap (Table 7-style).

Replaces the need for 4 subplots by using a compact heatmap layout:
  rows   = metrics (4)
  columns= devices (fridge, printer, Average)

Output:
  results/figures/fig_cross_device_analysis_RI.png
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


def fig_cross_device_analysis_heatmap():
    df = pd.read_csv(os.path.join(TABLES_DIR, "table_cross_device_RI.csv"))

    dev_order = ["fridge", "printer", "Average"]
    df = df.set_index("device").reindex(dev_order).reset_index()

    metric_cols = ["stability_RI", "fidelity_RI", "consistency_RI", "sensitivity_RI"]
    metric_labels = [
        "Stability RI",
        "Fidelity RI",
        "Consistency RI",
        "Sensitivity RI",
    ]

    mat = df[metric_cols].to_numpy(dtype=float).T  # (4 metrics, 3 devices)

    fig, ax = plt.subplots(figsize=(7.4, 3.8))

    vmax = float(np.max(np.abs(mat)) + 1e-12)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_yticks(np.arange(len(metric_labels)))
    ax.set_yticklabels(metric_labels)
    ax.set_xticks(np.arange(len(dev_order)))
    ax.set_xticklabels(dev_order)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("RI (%), relative improvement vs. LIME-std → LIME-CA")

    ax.set_title("Cross-Device Analysis (RI)", fontsize=11)

    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig_cross_device_analysis_RI.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_cross_device_analysis_heatmap()
    print("Saved:", os.path.join(FIG_DIR, "fig_cross_device_analysis_RI.png"))

