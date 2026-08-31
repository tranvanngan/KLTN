"""
make_figures.py
================
Generates all figures for the revised paper (results/figures/*.png).
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

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

METHOD_COLORS = {
    "SHAP": "#4C72B0",
    "LIME-std": "#DD8452",
    "LIME-CA": "#55A868",
    "DeepSHAP": "#C44E52",
}


# ---------------------------------------------------------------------------
# Fig 1: Runtime comparison (bar, log scale)
# ---------------------------------------------------------------------------
def fig_runtime_comparison():
    df = pd.read_csv(os.path.join(TABLES_DIR, "table_runtime_complexity.csv"))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    methods = df["method"].tolist()
    means = df["runtime_ms_mean"].values
    stds = df["runtime_ms_std"].values
    colors = [METHOD_COLORS[m] if m in METHOD_COLORS else "#888888" for m in methods]

    bars = ax.bar(methods, means, yerr=stds, capsize=4, color=colors, edgecolor="white")
    ax.set_yscale("log")
    ax.set_ylabel("Mean runtime per explanation (ms, log scale)")
    ax.set_title("Computational cost per explanation\n(averaged over 4 models x 2 devices)")
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.1f} ms",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_runtime_comparison.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2: Empirical complexity scaling (2x2 small multiples)
# ---------------------------------------------------------------------------
def fig_complexity_scaling():
    df = pd.read_csv(os.path.join(TABLES_DIR, "table_complexity_scaling.csv"))
    methods = ["SHAP", "LIME-std", "LIME-CA", "DeepSHAP"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), sharey=False)
    for ax, method in zip(axes, methods):
        sub = df[df.method == method]
        ax.errorbar(sub.param_value, sub.runtime_ms_mean, yerr=sub.runtime_ms_std,
                     marker="o", color=METHOD_COLORS[method], capsize=3)
        ax.set_title(method)
        ax.set_xlabel(sub.param_name.iloc[0])
        if ax is axes[0]:
            ax.set_ylabel("Runtime (ms)")
    fig.suptitle("Empirical runtime scaling vs. method-specific complexity parameter\n"
                  "(Fridge, IsolationForest for SHAP/LIME; MLP-AE for DeepSHAP)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_complexity_scaling.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3: XAI quality metrics, grouped bars (per device, per metric)
# ---------------------------------------------------------------------------
def fig_xai_metrics():
    metrics = ["stability_mean", "fidelity_mean", "consistency_mean", "sensitivity_mean"]
    titles = ["Stability (CV, lower=better)", "Fidelity (MSE, lower=better)",
              "Consistency (cosine, higher=better)", "Sensitivity (|score change|)"]

    for device in ["fridge", "printer"]:
        df = pd.read_csv(os.path.join(TABLES_DIR, f"table_xai_metrics_{device}.csv"))
        models_ = df["model"].unique().tolist()
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
        for ax, metric, title in zip(axes, metrics, titles):
            std_metric = metric.replace("_mean", "_std")
            x = np.arange(len(models_))
            width = 0.25
            for i, method in enumerate(["SHAP", "LIME-std", "LIME-CA"]):
                sub = df[df.method == method].set_index("model").reindex(models_)
                ax.bar(x + (i - 1) * width, sub[metric], width, yerr=sub[std_metric],
                       label=method, color=METHOD_COLORS[method], capsize=2)
            # DeepSHAP (MLP only) as a separate marker
            mlp_idx = models_.index("MLP")
            deep = df[(df.method == "DeepSHAP")]
            if len(deep):
                ax.scatter([mlp_idx + 1.5 * width], deep[metric].values,
                           color=METHOD_COLORS["DeepSHAP"], marker="*", s=120,
                           label="DeepSHAP", zorder=5)
            ax.set_xticks(x)
            ax.set_xticklabels(models_)
            ax.set_title(title, fontsize=9.5)
            if metric == "fidelity_mean":
                ax.set_yscale("symlog", linthresh=1e-3)
        axes[0].legend(fontsize=8, loc="upper left")
        fig.suptitle(f"Explanation-quality metrics -- {device.capitalize()} "
                      f"(mean +/- std over 10 runs, top-20 anomalies)")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"fig3_xai_metrics_{device}.png"))
        plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 4: Alignment score - human-driven vs device-driven
# ---------------------------------------------------------------------------
def fig_alignment():
    df = pd.read_csv(os.path.join(TABLES_DIR, "table_alignment.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, device in zip(axes, ["fridge", "printer"]):
        sub = df[df.device == device]
        x = np.arange(len(sub))
        width = 0.35
        ax.bar(x - width / 2, sub.human_mean, width, yerr=sub.human_std,
               label="Human-driven (occupancy > median)", color="#55A868", capsize=4)
        ax.bar(x + width / 2, sub.device_mean, width, yerr=sub.device_std,
               label="Device-driven (occupancy <= median)", color="#4C72B0", capsize=4)
        for i, row in enumerate(sub.itertuples()):
            sig = "*" if row.p_value < 0.05 else "ns"
            ymax = max(row.human_mean + row.human_std, row.device_mean + row.device_std)
            ax.text(i, ymax + 0.02, f"p={row.p_value:.3f} ({sig})", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["model"])
        ax.set_title(device.capitalize())
        ax.set_ylim(0, 0.85)
    axes[0].set_ylabel("Alignment Score  (sum |phi_hour,occ| / sum |phi|)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Occupancy-aware Alignment Score: human-driven vs. device-driven anomalies\n"
                  "(SHAP context-aware explanations, Welch's t-test)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_alignment.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 5: Tau sensitivity
# ---------------------------------------------------------------------------
def fig_tau_sensitivity():
    df = pd.read_csv(os.path.join(TABLES_DIR, "table_tau_sensitivity.csv"))
    fig, ax1 = plt.subplots(figsize=(5.5, 4))
    ax2 = ax1.twinx()
    ax1.plot(df.tau, df.consistency, "o-", color="#55A868", label="Consistency")
    ax2.plot(df.tau, df.fidelity_mse, "s--", color="#C44E52", label="Fidelity (MSE)")
    ax1.set_xlabel(r"$\tau$ (context-similarity threshold)")
    ax1.set_ylabel("Consistency (cosine)", color="#55A868")
    ax2.set_ylabel("Fidelity MSE", color="#C44E52")
    ax1.tick_params(axis="y", labelcolor="#55A868")
    ax2.tick_params(axis="y", labelcolor="#C44E52")
    ax1.set_title(r"SHAP-CA sensitivity to $\tau$ (Fridge, Isolation Forest)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_tau_sensitivity.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 6: Ablation study (heatmap version)
# ---------------------------------------------------------------------------
def fig_ablation():
    df = pd.read_csv(os.path.join(TABLES_DIR, "table_ablation.csv"))

    # Heatmap replacement for the previous 2-panel barh plots.
    # Rows: metrics (consistency, fidelity_mse), Columns: ablation configurations.
    values = df[["consistency", "fidelity_mse"]].T.values.astype(float)
    metric_labels = ["Consistency (cosine)", "Fidelity (MSE)"]
    config_labels = df["configuration"].tolist()

    fig, ax = plt.subplots(1, 1, figsize=(9.0, 3.8))

    # Use a more print-friendly colormap (higher contrast, less “dark blue”).
    # Use a high-contrast colormap; keep it simple & print-friendly.
    im = ax.imshow(values, aspect="auto", cmap="cividis")
    ax.set_xticks(range(len(config_labels)))
    ax.set_xticklabels(config_labels, rotation=20, ha="right")
    ax.set_yticks(range(len(metric_labels)))
    ax.set_yticklabels(metric_labels)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            # Improve readability: use white text on dark cells, black on light cells.
            # Threshold is based on colormap normalization (imshow uses 0..max by default here).
            txt_color = "white" if v < 0.5 * values.max() else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color=txt_color)

    # Use a single color group for print-friendliness: no diverging/secondary hues.
    # Keep the colorbar but it's now tied to a single sequential cmap.
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Metric value")


    ax.set_title("Ablation: which context features matter for LIME-CA?\n(Fridge, Isolation Forest)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig6_ablation.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 7: Trade-off scatter (runtime vs stability), per method
# ---------------------------------------------------------------------------
def fig_tradeoff():
    df = pd.read_csv(os.path.join(TABLES_DIR, "main_results_raw.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, ycol, ylabel in zip(axes, ["stability_mean", "fidelity_mean"],
                                  ["Stability (CV, lower=better)", "Fidelity (MSE, lower=better)"]):
        for method, color in METHOD_COLORS.items():
            sub = df[df.method == method]
            if len(sub) == 0:
                continue
            ax.scatter(sub.runtime_mean * 1000, sub[ycol], label=method, color=color, s=60, alpha=0.8)

        ax.set_xscale("log")
        if ycol == "fidelity_mean":
            ax.set_yscale("log")
        ax.set_xlabel("Runtime per explanation (ms, log scale)")
        ax.set_ylabel(ylabel)
    axes[0].legend(fontsize=8)
    fig.suptitle("Quality vs. computational-cost trade-off\n"
                  "(each point = one model x device combination)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig7_tradeoff.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 8 (new): Cross-device RI analysis (Table 7-style)
# ---------------------------------------------------------------------------
def fig_cross_device_analysis():
    """Visualize cross-device RI in a compact heatmap.


    Source: results/tables/table_cross_device_RI.csv
    Output: results/figures/fig_cross_device_analysis_RI.png

    RI columns are precomputed in run_experiments.py (LIME-std vs LIME-CA).
    """

    df = pd.read_csv(os.path.join(TABLES_DIR, "table_cross_device_RI.csv"))

    dev_order = ["fridge", "printer", "Average"]
    df = df.set_index("device").reindex(dev_order).reset_index()

    metrics = [
        ("stability_RI", "Stability RI (↓ lower better)"),
        ("fidelity_RI", "Fidelity RI (↓ lower better)"),
        ("consistency_RI", "Consistency RI (↑ higher better)"),
        ("sensitivity_RI", "Sensitivity RI (↓ lower better)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharey=False)
    axes = axes.flatten()

    bar_color = "#4C72B0"
    for ax, (col, title) in zip(axes, metrics):
        x = np.arange(len(df))
        vals = df[col].values.astype(float)

        ax.bar(x, vals, color=bar_color, edgecolor="white", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(df["device"].values)
        ax.set_title(title, fontsize=10)
        ax.axhline(0, color="black", linewidth=1)

        # value labels
        max_abs = float(np.max(np.abs(vals)) + 1e-12)
        y_pad = 0.02 * max_abs
        for i, v in enumerate(vals):
            ax.text(i, v + np.sign(v) * y_pad, f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        "Cross-Device Analysis: Relative Improvement (RI)\n"
        "(Table 7-style; computed from LIME-std vs LIME-CA)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    out_path = os.path.join(FIG_DIR, "fig_cross_device_analysis_RI.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_runtime_comparison()
    fig_complexity_scaling()
    fig_xai_metrics()
    fig_alignment()
    fig_tau_sensitivity()
    fig_ablation()
    fig_tradeoff()
    fig_cross_device_analysis()
    print("All figures written to", FIG_DIR)
    print(os.listdir(FIG_DIR))

