"""plot_xai_context_tsne_umap.py
================================
Vẽ t-SNE/UMAP để minh họa các “context points” được chọn trong

  - SHAP-CA: các background points `bg_idx` từ `explainers.context_aware_background`
  - LIME-CA: các context base points `base_indices` (context-reuse sampling)

Mục tiêu: trực quan hóa mối liên hệ giữa điểm anomalous và các điểm
bối cảnh được chọn khi tính attribution.

Output: `results/figures/fig_xai_context_{method}_{device}_{model}_tau{tau}_seed{seed}.png`
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
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
    "all": "#BDBDBD",
    "anomaly": "#D62728",
    "shapca": "#2CA02C",
    "limeca": "#FF7F0E",
}


def _fit_model(X: np.ndarray, model_name: str):
    fitted = models.build_and_fit_selected(X, model_names=[model_name])
    return fitted[model_name]


def _select_normal_and_anomalies(m, X: np.ndarray, topk: int):
    scores = m.score(X)
    normal_idx = np.where(scores <= m.threshold)[0]
    top_idx = m.top_k_indices(X, k=topk)
    return scores, normal_idx, top_idx


def _embed(method: str, X: np.ndarray, seed: int = 0):
    """Embed X into 2D using TSNE/UMAP."""
    if method == "tsne":
        from sklearn.manifold import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=min(30, max(5, X.shape[0] // 30)),
            learning_rate="auto",
            init="pca",
            random_state=seed,
        )
        return tsne.fit_transform(X)

    if method == "umap":
        try:
            import umap  # type: ignore
        except ModuleNotFoundError:
            # Fallback: if umap-learn isn't installed, use t-SNE so the script still runs.
            from sklearn.manifold import TSNE

            tsne = TSNE(
                n_components=2,
                perplexity=min(30, max(5, X.shape[0] // 30)),
                learning_rate="auto",
                init="pca",
                random_state=seed,
            )
            return tsne.fit_transform(X)

        reducer = umap.UMAP(
            n_components=2,
            random_state=seed,
            n_neighbors=min(30, max(5, X.shape[0] // 20)),
            min_dist=0.1,
            metric="euclidean",
        )
        return reducer.fit_transform(X)





    raise ValueError("method must be one of: tsne, umap")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="fridge", choices=["fridge", "printer"])
    parser.add_argument("--model", type=str, default="IF", choices=["IF", "LOF", "OCSVM", "MLP"])
    parser.add_argument("--method", type=str, default="umap", choices=["umap", "tsne"])

    parser.add_argument("--anomaly_rank", type=int, default=0, help="0 = most anomalous")
    parser.add_argument("--topk", type=int, default=20)

    parser.add_argument("--tau", type=float, default=0.5, help="cosine-sim threshold for context-aware selection")
    parser.add_argument("--k_bg", type=int, default=50, help="background size for SHAP-CA")

    parser.add_argument("--lime_num_samples", type=int, default=150, help="num_samples used inside LIME perturbations")
    parser.add_argument("--context_reuse_ratio", type=float, default=0.20)

    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    X_df = data_prep.load_device_dataframe(args.device)
    X = X_df[data_prep.FEATURE_COLUMNS].values.astype(np.float32)

    m = _fit_model(X, args.model)
    _, normal_idx, top_idx = _select_normal_and_anomalies(m, X, topk=args.topk)

    anomaly_idx = int(top_idx[args.anomaly_rank])

    # --- SHAP-CA selected context points (bg_idx)
    bg_idx = ex.context_aware_background(
        X,
        anomaly_idx=anomaly_idx,
        normal_idx=normal_idx,
        k=args.k_bg,
        tau=args.tau,
        rng=np.random.default_rng(args.seed),
    )

    # --- LIME-CA selected context base points (base_indices)
    # pipeline dùng context-reuse sampling; highlight các base points
    base_indices = ex.lime_ca_context_reuse_selected_base_indices(
        X,
        anomaly_idx=anomaly_idx,
        normal_idx=normal_idx,
        seed=args.seed,
        num_samples=args.lime_num_samples,
        context_idx=None,
        context_reuse_ratio=args.context_reuse_ratio,
    )

    base_unique = np.unique(base_indices)

    # --- Embedding
    X_emb = _embed(args.method, X, seed=args.seed)

    # --- Plot
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.scatter(X_emb[:, 0], X_emb[:, 1], s=18, c=METHOD_COLORS["all"], alpha=0.8, linewidths=0)

    # anomaly
    ax.scatter(
        X_emb[anomaly_idx, 0], X_emb[anomaly_idx, 1],
        s=140,
        c=METHOD_COLORS["anomaly"],
        marker="*",
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
        label="Anomaly point",
    )

    # SHAP-CA background
    ax.scatter(
        X_emb[bg_idx, 0], X_emb[bg_idx, 1],
        s=50,
        c=METHOD_COLORS["shapca"],
        marker="o",
        alpha=0.95,
        edgecolors="black",
        linewidths=0.6,
        zorder=4,
        label="SHAP-CA selected bg_idx",
    )

    # LIME-CA base indices
    ax.scatter(
        X_emb[base_unique, 0], X_emb[base_unique, 1],
        s=60,
        c=METHOD_COLORS["limeca"],
        marker="D",
        alpha=0.95,
        edgecolors="black",
        linewidths=0.6,
        zorder=4,
        label="LIME-CA selected base_indices (unique)",
    )

    ax.set_title(
        f"{args.method.upper()} context selection\n"
        f"device={args.device}, model={args.model}, anomaly_rank={args.anomaly_rank}\n"
        f"tau={args.tau}, k_bg={args.k_bg}, lime_num_samples={args.lime_num_samples}, context_reuse_ratio={args.context_reuse_ratio}"
    )
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    ax.legend(loc="best", fontsize=8)

    out_path = os.path.join(
        FIG_DIR,
        f"fig_xai_context_{args.method}_{args.device}_{args.model}_tau{args.tau}_seed{args.seed}.png",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Done. Wrote:", out_path)


if __name__ == "__main__":
    main()

