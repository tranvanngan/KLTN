"""
explainers.py
=============
Implements the four explanation methods compared in the CLEF pipeline:

  - SHAP            : KernelSHAP with a *context-aware* background set
                       (Section 3.3).
  - LIME-std        : standard LIME (Gaussian perturbations, Ridge surrogate).
  - LIME-CA         : Context-Aware LIME (Section 3.4) -- perturbations are
                       accepted with probability proportional to
                       exp(sim(c_i, c(z)) / tau), where sim is cosine
                       similarity of context vectors.
  - DeepSHAP        : shap.DeepExplainer applied to the MLP-AE
                       reconstruction-error model (MLP only).
  - IntegratedGradients : Integrated Gradients (IG) for MLP-AE only.


LIME-std and LIME-CA share a common, hand-rolled implementation
(`_lime_explain`) so that the *only* algorithmic difference between them is
the perturbation-acceptance rule -- this keeps the complexity / runtime
comparison clean and reproducible (no hidden differences from relying on
two different third-party LIME configurations).

All public functions return:
    (phi, n_calls, elapsed_seconds)
where `phi` is a length-d vector of feature attributions / coefficients,
`n_calls` is the number of black-box `score_fn` evaluations performed
(used for the complexity analysis), and `elapsed_seconds` is wall-clock
time (used for the runtime / trade-off analysis).
"""

from __future__ import annotations

import time
import numpy as np
from sklearn.linear_model import Ridge

from data_prep import CONTEXT_COLUMNS, FEATURE_COLUMNS

CONTEXT_IDX = [FEATURE_COLUMNS.index(c) for c in CONTEXT_COLUMNS]


# ---------------------------------------------------------------------------
# Context-aware background selection (shared by SHAP & DeepSHAP)
# ---------------------------------------------------------------------------
def cosine_sim(c1: np.ndarray, C2: np.ndarray) -> np.ndarray:
    num = C2 @ c1
    denom = (np.linalg.norm(C2, axis=1) * (np.linalg.norm(c1) + 1e-12)) + 1e-12
    return num / denom


def _euclidean_sim(c1: np.ndarray, C2: np.ndarray, scale: float | None = None) -> np.ndarray:
    """Convert Euclidean distance to a similarity in (0,1]."""
    d = np.linalg.norm(C2 - c1[None, :], axis=1)
    if scale is None:
        # robust scale estimate to keep values stable across datasets
        scale = np.median(d) + 1e-12
    return np.exp(-d / scale)


def _manhattan_sim(c1: np.ndarray, C2: np.ndarray, scale: float | None = None) -> np.ndarray:
    """Convert Manhattan distance to a similarity in (0,1]."""
    d = np.sum(np.abs(C2 - c1[None, :]), axis=1)
    if scale is None:
        scale = np.median(d) + 1e-12
    return np.exp(-d / scale)


def _mahalanobis_sim(c1: np.ndarray, C2: np.ndarray, cov_inv: np.ndarray | None = None) -> np.ndarray:
    """Convert Mahalanobis distance to a similarity in (0,1]."""
    if cov_inv is None:
        # If no covariance is provided, fall back to identity (i.e. scaled Euclidean).
        cov_inv = np.eye(c1.shape[0], dtype=float)
    dif = C2 - c1[None, :]
    d2 = np.sum(dif @ cov_inv * dif, axis=1)
    d = np.sqrt(np.maximum(d2, 0.0))
    scale = np.median(d) + 1e-12
    return np.exp(-d / scale)


def context_similarity(
    c1: np.ndarray,
    C2: np.ndarray,
    metric: str = "cosine",
    cov_inv: np.ndarray | None = None,
) -> np.ndarray:
    """Compute context similarity between one vector c1 and many vectors C2.

    Returns higher=more similar.
    """
    metric = metric.lower()
    if metric == "cosine":
        return cosine_sim(c1, C2)
    if metric == "euclidean":
        return _euclidean_sim(c1, C2)
    if metric == "manhattan":
        return _manhattan_sim(c1, C2)
    if metric == "mahalanobis":
        return _mahalanobis_sim(c1, C2, cov_inv=cov_inv)
    raise ValueError(f"Unknown context similarity metric: {metric}")



def context_aware_background(
    X: np.ndarray,
    anomaly_idx: int,
    normal_idx: np.ndarray,
    k: int = 50,
    tau: float = 0.5,
    rng: np.random.Generator | None = None,
    context_idx: list[int] | None = None,
    context_metric: str = "cosine",
    cov_inv: np.ndarray | None = None,
) -> np.ndarray:

    """
    Select (a random subset of) up to `k` normal points whose context
    vectors are cosine-similar (>= tau) to the anomaly's context vector.
    If fewer than `k` points pass the threshold, fall back to the `k`
    most similar normal points. A different random subset each call is
    the source of run-to-run stability variation for SHAP / DeepSHAP.

    `context_idx` allows overriding which feature indices form the
    context vector (used by the ablation study, Section 5.5).
    """
    rng = rng or np.random.default_rng()
    cidx = context_idx if context_idx is not None else CONTEXT_IDX
    ci = X[anomaly_idx, cidx]
    cand = normal_idx[normal_idx != anomaly_idx]
    sims = context_similarity(ci, X[cand][:, cidx], metric=context_metric, cov_inv=cov_inv)

    above = cand[sims >= tau]

    if len(above) >= k:
        chosen = rng.choice(above, size=k, replace=False)
    else:
        order = cand[np.argsort(sims)[::-1][:k]]
        chosen = order
    return chosen


# ---------------------------------------------------------------------------
# SHAP (KernelSHAP w/ context-aware background)
# ---------------------------------------------------------------------------
def shap_explain(score_fn, X, anomaly_idx, normal_idx, k=50, tau=0.5, nsamples=64, seed=0,
                  context_idx=None, context_metric: str = "cosine", cov_inv: np.ndarray | None = None):

    import shap

    rng = np.random.default_rng(seed)
    bg_idx = context_aware_background(
        X,
        anomaly_idx,
        normal_idx,
        k=k,
        tau=tau,
        rng=rng,
        context_idx=context_idx,
        context_metric=context_metric,
        cov_inv=cov_inv,
    )

    background = X[bg_idx]

    counter = {"n": 0}

    def f(Z):
        counter["n"] += Z.shape[0]
        return score_fn(Z)

    t0 = time.perf_counter()
    explainer = shap.KernelExplainer(f, background, seed=seed)
    sv = explainer.shap_values(X[anomaly_idx : anomaly_idx + 1], nsamples=nsamples, silent=True)
    elapsed = time.perf_counter() - t0
    sv = np.asarray(sv).ravel()
    return sv, counter["n"], elapsed


# ---------------------------------------------------------------------------
# DeepSHAP (MLP-AE only)
# ---------------------------------------------------------------------------
def deepshap_explain(
    err_model,
    X,
    anomaly_idx,
    normal_idx,
    k=50,
    tau=0.5,
    seed=0,
    context_metric: str = "cosine",
    cov_inv: np.ndarray | None = None,
):

    import shap
    import tensorflow as tf

    rng = np.random.default_rng(seed)
    bg_idx = context_aware_background(
        X,
        anomaly_idx,
        normal_idx,
        k=k,
        tau=tau,
        rng=rng,
        context_metric=context_metric,
        cov_inv=cov_inv,
    )

    background = X[bg_idx].astype(np.float32)
    x = X[anomaly_idx : anomaly_idx + 1].astype(np.float32)

    t0 = time.perf_counter()
    explainer = shap.GradientExplainer(err_model, background)
    sv = explainer.shap_values(x)
    elapsed = time.perf_counter() - t0
    sv = np.asarray(sv).reshape(-1)
    n_calls = len(background) * 2  # expected-gradients: ~2 forward/backward passes per ref
    return sv, n_calls, elapsed


# ---------------------------------------------------------------------------
# LIME (standard & context-aware), shared implementation
# ---------------------------------------------------------------------------
def _lime_explain(
    score_fn,
    X,
    anomaly_idx,
    num_samples=150,
    sigma=0.1,
    kernel_width=0.75,
    context_aware=False,
    tau=0.5,
    seed=0,
    max_attempts_factor=20,
    context_idx=None,
    weight_on_context: bool = False,
    context_reuse: bool = False,
    context_reuse_ratio: float = 0.40,
    context_metric: str = "cosine",
    cov_inv: np.ndarray | None = None,
):

    rng = np.random.default_rng(seed)
    x0 = X[anomaly_idx]
    d = X.shape[1]
    n_calls = 0
    cidx = context_idx if context_idx is not None else CONTEXT_IDX

    if context_aware and context_reuse:
        # Context-reuse variant:
        # - compute similarities on context dimensions
        # - pick base points from the top-most similar normal/background points
        #   (k = context_reuse_ratio * len(X))
        # - keep context close by using those bases and only add noise to device features
        c0 = x0[cidx]
        all_sims = context_similarity(c0, X[:, cidx], metric=context_metric, cov_inv=cov_inv)

        top_k = max(1, int(len(X) * context_reuse_ratio))
        similar_indices = np.argsort(all_sims)[::-1][:top_k]


        base_indices = rng.choice(similar_indices, size=num_samples, replace=True)
        Z = X[base_indices].copy()

        device_idx = [j for j in range(d) if j not in cidx]
        if len(device_idx) > 0:
            noise = rng.normal(0, sigma * 0.5, size=(num_samples, len(device_idx)))
            Z[:, device_idx] += noise
        Z = np.clip(Z, 0.0, 1.0)

        # 1 batch black-box call for all Z (the caller later adds the original point)
        # n_calls is handled after we vstack x0.
    elif context_aware:
        c0 = x0[cidx]
        Z = np.empty((0, d))
        attempts = 0
        max_attempts = num_samples * max_attempts_factor
        while Z.shape[0] < num_samples and attempts < max_attempts:
            batch = num_samples - Z.shape[0]
            cand = x0 + rng.normal(0, sigma, size=(batch, d))
            cand = np.clip(cand, 0.0, 1.0)
            attempts += batch
            sims = cosine_sim(c0, cand[:, cidx])
            accept_p = np.exp((sims - 1.0) / tau)  # in (0,1], peaks at sim=1
            accept = rng.random(batch) < accept_p
            Z = np.vstack([Z, cand[accept]])
        if Z.shape[0] == 0:
            Z = np.clip(x0 + rng.normal(0, sigma, size=(num_samples, d)), 0, 1)
    else:
        Z = np.clip(x0 + rng.normal(0, sigma, size=(num_samples, d)), 0.0, 1.0)


    # Always include the original point itself
    Z = np.vstack([x0[None, :], Z])

    y = score_fn(Z)
    n_calls += Z.shape[0]

    dist = np.linalg.norm(Z - x0, axis=1)
    weights = np.exp(-(dist ** 2) / (kernel_width ** 2))



    ridge = Ridge(alpha=1.0)
    ridge.fit(Z, y, sample_weight=weights)
    phi = ridge.coef_.copy()
    return phi, n_calls, Z, y, weights, ridge


def lime_explain(
    score_fn,
    X,
    anomaly_idx,
    num_samples=150,
    sigma=0.1,
    seed=0,
    context_idx=None,
):
    t0 = time.perf_counter()
    phi, n_calls, *_ = _lime_explain(
        score_fn,
        X,
        anomaly_idx,
        num_samples=num_samples,
        sigma=sigma,
        context_aware=False,
        seed=seed,
        context_idx=context_idx,
        weight_on_context=False,
    )
    elapsed = time.perf_counter() - t0
    return phi, n_calls, elapsed


def lime_ca_explain(
    score_fn,
    X,
    anomaly_idx,
    num_samples=150,
    sigma=0.1,
    tau=0.5,  # kept for API compatibility; not used in context-reuse variant
    seed=0,
    context_idx=None,
    context_metric: str = "cosine",
    cov_inv: np.ndarray | None = None,
):


    """LIME-CA (now using context-reuse sampling).

    This function intentionally aliases the previous LIME-CA-CR variant so that
    the codebase exposes only one LIME-CA implementation.
    """
    return lime_ca_context_reuse_explain(
        score_fn,
        X,
        anomaly_idx,
        num_samples=num_samples,
        sigma=sigma,
        tau=tau,
        seed=seed,
        context_idx=context_idx,
        context_reuse_ratio=0.40,
        context_metric=context_metric,
        cov_inv=cov_inv,
    )




def lime_ca_context_reuse_explain(
    score_fn,
    X,
    anomaly_idx,
    num_samples=150,
    sigma=0.1,
    tau=0.5,
    seed=0,
    context_idx=None,
    context_reuse_ratio: float = 0.40,
    context_metric: str = "cosine",
    cov_inv: np.ndarray | None = None,
):

    """LIME-CA variant: replace rejection sampling with context-reuse sampling."""
    t0 = time.perf_counter()
    phi, n_calls, *_ = _lime_explain(
        score_fn,
        X,
        anomaly_idx,
        num_samples=num_samples,
        sigma=sigma,
        context_aware=True,
        tau=tau,
        seed=seed,
        context_idx=context_idx,
        weight_on_context=True,
        context_reuse=True,
        context_reuse_ratio=context_reuse_ratio,
        context_metric=context_metric,
        cov_inv=cov_inv,
    )


    elapsed = time.perf_counter() - t0
    return phi, n_calls, elapsed


def lime_ca_context_reuse_selected_base_indices(
    X: np.ndarray,
    anomaly_idx: int,
    normal_idx: np.ndarray,
    seed: int = 0,
    num_samples: int = 150,
    context_idx: list[int] | None = None,
    context_reuse_ratio: float = 0.40,
    context_metric: str = "cosine",
    cov_inv: np.ndarray | None = None,
) -> np.ndarray:

    """Return the **selected context base point indices** for LIME-CA.

    Mirrors the `context_reuse=True` branch inside `_lime_explain`:
    - compute cosine similarities on context dims between the anomaly's
      context vector and all dataset points
    - take top `top_k = max(1, int(len(X) * context_reuse_ratio))`
    - randomly sample `num_samples` points (with replacement) from the top set

    Notes:
    - Returned indices are w.r.t. the original `X` array.
    - `normal_idx` is accepted for API symmetry but the current pipeline
      computes similarities over the full dataset.
    """
    rng = np.random.default_rng(seed)
    x0 = X[anomaly_idx]

    cidx = context_idx if context_idx is not None else CONTEXT_IDX
    c0 = x0[cidx]
    all_sims = context_similarity(c0, X[:, cidx], metric=context_metric, cov_inv=cov_inv)


    top_k = max(1, int(len(X) * context_reuse_ratio))
    similar_indices = np.argsort(all_sims)[::-1][:top_k]

    base_indices = rng.choice(similar_indices, size=num_samples, replace=True)
    return base_indices




# ---------------------------------------------------------------------------
# Surrogate fidelity helper (re-fit a local linear model & evaluate MSE
# against the black-box score on the k nearest neighbours of x_i)
# ---------------------------------------------------------------------------
def integrated_gradients_explain(
    err_model,
    X: np.ndarray,
    anomaly_idx: int,
    baseline: np.ndarray | None = None,
    steps: int = 50,
    seed: int = 0,
):
    """Integrated Gradients for the MLP-AE reconstruction-error model.

    Returns:
        (phi, n_calls, elapsed_seconds)

    Notes:
        - `err_model` is expected to be a tf.keras model with a single scalar
          output anomaly score.
        - `baseline` defaults to the feature-wise median of the dataset.
    """
    from integrated_gradients_impl import integrated_gradients_explain as _impl

    if baseline is None:
        baseline = np.median(X, axis=0)

    return _impl(
        err_model,
        X,
        anomaly_idx=int(anomaly_idx),
        baseline=np.asarray(baseline, dtype=np.float64),
        steps=steps,
        seed=seed,
    )


def fidelity_mse(score_fn, X, anomaly_idx, phi, n_neighbors=50):
    """

    Method-agnostic local-fidelity metric: build the first-order Taylor
    surrogate g(z) = f(x0) + phi . (z - x0) anchored at the explained point
    x0, and report its MSE against the true black-box score f(z) over the
    `n_neighbors` nearest neighbours of x0.
    """
    x0 = X[anomaly_idx]
    f_x0 = float(score_fn(x0[None, :])[0])
    dist = np.linalg.norm(X - x0, axis=1)
    nn_idx = np.argsort(dist)[1 : n_neighbors + 1]
    Z = X[nn_idx]
    y_true = score_fn(Z)
    y_pred = f_x0 + (Z - x0) @ phi
    return float(np.mean((y_true - y_pred) ** 2))
