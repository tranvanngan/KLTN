# Helper for Integrated Gradients.
# (Kept separate to avoid bloating explainers.py further if you want to extend later.)

from __future__ import annotations

import time
import numpy as np


def integrated_gradients(
    model,
    x: np.ndarray,
    baseline: np.ndarray,
    steps: int = 50,
):
    """Compute Integrated Gradients for a single input.

    Args:
        model: tf.keras.Model mapping (batch,d) -> (batch,1) or (batch,)
        x: (d,) numpy array
        baseline: (d,) numpy array
        steps: number of integration steps

    Returns:
        attr: (d,) numpy array
    """
    import tensorflow as tf

    x_tf = tf.convert_to_tensor(x[None, :], dtype=tf.float32)
    b_tf = tf.convert_to_tensor(baseline[None, :], dtype=tf.float32)

    # Scale inputs from baseline to x.
    alphas = tf.linspace(0.0, 1.0, steps + 1)

    grads = []
    for a in alphas:
        xi = b_tf + a * (x_tf - b_tf)
        with tf.GradientTape() as tape:
            tape.watch(xi)
            yi = model(xi)
            # Ensure scalar per batch
            yi = tf.reshape(yi, (1,))
        gi = tape.gradient(yi, xi)
        grads.append(gi[0])

    grads = tf.stack(grads, axis=0)  # (steps+1, d)
    avg_grads = (grads[:-1] + grads[1:]) / 2.0  # trapezoidal rule
    int_grads = tf.reduce_mean(avg_grads, axis=0)  # (d,)

    attr = (x_tf[0] - b_tf[0]) * int_grads
    return attr.numpy().astype(np.float64)


def integrated_gradients_explain(
    score_model,
    X: np.ndarray,
    anomaly_idx: int,
    baseline: np.ndarray,
    steps: int = 50,
    seed: int = 0,
):
    """Pipeline-compatible wrapper returning (phi, n_calls, elapsed)."""
    import tensorflow as tf

    rng = np.random.default_rng(seed)

    x0 = np.asarray(X[anomaly_idx], dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    assert x0.shape == b.shape

    # Fix TF determinism best-effort.
    tf.random.set_seed(int(seed))

    t0 = time.perf_counter()
    # Use TF model directly to compute scalar gradient wrt inputs.
    attr = integrated_gradients(score_model, x0, b, steps=steps)
    elapsed = time.perf_counter() - t0

    # Black-box eval count approximation: one forward+backward per step.
    n_calls = (steps + 1)
    return attr, n_calls, elapsed

