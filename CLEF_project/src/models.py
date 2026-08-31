"""
models.py
=========
Point-wise anomaly detectors used in the CLEF pipeline (Section 4.3):

  - Isolation Forest (IF)
  - Local Outlier Factor (LOF), novelty=True
  - One-Class SVM (OCSVM)
  - MLP Autoencoder (MLP-AE)

All four expose a uniform interface via `AnomalyModel`:
  - `.score(X)`      -> anomaly score (higher = more anomalous), shape (n,)
  - `.threshold`     -> 95th-percentile score on the training data
  - `.top_k_indices(X, k)` -> indices of the k most anomalous points

The MLP-AE additionally exposes `.keras_model` (for SHAP/DeepSHAP) and
`.reconstruction_error_model` -- a wrapper Keras model whose single scalar
output IS the anomaly score, so that SHAP/LIME explain the *anomaly score*
directly (consistent with the other three models).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

RANDOM_STATE = 42


class AnomalyModel:
    """Common wrapper exposing `.score()` (higher == more anomalous)."""

    name: str

    def fit(self, X: np.ndarray):
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_threshold(self, X: np.ndarray, percentile: float = 95.0):
        scores = self.score(X)
        self.threshold = float(np.percentile(scores, percentile))
        return self.threshold

    def top_k_indices(self, X: np.ndarray, k: int = 20) -> np.ndarray:
        scores = self.score(X)
        return np.argsort(scores)[::-1][:k]


class IFModel(AnomalyModel):
    name = "IF"

    def __init__(self):
        self.model = IsolationForest(contamination=0.05, random_state=RANDOM_STATE)

    def fit(self, X):
        self.model.fit(X)
        self.fit_threshold(X)
        return self

    def score(self, X):
        # decision_function: higher = more normal -> flip sign
        return -self.model.decision_function(X)


class LOFModel(AnomalyModel):
    name = "LOF"

    def __init__(self):
        self.model = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)

    def fit(self, X):
        self.model.fit(X)
        self.fit_threshold(X)
        return self

    def score(self, X):
        return -self.model.decision_function(X)


class OCSVMModel(AnomalyModel):
    name = "OCSVM"

    def __init__(self):
        self.model = OneClassSVM(kernel="rbf", nu=0.05, gamma="auto")

    def fit(self, X):
        self.model.fit(X)
        self.fit_threshold(X)
        return self

    def score(self, X):
        return -self.model.decision_function(X)


class MLPAEModel(AnomalyModel):
    name = "MLP"

    def __init__(self, input_dim: int = 6, epochs: int = 100, patience: int = 10):
        self.input_dim = input_dim
        self.epochs = epochs
        self.patience = patience
        self.keras_model = None
        self.reconstruction_error_model = None

    def _build(self):
        import tensorflow as tf

        tf.random.set_seed(RANDOM_STATE)
        inp = tf.keras.Input(shape=(self.input_dim,), name="input")
        x = tf.keras.layers.Dense(16, activation="relu")(inp)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(8, activation="relu")(x)
        x = tf.keras.layers.Dense(16, activation="relu")(x)
        out = tf.keras.layers.Dense(self.input_dim, activation="linear", name="reconstruction")(x)
        ae = tf.keras.Model(inp, out, name="mlp_ae")
        ae.compile(optimizer="adam", loss="mse")

        # Wrapper model whose scalar output is the per-sample MSE
        # reconstruction error (= anomaly score). Used by SHAP/DeepSHAP/LIME
        # so all four detectors are explained on a *single scalar output*.
        sq_err = tf.keras.layers.Subtract()([inp, out])
        sq_err = tf.keras.layers.Lambda(
            lambda t: tf.reduce_mean(tf.square(t), axis=-1, keepdims=True),
            output_shape=(1,),
            name="mse",
        )(sq_err)
        err_model = tf.keras.Model(inp, sq_err, name="mlp_ae_score")

        return ae, err_model

    def fit(self, X):
        import tensorflow as tf

        np.random.seed(RANDOM_STATE)
        self.keras_model, self.reconstruction_error_model = self._build()
        es = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=self.patience, restore_best_weights=True
        )
        self.keras_model.fit(
            X, X, epochs=self.epochs, batch_size=32, verbose=0, callbacks=[es], shuffle=True
        )
        self._extract_numpy_weights()
        self.fit_threshold(X)
        return self

    def _extract_numpy_weights(self):
        """Extract trained Dense-layer weights into plain NumPy arrays so that
        `.score()` can be evaluated with a hand-rolled forward pass. This is
        purely a performance optimisation (TF `.predict()` has ~5-10ms of
        Python/graph dispatch overhead per call, which dominates runtime
        when SHAP/LIME issue thousands of small batched calls); the
        numerical computation is identical (Dense + ReLU, dropout disabled
        at inference)."""
        dense_layers = [l for l in self.keras_model.layers if l.__class__.__name__ == "Dense"]
        self._weights = [(l.get_weights()[0], l.get_weights()[1]) for l in dense_layers]
        self._activations = ["relu", "relu", "relu", "linear"]

    def _forward_numpy(self, X):
        h = np.asarray(X, dtype=np.float64)
        for (W, b), act in zip(self._weights, self._activations):
            h = h @ W + b
            if act == "relu":
                h = np.maximum(h, 0.0)
        return h

    def fit(self, X):
        import tensorflow as tf

        np.random.seed(RANDOM_STATE)
        self.keras_model, self.reconstruction_error_model = self._build()
        es = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=self.patience, restore_best_weights=True
        )
        self.keras_model.fit(
            X, X, epochs=self.epochs, batch_size=32, verbose=0, callbacks=[es], shuffle=True
        )
        self._extract_numpy_weights()
        self.fit_threshold(X)
        return self

    def score(self, X):
        X = np.asarray(X, dtype=np.float64)
        recon = self._forward_numpy(X)
        return np.mean((X - recon) ** 2, axis=-1)


MODEL_REGISTRY = {
    "IF": IFModel,
    "LOF": LOFModel,
    "OCSVM": OCSVMModel,
    "MLP": MLPAEModel,
}


def build_and_fit_selected(X: np.ndarray, model_names: list[str] | None = None) -> dict[str, AnomalyModel]:
    """Fit only a selected subset of detectors for speed.

    `model_names` refers to keys in MODEL_REGISTRY (e.g., ['IF']).
    """
    if model_names is None:
        model_names = list(MODEL_REGISTRY.keys())

    fitted: dict[str, AnomalyModel] = {}
    for name in model_names:
        if name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{name}'. Expected one of {list(MODEL_REGISTRY.keys())}")
        cls = MODEL_REGISTRY[name]
        m = cls()
        m.fit(X)
        fitted[name] = m
    return fitted


def build_and_fit_all(X: np.ndarray) -> dict[str, AnomalyModel]:
    return build_and_fit_selected(X, model_names=None)



if __name__ == "__main__":
    import data_prep

    df = data_prep.load_device_dataframe("fridge")
    X = df[data_prep.FEATURE_COLUMNS].values.astype(np.float32)
    models = build_and_fit_all(X)
    for name, m in models.items():
        s = m.score(X)
        print(name, "score range:", s.min(), s.max(), "threshold(95%):", m.threshold,
              "n_anomalies:", (s > m.threshold).sum())
