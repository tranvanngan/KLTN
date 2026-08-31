"""
alignment.py
============
Occupancy-Aware Alignment Score (Section 3.7).

NOTE -- methodological correction vs. the original draft:
The previous version defined F_occ as an 8-feature set (3 room
temperatures + 3 room motion sensors + 1 door sensor + hour) that did NOT
match the actual 6-D model input space (which contains only ONE
temperature feature and ONE occupancy feature). For this rewritten
pipeline we redefine F_occ to be the subset of the model's OWN 6 input
features that are directly tied to human presence / temporal behaviour:

    F_occ = {hour, occupancy}            (2 of the 6 input features)

Align(t) = sum_{j in F_occ} |phi_j(t)|  /  sum_{j=1..6} |phi_j(t)|

Anomalies are split into:
    human-driven : occupancy(t) >  median(occupancy over all 361 samples)
    device-driven: occupancy(t) <= median(occupancy over all 361 samples)
and compared with Welch's t-test.

NOTE -- threshold correction vs. the v1 draft: the new common-area motion
proxy (see data_prep.py) is a continuous activity index with a strongly
right-skewed distribution (median ~0.047, mean ~0.074, only ~0.5% of
samples > 0.5). A fixed 0.5 cut therefore yields almost no "human-driven"
points. We instead use the sample median (computed over the full 361-point
series, not just the top-k anomalies), which produces a balanced ~50/50
split and is the natural choice for a continuous, unbounded-shape signal.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from data_prep import FEATURE_COLUMNS

OCC_FEATURES = ["hour", "occupancy"]
OCC_IDX = [FEATURE_COLUMNS.index(c) for c in OCC_FEATURES]
OCC_COL_IDX = FEATURE_COLUMNS.index("occupancy")


def alignment_score(phi: np.ndarray) -> float:
    num = np.sum(np.abs(phi[OCC_IDX]))
    den = np.sum(np.abs(phi)) + 1e-12
    return float(num / den)


def alignment_analysis(X, top_idx, phis, min_per_class: int = 9, all_idx=None, occ_threshold=None):
    """
    phis: array (n_anomalies, 6) main SHAP-CA explanation vectors aligned
    with top_idx.

    `occ_threshold`: cut point for the human-driven / device-driven split.
    If None, defaults to the median occupancy value over the FULL dataset
    (X[:, OCC_COL_IDX]), which is the recommended, distribution-aware choice
    (see module docstring).

    If a class has < min_per_class members, additional points are pulled
    from `all_idx` (ranked by anomaly score, lowest to highest -- i.e. the
    "next best" anomalies) and explained on the fly by the caller BEFORE
    calling this function (phis must already include them). This function
    only performs the split + statistical test given a pre-assembled set.
    """
    if occ_threshold is None:
        occ_threshold = float(np.median(X[:, OCC_COL_IDX]))

    occ = X[top_idx, OCC_COL_IDX]
    scores = np.array([alignment_score(p) for p in phis])

    human = scores[occ > occ_threshold]
    device = scores[occ <= occ_threshold]

    if len(human) >= 2 and len(device) >= 2:
        t, p = stats.ttest_ind(human, device, equal_var=False)
    else:
        t, p = np.nan, np.nan

    return {
        "occ_threshold": occ_threshold,
        "human_mean": float(np.mean(human)) if len(human) else np.nan,
        "human_std": float(np.std(human)) if len(human) else np.nan,
        "human_n": int(len(human)),
        "device_mean": float(np.mean(device)) if len(device) else np.nan,
        "device_std": float(np.std(device)) if len(device) else np.nan,
        "device_n": int(len(device)),
        "t_stat": float(t) if not np.isnan(t) else None,
        "p_value": float(p) if not np.isnan(p) else None,
    }
