"""stats_tests.py

Statistical tests for comparing multiple XAI methods using repeated measures.

Implements:
  - Friedman test (non-parametric repeated measures ANOVA)
  - Nemenyi post-hoc test based on average ranks

These utilities are tailored for the CLEF main pipeline output
(results/tables/main_results_raw.csv).

No external dependency besides SciPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import friedmanchisquare, studentized_range, wilcoxon

from statsmodels.stats.multitest import multipletests

import pandas as pd




@dataclass
class FriedmanResult:
    statistic: float
    pvalue: float


@dataclass
class NemenyiPairwiseResult:
    method_i: str
    method_j: str
    rank_diff: float
    q_stat: float
    pvalue: float
    critical_diff: float
    reject: bool



def _avg_ranks(matrix: np.ndarray) -> np.ndarray:
    """Compute average ranks across blocks.

    matrix shape: (n_blocks, n_methods)
    Lower values mean better? This is controlled by caller via transformation.

    Ranks are 1..k where 1 is the best (lowest value).
    """
    n_blocks, n_methods = matrix.shape
    ranks = np.empty_like(matrix, dtype=float)
    # rank within each block (row)
    for b in range(n_blocks):
        ranks[b] = np.argsort(np.argsort(matrix[b])) + 1
    return ranks.mean(axis=0)


def friedman_nemenyi(
    data: np.ndarray,
    *,
    methods: Sequence[str],
    alpha: float = 0.05,
) -> tuple[FriedmanResult, list[NemenyiPairwiseResult], np.ndarray]:

    """Run Friedman + Nemenyi.

    Parameters
    ----------
    data:
        Shape (n_blocks, n_methods). Each row is a block (e.g., one (device, model)).
        Each column is a method.
        NOTE: Caller must already transform so that "smaller is better".
    methods:
        Names for columns in `data`.
    alpha:
        Significance level.

    Returns
    -------
    friedman_result:
    nemenyi_pairs:
    avg_ranks:
        Average ranks (lower is better).
    """

    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be 2D (n_blocks, n_methods)")
    n_blocks, n_methods = data.shape
    if len(methods) != n_methods:
        raise ValueError("len(methods) must match data.shape[1]")
    if n_blocks < 2:
        raise ValueError("Need at least 2 blocks for Friedman test")

    # Friedman expects separate arrays per method/column.
    # Use nan_policy by filtering rows that have any nan.
    row_mask = ~np.any(np.isnan(data), axis=1)
    data2 = data[row_mask]
    if data2.shape[0] < 2:
        raise ValueError("Not enough non-NaN blocks after filtering")

    cols = [data2[:, j] for j in range(n_methods)]
    stat, p = friedmanchisquare(*cols)
    fried = FriedmanResult(statistic=float(stat), pvalue=float(p))

    avg_r = _avg_ranks(data2)

    # Nemenyi uses Studentized range distribution.
    # q_{alpha} derived from studentized_range distribution.
    # p-value for each pair can be computed using the studentized range CDF.
    nemenyi_pairs: list[NemenyiPairwiseResult] = []
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            rank_diff = abs(avg_r[i] - avg_r[j])
            # q_stat = rank_diff / sqrt(k*(k+1)/(6N)) where N=n_blocks, k=n_methods
            k = n_methods
            N = data2.shape[0]
            denom = np.sqrt(k * (k + 1) / (6.0 * N))
            q_stat = float(rank_diff / denom) if denom > 0 else np.inf

            # Nemenyi pairwise significance:
            # Compute p-value using the studentized range distribution.
            # For Nemenyi, the test statistic uses q_stat = |Ri - Rj| / sqrt(k(k+1)/(6N)).
            # Using SciPy: studentized_range.cdf(q, k, df) where df = N-1.
            # We use a two-sided p-value based on |q| -> p = 1 - CDF(q).
            # (This matches common Nemenyi implementations where critical difference
            # uses studentized_range.ppf(1-alpha, k, df).)
            df_denom = N - 1
            p_pair = float(1.0 - studentized_range.cdf(q_stat, k, df_denom))
            p_pair = max(0.0, min(1.0, p_pair))

            # Critical Difference (CD) variant: reject if |Ri - Rj| >= CD.
            # CD = q_alpha * sqrt(k*(k+1)/(6N))
            q_alpha = float(studentized_range.ppf(1.0 - alpha, k, df_denom))
            cd = float(q_alpha * np.sqrt(k * (k + 1) / (6.0 * N)))

            reject = bool(rank_diff >= cd)

            # store nemenyi results

            nemenyi_pairs.append(
                NemenyiPairwiseResult(
                    method_i=methods[i],
                    method_j=methods[j],
                    rank_diff=float(rank_diff),
                    q_stat=q_stat,
                    pvalue=p_pair,
                    critical_diff=float(cd),
                    reject=bool(reject),
                )
            )

    # Keep for downstream: caller can decide significance by alpha.
    return fried, nemenyi_pairs, avg_r


def _wilcoxon_holm_pairwise(
    data_by_block: dict[tuple[str, str], dict[str, np.ndarray]],
    *,
    methods: Sequence[str],
    metric_name: str,
    alpha: float,
    blocks: Sequence[tuple[str, str]],
) -> list[dict]:
    """Paired Wilcoxon signed-rank + Holm correction over pairwise p-values.

    data_by_block[(device,model)][method] -> 1D array of per-run values for this method.
    blocks are (device,model); here each block has length RUNS samples.

    We perform Wilcoxon on the per-run values inside each block, producing a single p-value per pair.
    Then Holm correction is applied across the 3 pairs for this metric.
    """

    # Collect pairwise p-values
    pairs = []
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            pairs.append((methods[i], methods[j]))

    pvals = []
    pairwise_tmp = []

    for mi, mj in pairs:
        p_block = []
        for b in blocks:
            vi = data_by_block[b][mi]
            vj = data_by_block[b][mj]

            # Paired Wilcoxon within block on vectors.
            # Using zero_method='wilcox' default; set alternative='two-sided'.
            # If all differences are zero, scipy can return p=1.0.
            try:
                stat, p = wilcoxon(vi, vj)
            except ValueError:
                p = 1.0
            p_block.append(p)

        # Combine block-level p-values conservatively by taking median p.
        # This is a pragmatic choice because we don't have a clear repeated-measures
        # Wilcoxon aggregation standard.
        # (We can change this to Fisher's method if you prefer.)
        p_med = float(np.median(p_block))
        pvals.append(p_med)
        pairwise_tmp.append((mi, mj, p_med, p_block))

    # Holm correction across pairwise tests
    reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="holm")

    out = []
    for (mi, mj, p_med, p_block), p, padj, rej in zip(pairwise_tmp, pvals, p_adj, reject):
        out.append({
            "metric": metric_name,
            "method_i": mi,
            "method_j": mj,
            "wilcoxon_pvalue": float(p_med),
            "holm_pvalue": float(padj),
            "holmn_reject": bool(rej),
            "alpha": float(alpha),
        })

    return out


def wilcoxon_holm_main(
    df_all: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute Wilcoxon + Holm pairwise for main pipeline.

    Uses blocks=(device,model). For each metric, each method must have columns:
      stability_run0..RUNS-1, fidelity_run0.., consistency_run0.., sensitivity_run0..

    Returns a long dataframe with pairwise results.
    """

    methods = ["SHAP", "LIME-std", "LIME-CA"]
    metrics_cfg = [
        ("stability", "stability_mean", False),
        ("fidelity", "fidelity_mean", False),
        ("consistency", "consistency_mean", True),
        ("sensitivity", "sensitivity_mean", False),
    ]

    blocks = (
        df_all[["device", "model"]]
        .drop_duplicates()
        .apply(lambda r: (r["device"], r["model"]), axis=1)
        .tolist()
    )

    # Determine RUNS from columns (stability_runX)
    # Assume stability_run0 exists.
    run0_col = "stability_run0"
    if run0_col not in df_all.columns:
        raise ValueError(
            "Missing per-run columns. Re-run run_experiments.py so main_results_raw.csv exports *_runN columns."
        )

    # Find max run index from any method
    run_idxs = []
    for c in df_all.columns:
        if c.startswith("stability_run"):
            try:
                run_idxs.append(int(c.split("run", 1)[1]))
            except Exception:
                pass
    runs = max(run_idxs) + 1 if run_idxs else 0

    data_by_block: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for b in blocks:
        device, model = b
        data_by_block[b] = {}
        for m in methods:
            sub = df_all[(df_all["device"] == device) & (df_all["model"] == model) & (df_all["method"] == m)]
            if len(sub) != 1:
                # If duplicates remain, take mean across duplicates.
                sub = sub.groupby(["device", "model", "method"], as_index=False).mean(numeric_only=True)

            sub_row = sub.iloc[0]
            # Extract run vectors for each metric inside loop later

    pair_rows = []
    for metric_base, metric_name_col, higher_is_better in metrics_cfg:
        # higher_is_better: if True, make smaller better by negating
        for b in blocks:
            device, model = b
            if b not in data_by_block:
                data_by_block[b] = {}
            for m in methods:
                sub = df_all[(df_all["device"] == device) & (df_all["model"] == model) & (df_all["method"] == m)]
                if len(sub) == 0:
                    continue
                if len(sub) > 1:
                    sub = sub.groupby(["device", "model", "method"], as_index=False).mean(numeric_only=True)
                row = sub.iloc[0]
                vals = []
                for r_i in range(runs):
                    col = f"{metric_base}_run{r_i}"
                    vals.append(float(row[col]))
                vals = np.array(vals, dtype=float)
                if higher_is_better:
                    vals = -vals
                data_by_block[b][m] = vals

        pair_results = _wilcoxon_holm_pairwise(
            data_by_block,
            methods=methods,
            metric_name=metric_base,
            alpha=alpha,
            blocks=blocks,
        )
        pair_rows.extend(pair_results)

    out = pd.DataFrame(pair_rows)
    return out


def build_friedman_matrix(

    df,
    *,
    blocks: Iterable[str],
    methods: Sequence[str],
    metric_col: str,
    higher_is_better: bool,
    block_cols: tuple[str, str] = ("device", "model"),
) -> np.ndarray:
    """Create (n_blocks, n_methods) matrix from df.

    blocks are identifiers; but we use (device, model) columns.

    higher_is_better controls transformation to "smaller is better".
    """
    device_col, model_col = block_cols


    block_ids = list(blocks)
    idx = {bid: t for bid, t in enumerate(block_ids)}

    # Map block_id -> (device, model)
    # Here bid format is f"{device}||{model}".
    mat = np.empty((len(block_ids), len(methods)), dtype=float)
    mat[:] = np.nan

    for r, bid in enumerate(block_ids):
        device, model = bid.split("||", 1)
        for c, method in enumerate(methods):
            val = df[
                (df[device_col] == device)
                & (df[model_col] == model)
                & (df["method"] == method)
            ][metric_col]
            if len(val):
                v = float(val.values[0])
                # Transform to make smaller = better.
                mat[r, c] = -v if higher_is_better else v

    return mat

