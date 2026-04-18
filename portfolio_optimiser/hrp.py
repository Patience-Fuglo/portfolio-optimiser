"""
Hierarchical Risk Parity (HRP)
==============================

López de Prado (2016) — A clustering-based portfolio construction method
that avoids matrix inversion entirely, producing more stable out-of-sample
allocations than classical mean-variance optimization.

Algorithm
---------
1. Build a correlation-based distance matrix
2. Hierarchical clustering (single linkage) groups correlated assets
3. Quasi-diagonalisation reorders assets by cluster membership
4. Recursive bisection allocates weights inversely to cluster variance

Advantages over MVO
-------------------
- No matrix inversion → immune to ill-conditioning
- Naturally diversifies within and across clusters
- More robust out-of-sample than Markowitz with estimated inputs
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import squareform


def _corr_to_dist(corr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Correlation → distance: d_ij = sqrt(0.5 * (1 − ρ_ij))."""
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    return dist


def _ivp_weights(cov_sub: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Inverse-variance portfolio weights using diagonal of sub-covariance."""
    diag_var = np.diag(cov_sub)
    diag_var = np.where(diag_var > 1e-12, diag_var, 1e-12)
    ivp = 1.0 / diag_var
    return ivp / ivp.sum()


def _cluster_var(cov: npt.NDArray[np.float64], items: list[int]) -> float:
    """Variance of a cluster using inverse-variance weights (López de Prado §16.4)."""
    sub = cov[np.ix_(items, items)]
    w = _ivp_weights(sub)
    return float(w @ sub @ w)


def hrp_weights(
    cov_matrix: npt.NDArray[np.float64],
    corr_matrix: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """
    Hierarchical Risk Parity portfolio weights (López de Prado 2016).

    Args:
        cov_matrix: Annualized covariance matrix (n × n).
        corr_matrix: Correlation matrix (n × n). Derived from cov_matrix
            when not provided.

    Returns:
        HRP weights (n,) summing to 1.
    """
    n = cov_matrix.shape[0]

    if corr_matrix is None:
        vol = np.sqrt(np.maximum(np.diag(cov_matrix), 1e-12))
        corr_matrix = cov_matrix / np.outer(vol, vol)
        np.clip(corr_matrix, -1.0, 1.0, out=corr_matrix)
        np.fill_diagonal(corr_matrix, 1.0)

    # Step 1 — distance matrix
    dist = _corr_to_dist(corr_matrix)

    # Step 2 — single-linkage hierarchical clustering
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")

    # Step 3 — quasi-diagonalisation: sort assets by cluster leaf order
    sort_ix = list(leaves_list(link))

    # Step 4 — recursive bisection
    weights = np.ones(n)
    clusters: list[list[int]] = [sort_ix]

    while clusters:
        next_clusters: list[list[int]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left, right = cluster[:mid], cluster[mid:]

            v_left = _cluster_var(cov_matrix, left)
            v_right = _cluster_var(cov_matrix, right)
            total = v_left + v_right
            alpha = v_right / total if total > 1e-12 else 0.5

            for idx in left:
                weights[idx] *= alpha
            for idx in right:
                weights[idx] *= (1.0 - alpha)

            if len(left) > 1:
                next_clusters.append(left)
            if len(right) > 1:
                next_clusters.append(right)

        clusters = next_clusters

    return weights / weights.sum()


def plot_hrp_dendrogram(
    cov_matrix: npt.NDArray[np.float64],
    asset_names: list[str],
    corr_matrix: npt.NDArray[np.float64] | None = None,
    save_path: str = "outputs/hrp_dendrogram.png",
) -> None:
    """
    Plot the hierarchical clustering dendrogram underlying HRP.

    Args:
        cov_matrix: Annualized covariance matrix.
        asset_names: Asset ticker labels.
        corr_matrix: Optional correlation matrix.
        save_path: Where to save the figure.
    """
    import matplotlib.pyplot as plt

    if corr_matrix is None:
        vol = np.sqrt(np.maximum(np.diag(cov_matrix), 1e-12))
        corr_matrix = cov_matrix / np.outer(vol, vol)
        np.clip(corr_matrix, -1.0, 1.0, out=corr_matrix)
        np.fill_diagonal(corr_matrix, 1.0)

    dist = _corr_to_dist(corr_matrix)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")

    plt.figure(figsize=(10, 5))
    dendrogram(link, labels=asset_names, leaf_rotation=45, leaf_font_size=11)
    plt.title("HRP — Hierarchical Clustering Dendrogram")
    plt.xlabel("Asset")
    plt.ylabel("Distance  √(0.5·(1−ρ))")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
