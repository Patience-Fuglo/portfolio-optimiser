"""
Factor Risk Model
=================

Fama-French style factor risk model for portfolio analytics.

Estimates per-asset factor loadings (betas) via OLS, then decomposes
portfolio risk into systematic (factor-driven) and idiosyncratic components.

The factor-based covariance matrix can replace the sample or Ledoit-Wolf
covariance in any optimizer, providing economically motivated risk estimates.

Model
-----
    r_i = α_i + β_i1·f_1 + β_i2·f_2 + ... + β_ik·f_k + ε_i

    Σ_factor = B × F × Bᵀ + D

Where:
    B  = loadings matrix (n_assets × n_factors)
    F  = annualised factor covariance (n_factors × n_factors)
    D  = diagonal idiosyncratic variance matrix (n_assets × n_assets)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from portfolio_optimiser.data_loader import TRADING_DAYS


@dataclass
class FactorModelResult:
    """Results from a factor model estimation."""
    loadings: pd.DataFrame           # n_assets × n_factors — factor betas
    alphas: pd.Series                # annualised per-asset intercepts
    r_squared: pd.Series             # per-asset in-sample R²
    residual_vol: pd.Series          # annualised idiosyncratic volatility
    factor_cov: npt.NDArray[np.float64]          # n_factors × n_factors (annualised)
    idio_var: npt.NDArray[np.float64]            # n_assets (annualised)
    factor_cov_matrix: npt.NDArray[np.float64]   # n_assets × n_assets full cov


def estimate_factor_model(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
) -> FactorModelResult:
    """
    Estimate a factor risk model via OLS regression.

    For each asset i:   r_i = α_i + B_i · f + ε_i

    The resulting factor-based covariance matrix Σ_f = B·F·Bᵀ + D can be
    fed directly into any optimizer as an alternative to the sample or
    Ledoit-Wolf covariance.

    Args:
        asset_returns: Daily asset returns (T × n_assets).
        factor_returns: Daily factor returns (T × n_factors).
            Expected columns: e.g. ["Mkt-RF", "SMB", "HML"].

    Returns:
        FactorModelResult with loadings, alphas, R², and covariance matrices.
    """
    idx = asset_returns.index.intersection(factor_returns.index)
    Y = asset_returns.loc[idx].values          # T × n_assets
    X_raw = factor_returns.loc[idx].values     # T × n_factors
    X = np.column_stack([np.ones(len(X_raw)), X_raw])  # add intercept

    # OLS: (XᵀX)⁻¹Xᵀy  →  coeffs shape: (1 + n_factors) × n_assets
    XtX_inv = np.linalg.inv(X.T @ X)
    coeffs = XtX_inv @ X.T @ Y

    alphas_arr = coeffs[0]        # (n_assets,)
    B = coeffs[1:].T              # n_assets × n_factors

    fitted = X @ coeffs
    residuals = Y - fitted        # T × n_assets

    ss_res = np.sum(residuals ** 2, axis=0)
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.where(ss_tot > 1e-12, ss_tot, 1.0)

    factor_cov = np.cov(X_raw, rowvar=False) * TRADING_DAYS  # n_factors × n_factors
    idio_var = np.var(residuals, axis=0, ddof=1) * TRADING_DAYS

    # Full factor-based covariance: Σ_f = B·F·Bᵀ + D
    factor_cov_matrix = B @ factor_cov @ B.T + np.diag(idio_var)

    asset_names = asset_returns.columns.tolist()
    factor_names = factor_returns.columns.tolist()

    return FactorModelResult(
        loadings=pd.DataFrame(B, index=asset_names, columns=factor_names),
        alphas=pd.Series(alphas_arr * TRADING_DAYS, index=asset_names, name="Alpha (ann.)"),
        r_squared=pd.Series(r2, index=asset_names, name="R²"),
        residual_vol=pd.Series(np.sqrt(idio_var), index=asset_names, name="Idio Vol (ann.)"),
        factor_cov=factor_cov,
        idio_var=idio_var,
        factor_cov_matrix=factor_cov_matrix,
    )


def systematic_vs_idiosyncratic(
    weights: npt.NDArray[np.float64],
    result: FactorModelResult,
) -> dict:
    """
    Decompose portfolio variance into systematic and idiosyncratic components.

    Args:
        weights: Portfolio weights (n_assets,).
        result: Output of estimate_factor_model().

    Returns:
        Dict with total_volatility, pct_systematic, pct_idiosyncratic, and raw variances.
    """
    B = result.loadings.values   # n_assets × n_factors
    F = result.factor_cov        # n_factors × n_factors
    D = np.diag(result.idio_var) # n_assets × n_assets

    sys_var = float(weights @ B @ F @ B.T @ weights)
    idio = float(weights @ D @ weights)
    total_var = sys_var + idio

    return {
        "total_variance": total_var,
        "total_volatility": float(np.sqrt(max(total_var, 0))),
        "systematic_variance": sys_var,
        "idiosyncratic_variance": idio,
        "pct_systematic": sys_var / total_var * 100 if total_var > 1e-12 else 0.0,
        "pct_idiosyncratic": idio / total_var * 100 if total_var > 1e-12 else 0.0,
    }


def print_factor_model_summary(result: FactorModelResult) -> None:
    """Print a formatted factor model summary table."""
    print("\n=== FACTOR MODEL SUMMARY ===\n")

    display = result.loadings.copy()
    display["Alpha (ann.)"] = result.alphas
    display["Idio Vol"] = result.residual_vol
    display["R²"] = result.r_squared

    print(display.round(4).to_string())

    print("\nFactor Covariance Matrix (annualised):")
    print(pd.DataFrame(
        result.factor_cov,
        index=result.loadings.columns,
        columns=result.loadings.columns,
    ).round(6).to_string())


def plot_factor_loadings(
    result: FactorModelResult,
    save_path: str = "outputs/factor_loadings.png",
) -> None:
    """
    Heatmap of factor loadings (betas) for each asset.

    Args:
        result: Output of estimate_factor_model().
        save_path: Where to save the figure.
    """
    import matplotlib.pyplot as plt

    data = result.loadings.values
    assets = result.loadings.index.tolist()
    factors = result.loadings.columns.tolist()

    fig, ax = plt.subplots(figsize=(max(6, len(factors) * 2), max(4, len(assets) * 0.9)))
    im = ax.imshow(data, cmap="RdBu_r", aspect="auto",
                   vmin=-max(abs(data.min()), abs(data.max())),
                   vmax=max(abs(data.min()), abs(data.max())))

    ax.set_xticks(range(len(factors)))
    ax.set_xticklabels(factors, fontsize=11)
    ax.set_yticks(range(len(assets)))
    ax.set_yticklabels(assets, fontsize=11)
    ax.set_title("Factor Loadings (β)", fontsize=13)
    plt.colorbar(im, ax=ax, label="β")

    for i in range(len(assets)):
        for j in range(len(factors)):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                    fontsize=9, color="black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
