"""
Report Module
=============

Risk attribution, tail-risk analysis, and portfolio visualization.

Functions
---------
- risk_contribution       : Marginal risk contribution per asset
- pct_risk_contribution   : Risk contributions as % of portfolio volatility
- diversification_ratio   : DR = weighted-avg vol / portfolio vol
- cvar                    : Conditional Value at Risk (Expected Shortfall)
- print_portfolio_summary : Full risk table including CVaR
- plot_risk_pie           : Pie chart of % risk contributions
- plot_correlation_heatmap: Annotated correlation heatmap
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from portfolio_optimiser.data_loader import TRADING_DAYS
from portfolio_optimiser.optimizer import portfolio_volatility, portfolio_return


def risk_contribution(
    weights: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Marginal risk contribution of each asset.

    RC_i = w_i * (Σ w)_i / σ_p

    Args:
        weights: Portfolio weights (n,).
        cov_matrix: Annualized covariance matrix (n × n).

    Returns:
        Array of risk contributions; sums to portfolio volatility.
    """
    portfolio_vol = portfolio_volatility(weights, cov_matrix)

    if portfolio_vol < 1e-10:
        return np.zeros(len(weights))

    marginal_contrib = np.dot(cov_matrix, weights)
    return weights * marginal_contrib / portfolio_vol


def pct_risk_contribution(
    risk_contributions: npt.NDArray[np.float64],
    portfolio_vol: float,
) -> npt.NDArray[np.float64]:
    """
    Convert risk contributions into percentage of total portfolio risk.

    Args:
        risk_contributions: Array from risk_contribution().
        portfolio_vol: Portfolio volatility scalar.

    Returns:
        Array of percentage contributions; sums to ~100%.
    """
    if portfolio_vol < 1e-10:
        return np.zeros(len(risk_contributions))

    return (risk_contributions / portfolio_vol) * 100


def diversification_ratio(
    weights: npt.NDArray[np.float64],
    individual_vols: npt.NDArray[np.float64],
    portfolio_vol: float,
) -> float:
    """
    Diversification Ratio = Σ(w_i * σ_i) / σ_p.

    DR > 1 indicates diversification is reducing portfolio risk.

    Args:
        weights: Portfolio weights.
        individual_vols: Individual asset volatilities (sqrt of diag(Σ)).
        portfolio_vol: Portfolio volatility.

    Returns:
        Diversification ratio (scalar).
    """
    if portfolio_vol < 1e-10:
        return 0.0

    return float(np.sum(weights * individual_vols) / portfolio_vol)


def cvar(
    daily_returns: npt.NDArray[np.float64],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Conditional Value at Risk (Expected Shortfall).

    CVaR at α% is the expected loss on the worst (1-α)% of days.
    More sensitive to tail risk than VaR; required under Basel III.

    Args:
        daily_returns: Array of daily portfolio returns.
        confidence: Confidence level (default 0.95 = 95%).

    Returns:
        Tuple of (daily_cvar, annualized_cvar) expressed as positive loss values.
    """
    sorted_r = np.sort(daily_returns)
    cutoff = max(1, int(len(sorted_r) * (1.0 - confidence)))
    tail = sorted_r[:cutoff]
    daily_es = float(-np.mean(tail))
    annualized_es = daily_es * np.sqrt(TRADING_DAYS)
    return daily_es, annualized_es


def print_portfolio_summary(
    weights: npt.NDArray[np.float64],
    expected_returns: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    asset_names: list[str],
    daily_returns: npt.NDArray[np.float64] | None = None,
) -> dict:
    """
    Print a full portfolio risk report and return key metrics.

    Args:
        weights: Portfolio weights.
        expected_returns: Annualized expected returns per asset.
        cov_matrix: Annualized covariance matrix.
        asset_names: Asset ticker labels.
        daily_returns: Optional array of historical daily portfolio returns
            used to compute CVaR.  If None, CVaR is omitted.

    Returns:
        Dict with portfolio_return, portfolio_volatility, risk_contributions,
        pct_risk_contributions, individual_vols, diversification_ratio,
        and (if daily_returns provided) daily_cvar and annualized_cvar.
    """
    portfolio_vol = portfolio_volatility(weights, cov_matrix)
    portfolio_ret = portfolio_return(weights, expected_returns)
    individual_vols = np.sqrt(np.diag(cov_matrix))

    rc = risk_contribution(weights, cov_matrix)
    pct_rc = pct_risk_contribution(rc, portfolio_vol)
    div_ratio = diversification_ratio(weights, individual_vols, portfolio_vol)

    print("\n=== PORTFOLIO RISK REPORT ===\n")
    print(f"{'Asset':<10} {'Weight':>10} {'Exp Return':>12} {'Volatility':>12} {'Risk %':>12}")
    print("-" * 60)

    for i, asset in enumerate(asset_names):
        print(
            f"{asset:<10} "
            f"{weights[i]:>10.4f} "
            f"{expected_returns[i]:>12.4f} "
            f"{individual_vols[i]:>12.4f} "
            f"{pct_rc[i]:>11.2f}%"
        )

    print("-" * 60)
    print(
        f"{'PORTFOLIO':<10} "
        f"{np.sum(weights):>10.4f} "
        f"{portfolio_ret:>12.4f} "
        f"{portfolio_vol:>12.4f} "
        f"{np.sum(pct_rc):>11.2f}%"
    )
    print(f"\nDiversification Ratio: {div_ratio:.4f}")

    result: dict = {
        "portfolio_return": portfolio_ret,
        "portfolio_volatility": portfolio_vol,
        "risk_contributions": rc,
        "pct_risk_contributions": pct_rc,
        "individual_vols": individual_vols,
        "diversification_ratio": div_ratio,
    }

    if daily_returns is not None:
        daily_cvar, ann_cvar = cvar(np.asarray(daily_returns))
        print(f"CVaR (95%, annualised):  {ann_cvar:.4f}  ({ann_cvar * 100:.2f}%)")
        result["daily_cvar"] = daily_cvar
        result["annualized_cvar"] = ann_cvar

    return result


def plot_risk_pie(
    asset_names: list[str],
    pct_contributions: npt.NDArray[np.float64],
) -> None:
    """
    Pie chart of percentage risk contributions.

    Args:
        asset_names: Asset ticker labels.
        pct_contributions: Array of % risk contributions from pct_risk_contribution().
    """
    plt.figure(figsize=(8, 8))
    plt.pie(
        pct_contributions,
        labels=asset_names,
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.title("Portfolio Risk Contribution")
    plt.axis("equal")
    plt.savefig("outputs/portfolio_risk_contribution.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_correlation_heatmap(
    correlation_matrix: npt.NDArray[np.float64],
    asset_names: list[str],
) -> None:
    """
    Annotated correlation heatmap using matplotlib imshow.

    Args:
        correlation_matrix: Correlation matrix as a 2-D numpy array.
        asset_names: Asset ticker labels.
    """
    plt.figure(figsize=(8, 6))
    im = plt.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)

    plt.xticks(range(len(asset_names)), asset_names, rotation=45)
    plt.yticks(range(len(asset_names)), asset_names)

    plt.title("Correlation Heatmap")
    plt.colorbar(im, label="Correlation")

    for i in range(len(asset_names)):
        for j in range(len(asset_names)):
            plt.text(
                j,
                i,
                f"{correlation_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig("outputs/correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
