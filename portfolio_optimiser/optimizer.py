"""
Optimizer Module
================

Portfolio optimization algorithms including efficient frontier construction,
Max Sharpe, Min Variance, Equal Weight, and Risk Parity strategies.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.optimize import minimize, OptimizeResult

from portfolio_optimiser.constraints import PortfolioConstraints


# Type aliases for clarity
Weights = npt.NDArray[np.float64]
CovMatrix = npt.NDArray[np.float64]
Returns = npt.NDArray[np.float64]


def portfolio_return(weights: Weights, expected_returns: Returns) -> float:
    """
    Calculate expected portfolio return.

    Args:
        weights: Array of portfolio weights (must sum to 1).
        expected_returns: Array of expected returns for each asset.

    Returns:
        Expected portfolio return as a scalar.
    """
    return float(np.dot(weights, expected_returns))


def portfolio_volatility(weights: Weights, cov_matrix: CovMatrix) -> float:
    """
    Calculate portfolio volatility (standard deviation).

    Args:
        weights: Array of portfolio weights.
        cov_matrix: Covariance matrix of asset returns.

    Returns:
        Portfolio volatility as a scalar.
    """
    return float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))


def negative_sharpe_ratio(
    weights: Weights,
    expected_returns: Returns,
    cov_matrix: CovMatrix,
    risk_free_rate: float = 0.02,
) -> float:
    """
    Calculate negative Sharpe ratio (for minimization).

    Args:
        weights: Portfolio weights.
        expected_returns: Expected returns for each asset.
        cov_matrix: Covariance matrix.
        risk_free_rate: Risk-free rate (default 2%).

    Returns:
        Negative Sharpe ratio (minimize this to maximize Sharpe).
    """
    port_return = portfolio_return(weights, expected_returns)
    port_vol = portfolio_volatility(weights, cov_matrix)

    if port_vol == 0:
        return 1e6

    sharpe = (port_return - risk_free_rate) / port_vol
    return -sharpe


def minimize_volatility(
    target_return: float,
    expected_returns: Returns,
    cov_matrix: CovMatrix,
    n_assets: int,
    constraints_obj: PortfolioConstraints | None = None,
    asset_sectors: list[str] | None = None,
) -> OptimizeResult:
    """
    Find minimum volatility portfolio for a given target return.

    Args:
        target_return: Target portfolio return.
        expected_returns: Expected returns for each asset.
        cov_matrix: Covariance matrix.
        n_assets: Number of assets.
        constraints_obj: Optional portfolio constraints.
        asset_sectors: Optional list of sector labels for each asset.

    Returns:
        SciPy OptimizeResult object.
    """
    initial_weights = np.array([1 / n_assets] * n_assets)

    if constraints_obj is None:
        bounds = tuple((0, 1) for _ in range(n_assets))
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    else:
        bounds = constraints_obj.get_bounds(n_assets)
        constraints = constraints_obj.get_all_constraints(asset_sectors)

    constraints = list(constraints) + [
        {"type": "eq", "fun": lambda w: portfolio_return(w, expected_returns) - target_return}
    ]

    result = minimize(
        portfolio_volatility,
        initial_weights,
        args=(cov_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    return result


def efficient_frontier(
    expected_returns: Returns,
    cov_matrix: CovMatrix,
    n_points: int = 50,
    constraints_obj: PortfolioConstraints | None = None,
    asset_sectors: list[str] | None = None,
) -> tuple[list[float], list[float], list[Weights]]:
    """
    Construct the efficient frontier.

    Args:
        expected_returns: Expected returns for each asset.
        cov_matrix: Covariance matrix.
        n_points: Number of points on the frontier.
        constraints_obj: Optional portfolio constraints.
        asset_sectors: Optional sector labels.

    Returns:
        Tuple of (volatilities, returns, weights) for each frontier point.
    """
    n_assets = len(expected_returns)
    target_returns = np.linspace(expected_returns.min(), expected_returns.max(), n_points)

    frontier_vols: list[float] = []
    frontier_returns: list[float] = []
    frontier_weights: list[Weights] = []

    for target in target_returns:
        result = minimize_volatility(
            target,
            expected_returns,
            cov_matrix,
            n_assets,
            constraints_obj=constraints_obj,
            asset_sectors=asset_sectors,
        )

        if result.success:
            weights = result.x
            vol = portfolio_volatility(weights, cov_matrix)
            ret = portfolio_return(weights, expected_returns)

            frontier_vols.append(vol)
            frontier_returns.append(ret)
            frontier_weights.append(weights)

    return frontier_vols, frontier_returns, frontier_weights


def max_sharpe_ratio(
    expected_returns: Returns,
    cov_matrix: CovMatrix,
    risk_free_rate: float = 0.02,
) -> Weights:
    """
    Find the maximum Sharpe ratio portfolio.

    Args:
        expected_returns: Expected returns for each asset.
        cov_matrix: Covariance matrix.
        risk_free_rate: Risk-free rate (default 2%).

    Returns:
        Optimal portfolio weights.
    """
    n_assets = len(expected_returns)
    initial_weights = np.array([1 / n_assets] * n_assets)

    bounds = tuple((0, 1) for _ in range(n_assets))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(expected_returns, cov_matrix, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    return result.x


def min_variance(expected_returns: Returns, cov_matrix: CovMatrix) -> Weights:
    """
    Find the minimum variance portfolio.

    Args:
        expected_returns: Expected returns (used for array length).
        cov_matrix: Covariance matrix.

    Returns:
        Optimal portfolio weights.
    """
    n_assets = len(expected_returns)
    initial_weights = np.array([1 / n_assets] * n_assets)

    bounds = tuple((0, 1) for _ in range(n_assets))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    result = minimize(
        portfolio_volatility,
        initial_weights,
        args=(cov_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    return result.x


def equal_weight(n_assets: int) -> Weights:
    """
    Create equal-weight portfolio.

    Args:
        n_assets: Number of assets.

    Returns:
        Array of equal weights (1/n each).
    """
    return np.array([1 / n_assets] * n_assets)


def risk_contributions(weights: Weights, cov_matrix: CovMatrix) -> Weights:
    """
    Calculate risk contribution of each asset.

    Args:
        weights: Portfolio weights.
        cov_matrix: Covariance matrix.

    Returns:
        Array of risk contributions (sum equals portfolio volatility).
    """
    port_vol = portfolio_volatility(weights, cov_matrix)
    marginal_contrib = np.dot(cov_matrix, weights)

    if port_vol == 0:
        return np.zeros_like(weights)

    return weights * marginal_contrib / port_vol


def risk_parity_objective(weights: Weights, cov_matrix: CovMatrix) -> float:
    """
    Objective function for risk parity optimization.

    Minimizes the squared difference between each asset's risk contribution
    and the target (equal) risk contribution.

    Args:
        weights: Portfolio weights.
        cov_matrix: Covariance matrix.

    Returns:
        Sum of squared deviations from target risk contributions.
    """
    n_assets = len(weights)
    port_vol = portfolio_volatility(weights, cov_matrix)
    rc = risk_contributions(weights, cov_matrix)

    target_rc = np.array([port_vol / n_assets] * n_assets)
    return float(np.sum((rc - target_rc) ** 2))


def risk_parity(cov_matrix: CovMatrix) -> Weights:
    """
    Find the risk parity portfolio.

    Equalizes risk contribution across all assets.

    Args:
        cov_matrix: Covariance matrix.

    Returns:
        Risk parity portfolio weights.
    """
    n_assets = cov_matrix.shape[0]
    initial_weights = np.array([1 / n_assets] * n_assets)

    bounds = tuple((0, 1) for _ in range(n_assets))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    result = minimize(
        risk_parity_objective,
        initial_weights,
        args=(cov_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    return result.x


def maximize_sharpe(
    expected_returns: Returns,
    cov_matrix: CovMatrix,
    n_assets: int,
    constraints_obj: PortfolioConstraints | None = None,
    asset_sectors: list[str] | None = None,
    risk_free_rate: float = 0.0,
) -> OptimizeResult:
    """
    Find max-Sharpe portfolio with optional constraints.

    Args:
        expected_returns: Expected returns for each asset.
        cov_matrix: Covariance matrix.
        n_assets: Number of assets.
        constraints_obj: Optional portfolio constraints.
        asset_sectors: Optional sector labels.
        risk_free_rate: Risk-free rate.

    Returns:
        SciPy OptimizeResult object with optimal weights in result.x.
    """
    initial_weights = np.array([1 / n_assets] * n_assets)

    if constraints_obj is None:
        bounds = tuple((0, 1) for _ in range(n_assets))
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    else:
        bounds = constraints_obj.get_bounds(n_assets)
        constraints = constraints_obj.get_all_constraints(asset_sectors)

    result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(expected_returns, cov_matrix, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    return result


def plot_frontier(frontier_vols, frontier_returns, stock_vols, stock_returns, stock_names):
    plt.figure(figsize=(10, 6))
    plt.plot(frontier_vols, frontier_returns, label="Efficient Frontier", linewidth=2)
    plt.scatter(stock_vols, stock_returns, label="Stocks", s=80)

    for i, name in enumerate(stock_names):
        plt.annotate(name, (stock_vols[i], stock_returns[i]), xytext=(5, 5), textcoords="offset points")

    plt.xlabel("Risk (Volatility)")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/efficient_frontier.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_frontiers(unconstrained_vols, unconstrained_returns, constrained_vols, constrained_returns):
    plt.figure(figsize=(10, 6))

    plt.plot(unconstrained_vols, unconstrained_returns, label="Unconstrained Frontier", linewidth=2)
    plt.plot(constrained_vols, constrained_returns, label="Constrained Frontier", linewidth=2, linestyle="--")

    plt.xlabel("Risk (Volatility)")
    plt.ylabel("Expected Return")
    plt.title("Constrained vs Unconstrained Efficient Frontier")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/constrained_vs_unconstrained.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_frontier_with_strategies(
    frontier_vols,
    frontier_returns,
    stock_vols,
    stock_returns,
    stock_names,
    strategy_points,
):
    plt.figure(figsize=(10, 6))

    plt.plot(frontier_vols, frontier_returns, label="Efficient Frontier", linewidth=2)
    plt.scatter(stock_vols, stock_returns, label="Stocks", s=80)

    for i, name in enumerate(stock_names):
        plt.annotate(name, (stock_vols[i], stock_returns[i]), xytext=(5, 5), textcoords="offset points")

    for strategy_name, (vol, ret) in strategy_points.items():
        plt.scatter(vol, ret, s=120, label=strategy_name)
        plt.annotate(strategy_name, (vol, ret), xytext=(5, -10), textcoords="offset points")

    plt.xlabel("Risk (Volatility)")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier with Portfolio Strategies")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/frontier_with_strategies.png", dpi=150, bbox_inches="tight")
    plt.show()


def compare_strategies(expected_returns, cov_matrix, stock_names, risk_free_rate=0.02):
    n_assets = len(expected_returns)

    strategies = {
        "Max Sharpe": max_sharpe_ratio(expected_returns, cov_matrix, risk_free_rate),
        "Min Variance": min_variance(expected_returns, cov_matrix),
        "Equal Weight": equal_weight(n_assets),
        "Risk Parity": risk_parity(cov_matrix),
    }

    print("\n=== STRATEGY COMPARISON ===\n")
    print(f"{'Strategy':<15} {'Return':>10} {'Volatility':>12} {'Sharpe':>10}")
    print("-" * 50)

    strategy_points = {}

    for name, weights in strategies.items():
        port_ret = portfolio_return(weights, expected_returns)
        port_vol = portfolio_volatility(weights, cov_matrix)
        sharpe = (port_ret - risk_free_rate) / port_vol if port_vol != 0 else 0

        strategy_points[name] = (port_vol, port_ret)

        print(f"{name:<15} {port_ret:>10.4f} {port_vol:>12.4f} {sharpe:>10.4f}")
        print("Weights:")
        for stock, weight in zip(stock_names, weights):
            print(f"  {stock}: {weight:.4f}")
        print()

    return strategies, strategy_points