import sys; sys.path.insert(0, ".")
import numpy as np

from portfolio_optimiser.data_loader import (
    load_prices,
    calculate_returns,
    annualize_returns,
    calculate_covariance,
)
from portfolio_optimiser.optimizer import efficient_frontier, maximize_sharpe, portfolio_return, portfolio_volatility, plot_frontiers
from portfolio_optimiser.constraints import PortfolioConstraints


def compare_constrained_vs_unconstrained(expected_returns, cov_matrix, constraints_obj, asset_sectors=None):
    unconstrained_vols, unconstrained_returns, unconstrained_weights = efficient_frontier(
        expected_returns,
        cov_matrix,
        n_points=50,
        constraints_obj=None,
        asset_sectors=asset_sectors,
    )

    constrained_vols, constrained_returns, constrained_weights = efficient_frontier(
        expected_returns,
        cov_matrix,
        n_points=50,
        constraints_obj=constraints_obj,
        asset_sectors=asset_sectors,
    )

    plot_frontiers(
        unconstrained_vols,
        unconstrained_returns,
        constrained_vols,
        constrained_returns,
    )

    return (
        unconstrained_vols,
        unconstrained_returns,
        unconstrained_weights,
        constrained_vols,
        constrained_returns,
        constrained_weights,
    )


def main():
    filepath = "data/sample_price_data.csv"

    prices = load_prices(filepath)
    prices = prices[["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]]

    returns = calculate_returns(prices)
    expected_returns = annualize_returns(returns).values
    cov_matrix = calculate_covariance(returns).values
    stock_names = returns.columns.tolist()

    # Example sectors
    asset_sectors = ["Tech", "Tech", "Tech", "Consumer", "Auto"]

    constraints_obj = PortfolioConstraints(
        max_weight=0.30,
        min_weight=0.00,
        sector_limits={"Tech": 0.70}
    )

    compare_constrained_vs_unconstrained(
        expected_returns,
        cov_matrix,
        constraints_obj,
        asset_sectors=asset_sectors,
    )

    n_assets = len(stock_names)

    unconstrained_result = maximize_sharpe(
        expected_returns,
        cov_matrix,
        n_assets,
        constraints_obj=None,
        asset_sectors=asset_sectors,
        risk_free_rate=0.0,
    )

    constrained_result = maximize_sharpe(
        expected_returns,
        cov_matrix,
        n_assets,
        constraints_obj=constraints_obj,
        asset_sectors=asset_sectors,
        risk_free_rate=0.0,
    )

    print("\n=== MAX SHARPE PORTFOLIO: UNCONSTRAINED ===")
    unconstrained_weights = unconstrained_result.x
    for name, weight in zip(stock_names, unconstrained_weights):
        print(f"{name}: {weight:.4f}")

    unconstrained_ret = portfolio_return(unconstrained_weights, expected_returns)
    unconstrained_vol = portfolio_volatility(unconstrained_weights, cov_matrix)
    unconstrained_sharpe = unconstrained_ret / unconstrained_vol

    print(f"Return: {unconstrained_ret:.4f}")
    print(f"Volatility: {unconstrained_vol:.4f}")
    print(f"Sharpe: {unconstrained_sharpe:.4f}")

    print("\n=== MAX SHARPE PORTFOLIO: CONSTRAINED ===")
    constrained_weights = constrained_result.x
    for name, weight in zip(stock_names, constrained_weights):
        print(f"{name}: {weight:.4f}")

    constrained_ret = portfolio_return(constrained_weights, expected_returns)
    constrained_vol = portfolio_volatility(constrained_weights, cov_matrix)
    constrained_sharpe = constrained_ret / constrained_vol

    print(f"Return: {constrained_ret:.4f}")
    print(f"Volatility: {constrained_vol:.4f}")
    print(f"Sharpe: {constrained_sharpe:.4f}")

    print("\n=== PERFORMANCE LOSS FROM CONSTRAINTS ===")
    print(f"Return loss: {unconstrained_ret - constrained_ret:.4f}")
    print(f"Sharpe loss: {unconstrained_sharpe - constrained_sharpe:.4f}")

    print("\n=== CHECK MAX WEIGHT <= 30% ===")
    print(f"Largest constrained weight: {constrained_weights.max():.4f}")


if __name__ == "__main__":
    main()
