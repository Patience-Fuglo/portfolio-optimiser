import sys; sys.path.insert(0, ".")
from portfolio_optimiser.data_loader import (
    load_prices,
    calculate_returns,
    annualize_returns,
    calculate_covariance,
    calculate_annual_volatility,
)
from portfolio_optimiser.optimizer import (
    efficient_frontier,
    compare_strategies,
    plot_frontier_with_strategies,
)


def main():
    filepath = "data/sample_price_data.csv"

    prices = load_prices(filepath)
    prices = prices[["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]]

    returns = calculate_returns(prices)

    expected_returns = annualize_returns(returns).values
    cov_matrix = calculate_covariance(returns).values
    stock_vols = calculate_annual_volatility(returns).values
    stock_returns = annualize_returns(returns).values
    stock_names = returns.columns.tolist()

    frontier_vols, frontier_returns, frontier_weights = efficient_frontier(
        expected_returns,
        cov_matrix,
        n_points=50,
    )

    strategies, strategy_points = compare_strategies(
        expected_returns,
        cov_matrix,
        stock_names,
        risk_free_rate=0.02,
    )

    plot_frontier_with_strategies(
        frontier_vols,
        frontier_returns,
        stock_vols,
        stock_returns,
        stock_names,
        strategy_points,
    )


if __name__ == "__main__":
    main()
