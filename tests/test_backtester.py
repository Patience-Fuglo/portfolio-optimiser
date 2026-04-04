import sys; sys.path.insert(0, ".")
from portfolio_optimiser.data_loader import load_prices
from portfolio_optimiser.optimizer import max_sharpe_ratio
from portfolio_optimiser.costs import TransactionCostModel
from portfolio_optimiser.backtester import PortfolioBacktester


def optimizer_wrapper(expected_returns, cov_matrix):
    return max_sharpe_ratio(expected_returns, cov_matrix, risk_free_rate=0.02)


def main():
    filepath = "data/sample_price_data.csv"
    prices = load_prices(filepath)

    # use 5 stocks
    prices = prices[["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]]

    cost_model = TransactionCostModel(
        commission_rate=0.001,
        spread_cost=0.0005,
        min_commission=1.0,
    )

    backtester = PortfolioBacktester(
        optimizer_func=optimizer_wrapper,
        rebalance_months=1,
        cost_model=cost_model,
        starting_value=100000,
        risk_free_rate=0.02,
        lookback_days=60,
    )

    backtester.run(prices)
    backtester.calculate_metrics()
    backtester.print_metrics()
    backtester.plot_results(title="Max Sharpe Monthly Rebalancing Backtest")


if __name__ == "__main__":
    main()
