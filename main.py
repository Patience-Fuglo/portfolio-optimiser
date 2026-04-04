"""
Portfolio Optimiser - Main Entry Point

This script runs the complete portfolio optimization pipeline:
1. Load price data
2. Compare portfolio strategies
3. Run backtest with transaction costs
4. Generate risk attribution report
"""

from portfolio_optimiser.data_loader import (
    load_prices,
    calculate_returns,
    annualize_returns,
    calculate_covariance,
    calculate_correlation,
    calculate_annual_volatility,
)
from portfolio_optimiser.optimizer import (
    efficient_frontier,
    max_sharpe_ratio,
    compare_strategies,
    plot_frontier_with_strategies,
)
from portfolio_optimiser.constraints import PortfolioConstraints
from portfolio_optimiser.costs import TransactionCostModel
from portfolio_optimiser.backtester import PortfolioBacktester
from portfolio_optimiser.report import (
    print_portfolio_summary,
    plot_risk_pie,
    plot_correlation_heatmap,
)


def main():
    print("=" * 60)
    print("PORTFOLIO OPTIMISER")
    print("=" * 60)

    # =========================================================
    # 1. LOAD DATA
    # =========================================================
    filepath = "data/sample_price_data.csv"
    prices = load_prices(filepath)
    prices = prices[["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]]

    returns = calculate_returns(prices)
    expected_returns = annualize_returns(returns).values
    cov_matrix = calculate_covariance(returns).values
    corr_matrix = calculate_correlation(returns).values
    stock_vols = calculate_annual_volatility(returns).values
    stock_returns = annualize_returns(returns).values
    stock_names = returns.columns.tolist()

    print(f"\nLoaded {len(prices)} days of price data for {len(stock_names)} stocks")
    print(f"Stocks: {', '.join(stock_names)}")

    # =========================================================
    # 2. COMPARE STRATEGIES
    # =========================================================
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON")
    print("=" * 60)

    strategies, strategy_points = compare_strategies(
        expected_returns,
        cov_matrix,
        stock_names,
        risk_free_rate=0.02,
    )

    # =========================================================
    # 3. EFFICIENT FRONTIER WITH STRATEGIES
    # =========================================================
    frontier_vols, frontier_returns, frontier_weights = efficient_frontier(
        expected_returns,
        cov_matrix,
        n_points=50,
    )

    plot_frontier_with_strategies(
        frontier_vols,
        frontier_returns,
        stock_vols,
        stock_returns,
        stock_names,
        strategy_points,
    )

    # =========================================================
    # 4. BACKTEST MAX SHARPE STRATEGY
    # =========================================================
    print("\n" + "=" * 60)
    print("BACKTEST: MAX SHARPE WITH MONTHLY REBALANCING")
    print("=" * 60)

    cost_model = TransactionCostModel(
        commission_rate=0.001,
        spread_cost=0.0005,
        min_commission=1.0,
    )

    def optimizer_wrapper(exp_ret, cov_mat):
        return max_sharpe_ratio(exp_ret, cov_mat, risk_free_rate=0.02)

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

    # =========================================================
    # 5. RISK ATTRIBUTION REPORT
    # =========================================================
    print("\n" + "=" * 60)
    print("RISK ATTRIBUTION: MAX SHARPE PORTFOLIO")
    print("=" * 60)

    weights = max_sharpe_ratio(expected_returns, cov_matrix, risk_free_rate=0.02)

    report = print_portfolio_summary(
        weights,
        expected_returns,
        cov_matrix,
        stock_names,
    )

    pct_contributions = report["pct_risk_contributions"]
    max_risk_idx = pct_contributions.argmax()

    print(f"\nBiggest Risk Contributor: {stock_names[max_risk_idx]} ({pct_contributions[max_risk_idx]:.2f}%)")

    if report["diversification_ratio"] > 1:
        print("Diversification Ratio > 1: diversification is reducing portfolio risk.")
    else:
        print("Diversification Ratio <= 1: limited diversification benefit.")

    plot_risk_pie(stock_names, pct_contributions)
    plot_correlation_heatmap(corr_matrix, stock_names)

    # =========================================================
    # 6. CONSTRAINED PORTFOLIO COMPARISON
    # =========================================================
    print("\n" + "=" * 60)
    print("CONSTRAINED vs UNCONSTRAINED (max 30% per asset)")
    print("=" * 60)

    from portfolio_optimiser.optimizer import maximize_sharpe, portfolio_return, portfolio_volatility

    constraints_obj = PortfolioConstraints(
        max_weight=0.30,
        min_weight=0.00,
    )

    n_assets = len(stock_names)

    unconstrained_result = maximize_sharpe(
        expected_returns,
        cov_matrix,
        n_assets,
        constraints_obj=None,
        risk_free_rate=0.02,
    )

    constrained_result = maximize_sharpe(
        expected_returns,
        cov_matrix,
        n_assets,
        constraints_obj=constraints_obj,
        risk_free_rate=0.02,
    )

    print("\nUNCONSTRAINED MAX SHARPE:")
    unc_weights = unconstrained_result.x
    for name, w in zip(stock_names, unc_weights):
        print(f"  {name}: {w:.4f}")
    unc_ret = portfolio_return(unc_weights, expected_returns)
    unc_vol = portfolio_volatility(unc_weights, cov_matrix)
    print(f"  Return: {unc_ret:.4f}, Vol: {unc_vol:.4f}, Sharpe: {(unc_ret - 0.02) / unc_vol:.4f}")

    print("\nCONSTRAINED MAX SHARPE (max 30%):")
    con_weights = constrained_result.x
    for name, w in zip(stock_names, con_weights):
        print(f"  {name}: {w:.4f}")
    con_ret = portfolio_return(con_weights, expected_returns)
    con_vol = portfolio_volatility(con_weights, cov_matrix)
    print(f"  Return: {con_ret:.4f}, Vol: {con_vol:.4f}, Sharpe: {(con_ret - 0.02) / con_vol:.4f}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
