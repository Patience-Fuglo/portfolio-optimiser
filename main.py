"""
Portfolio Optimiser - Main Entry Point

Complete portfolio optimization pipeline (v2.0):
1. Load price data
2. Compare all five strategies (including Max Diversification)
3. Black-Litterman view-blending demo
4. Walk-forward backtest with full metrics (Sharpe/Sortino/Calmar/CVaR/Alpha/Beta)
5. Risk attribution report with CVaR
6. Constrained vs unconstrained comparison
"""

import numpy as np

from portfolio_optimiser.data_loader import (
    load_prices,
    calculate_returns,
    annualize_returns,
    calculate_covariance,
    calculate_covariance_shrunk,
    calculate_correlation,
    calculate_annual_volatility,
)
from portfolio_optimiser.optimizer import (
    efficient_frontier,
    max_sharpe_ratio,
    compare_strategies,
    plot_frontier_with_strategies,
    black_litterman,
    maximize_sharpe,
    portfolio_return,
    portfolio_volatility,
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
    print("PORTFOLIO OPTIMISER  v2.0")
    print("=" * 60)

    # =========================================================
    # 1. LOAD DATA
    # =========================================================
    filepath = "data/sample_price_data.csv"
    prices = load_prices(filepath)
    prices = prices[["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]]

    returns = calculate_returns(prices)
    expected_returns = annualize_returns(returns).values

    # Ledoit-Wolf shrinkage for a well-conditioned covariance estimate
    cov_matrix = calculate_covariance_shrunk(returns).values
    corr_matrix = calculate_correlation(returns).values
    stock_vols = calculate_annual_volatility(returns).values
    stock_names = returns.columns.tolist()
    n_assets = len(stock_names)

    print(f"\nLoaded {len(prices)} days of price data for {n_assets} stocks")
    print(f"Stocks: {', '.join(stock_names)}")
    print("Covariance: Ledoit-Wolf shrinkage applied")

    # =========================================================
    # 2. COMPARE ALL FIVE STRATEGIES
    # =========================================================
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON  (incl. Max Diversification)")
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
    frontier_vols, frontier_returns, _ = efficient_frontier(
        expected_returns,
        cov_matrix,
        n_points=50,
    )

    plot_frontier_with_strategies(
        frontier_vols,
        frontier_returns,
        stock_vols,
        expected_returns,
        stock_names,
        strategy_points,
    )

    # =========================================================
    # 4. BLACK-LITTERMAN DEMO
    # =========================================================
    print("\n" + "=" * 60)
    print("BLACK-LITTERMAN DEMO")
    print("=" * 60)

    market_weights = np.ones(n_assets) / n_assets  # equal-weight as equilibrium

    # Views: bullish on AAPL (+25% p.a.), MSFT outperforms GOOGL by +5%
    P = np.array([
        [1, 0, 0, 0, 0],   # absolute: AAPL
        [0, 1, -1, 0, 0],  # relative: MSFT vs GOOGL
    ], dtype=float)
    Q = np.array([0.25, 0.05])

    mu_bl = black_litterman(cov_matrix, market_weights, P, Q)

    print("\nEquilibrium vs Black-Litterman expected returns:")
    print(f"  {'Asset':<8} {'Equilibrium':>14} {'BL Posterior':>14}")
    print("  " + "-" * 38)
    from portfolio_optimiser.optimizer import portfolio_volatility as _pv
    delta = 2.5
    pi = delta * cov_matrix @ market_weights
    for name, eq_r, bl_r in zip(stock_names, pi, mu_bl):
        print(f"  {name:<8} {eq_r:>13.4f}  {bl_r:>13.4f}")

    # Optimise with BL returns
    bl_weights = max_sharpe_ratio(mu_bl, cov_matrix, risk_free_rate=0.02)
    bl_ret = portfolio_return(bl_weights, mu_bl)
    bl_vol = portfolio_volatility(bl_weights, cov_matrix)
    print(f"\nBL Max-Sharpe portfolio:  Return={bl_ret:.4f}  Vol={bl_vol:.4f}  "
          f"Sharpe={(bl_ret - 0.02) / bl_vol:.4f}")
    print("Weights:")
    for name, w in zip(stock_names, bl_weights):
        print(f"  {name}: {w:.4f}")

    # =========================================================
    # 5. BACKTEST MAX SHARPE STRATEGY
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
        starting_value=100_000,
        risk_free_rate=0.02,
        lookback_days=60,
    )

    backtester.run(prices)
    backtester.calculate_metrics()
    backtester.print_metrics()
    backtester.plot_results(title="Max Sharpe — Monthly Rebalancing")

    # =========================================================
    # 6. RISK ATTRIBUTION REPORT WITH CVaR
    # =========================================================
    print("\n" + "=" * 60)
    print("RISK ATTRIBUTION: MAX SHARPE PORTFOLIO")
    print("=" * 60)

    weights = max_sharpe_ratio(expected_returns, cov_matrix, risk_free_rate=0.02)

    # Compute realised portfolio daily returns for CVaR
    daily_port_returns = (calculate_returns(prices).values @ weights)

    report = print_portfolio_summary(
        weights,
        expected_returns,
        cov_matrix,
        stock_names,
        daily_returns=daily_port_returns,
    )

    pct_contributions = report["pct_risk_contributions"]
    max_risk_idx = int(pct_contributions.argmax())
    print(f"\nBiggest Risk Contributor: {stock_names[max_risk_idx]} "
          f"({pct_contributions[max_risk_idx]:.2f}%)")

    if report["diversification_ratio"] > 1:
        print("Diversification Ratio > 1: diversification is reducing portfolio risk.")
    else:
        print("Diversification Ratio <= 1: limited diversification benefit.")

    plot_risk_pie(stock_names, pct_contributions)
    plot_correlation_heatmap(corr_matrix, stock_names)

    # =========================================================
    # 7. CONSTRAINED vs UNCONSTRAINED
    # =========================================================
    print("\n" + "=" * 60)
    print("CONSTRAINED vs UNCONSTRAINED (max 30% per asset)")
    print("=" * 60)

    constraints_obj = PortfolioConstraints(max_weight=0.30, min_weight=0.00)

    unconstrained_result = maximize_sharpe(
        expected_returns, cov_matrix, n_assets,
        constraints_obj=None, risk_free_rate=0.02,
    )
    constrained_result = maximize_sharpe(
        expected_returns, cov_matrix, n_assets,
        constraints_obj=constraints_obj, risk_free_rate=0.02,
    )

    print("\nUNCONSTRAINED MAX SHARPE:")
    unc_w = unconstrained_result.x
    for name, w in zip(stock_names, unc_w):
        print(f"  {name}: {w:.4f}")
    unc_ret = portfolio_return(unc_w, expected_returns)
    unc_vol = portfolio_volatility(unc_w, cov_matrix)
    print(f"  Return: {unc_ret:.4f}  Vol: {unc_vol:.4f}  "
          f"Sharpe: {(unc_ret - 0.02) / unc_vol:.4f}")

    print("\nCONSTRAINED MAX SHARPE (max 30%):")
    con_w = constrained_result.x
    for name, w in zip(stock_names, con_w):
        print(f"  {name}: {w:.4f}")
    con_ret = portfolio_return(con_w, expected_returns)
    con_vol = portfolio_volatility(con_w, cov_matrix)
    print(f"  Return: {con_ret:.4f}  Vol: {con_vol:.4f}  "
          f"Sharpe: {(con_ret - 0.02) / con_vol:.4f}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
