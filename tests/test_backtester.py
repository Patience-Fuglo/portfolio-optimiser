"""Tests for PortfolioBacktester: walk-forward engine and full metrics suite."""

import numpy as np
import pytest

from portfolio_optimiser.backtester import PortfolioBacktester
from portfolio_optimiser.costs import TransactionCostModel
from portfolio_optimiser.optimizer import max_sharpe_ratio


@pytest.fixture
def backtester(sample_prices):
    cost_model = TransactionCostModel(commission_rate=0.001, spread_cost=0.0005, min_commission=1.0)
    bt = PortfolioBacktester(
        optimizer_func=lambda e, c: max_sharpe_ratio(e, c, risk_free_rate=0.02),
        rebalance_months=1,
        cost_model=cost_model,
        starting_value=100_000,
        risk_free_rate=0.02,
        lookback_days=60,
    )
    bt.run(sample_prices)
    return bt


def test_backtest_runs(backtester):
    assert backtester.portfolio_history is not None
    assert len(backtester.portfolio_history) > 0


def test_portfolio_value_positive(backtester):
    values = [v for _, v in backtester.portfolio_history]
    assert all(v > 0 for v in values)


def test_metrics_keys(backtester):
    metrics, _ = backtester.calculate_metrics()
    required = {
        "Annualized Return", "Annualized Volatility", "Sharpe Ratio",
        "Sortino Ratio", "Calmar Ratio", "Max Drawdown", "CVaR (95%)",
        "Beta", "Alpha", "Information Ratio",
    }
    assert required.issubset(set(metrics.keys()))


def test_sharpe_ratio_positive(backtester):
    metrics, _ = backtester.calculate_metrics()
    assert metrics["Sharpe Ratio"] > 0


def test_max_drawdown_negative(backtester):
    metrics, _ = backtester.calculate_metrics()
    assert metrics["Max Drawdown"] < 0


def test_cvar_positive(backtester):
    metrics, _ = backtester.calculate_metrics()
    assert metrics["CVaR (95%)"] > 0


def test_sortino_ratio_present(backtester):
    metrics, _ = backtester.calculate_metrics()
    assert "Sortino Ratio" in metrics
    assert np.isfinite(metrics["Sortino Ratio"])


def test_calmar_ratio_present(backtester):
    metrics, _ = backtester.calculate_metrics()
    assert "Calmar Ratio" in metrics
    assert np.isfinite(metrics["Calmar Ratio"])


def test_annualized_volatility_reasonable(backtester):
    metrics, _ = backtester.calculate_metrics()
    vol = metrics["Annualized Volatility"]
    assert 0.01 < vol < 2.0


def test_starting_value_respected(sample_prices):
    bt = PortfolioBacktester(
        optimizer_func=lambda e, c: max_sharpe_ratio(e, c, risk_free_rate=0.02),
        rebalance_months=1,
        starting_value=50_000,
        risk_free_rate=0.02,
        lookback_days=60,
    )
    bt.run(sample_prices)
    # starting_value attribute holds the configured capital
    assert bt.starting_value == pytest.approx(50_000)
    # first recorded value is slightly below starting due to initial transaction costs
    assert bt.portfolio_history[0][1] < 50_000 * 1.01
    assert bt.portfolio_history[0][1] > 50_000 * 0.90
