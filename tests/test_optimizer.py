"""Tests for optimizer module: efficient frontier and all optimization strategies."""

import numpy as np
import pytest

from portfolio_optimiser.optimizer import (
    efficient_frontier,
    max_sharpe_ratio,
    min_variance,
    equal_weight,
    risk_parity,
    maximum_diversification,
    black_litterman,
    maximize_sharpe,
    portfolio_return,
    portfolio_volatility,
    compare_strategies,
)


def test_portfolio_return(expected_returns, cov_matrix, n_assets):
    w = np.ones(n_assets) / n_assets
    ret = portfolio_return(w, expected_returns)
    assert isinstance(ret, float)
    assert ret == pytest.approx(np.dot(w, expected_returns), rel=1e-6)


def test_portfolio_volatility(expected_returns, cov_matrix, n_assets):
    w = np.ones(n_assets) / n_assets
    vol = portfolio_volatility(w, cov_matrix)
    assert vol > 0


def test_efficient_frontier_shape(expected_returns, cov_matrix):
    vols, rets, weights = efficient_frontier(expected_returns, cov_matrix, n_points=20)
    assert len(vols) == len(rets) == len(weights)
    assert len(vols) > 0
    for w in weights:
        assert w.sum() == pytest.approx(1.0, abs=1e-5)
        assert np.all(w >= -1e-6)


def test_efficient_frontier_frontier_shape(expected_returns, cov_matrix):
    vols, rets, _ = efficient_frontier(expected_returns, cov_matrix, n_points=20)
    # Returns should span a range (not all identical)
    assert max(rets) > min(rets)


def test_max_sharpe_weights_sum(expected_returns, cov_matrix):
    w = max_sharpe_ratio(expected_returns, cov_matrix, risk_free_rate=0.02)
    assert w.sum() == pytest.approx(1.0, abs=1e-5)
    assert np.all(w >= -1e-6)


def test_max_sharpe_beats_equal_weight(expected_returns, cov_matrix):
    w_ms = max_sharpe_ratio(expected_returns, cov_matrix, risk_free_rate=0.02)
    w_ew = equal_weight(len(expected_returns))
    sharpe_ms = (portfolio_return(w_ms, expected_returns) - 0.02) / portfolio_volatility(w_ms, cov_matrix)
    sharpe_ew = (portfolio_return(w_ew, expected_returns) - 0.02) / portfolio_volatility(w_ew, cov_matrix)
    assert sharpe_ms >= sharpe_ew - 1e-4


def test_min_variance_weights_sum(expected_returns, cov_matrix):
    w = min_variance(expected_returns, cov_matrix)
    assert w.sum() == pytest.approx(1.0, abs=1e-5)
    assert np.all(w >= -1e-6)


def test_min_variance_lowest_vol(expected_returns, cov_matrix):
    w_mv = min_variance(expected_returns, cov_matrix)
    w_ms = max_sharpe_ratio(expected_returns, cov_matrix, risk_free_rate=0.02)
    w_ew = equal_weight(len(expected_returns))
    vol_mv = portfolio_volatility(w_mv, cov_matrix)
    assert vol_mv <= portfolio_volatility(w_ms, cov_matrix) + 1e-4
    assert vol_mv <= portfolio_volatility(w_ew, cov_matrix) + 1e-4


def test_equal_weight(n_assets):
    w = equal_weight(n_assets)
    assert w.sum() == pytest.approx(1.0, abs=1e-8)
    assert np.allclose(w, 1.0 / n_assets)


def test_risk_parity_weights(expected_returns, cov_matrix, n_assets):
    w = risk_parity(cov_matrix)
    assert w.sum() == pytest.approx(1.0, abs=1e-5)
    assert np.all(w >= -1e-6)
    # Risk contributions should be approximately equal
    sigma_w = cov_matrix @ w
    port_vol = portfolio_volatility(w, cov_matrix)
    rc = w * sigma_w / port_vol
    assert np.std(rc) < 0.05


def test_maximum_diversification_weights(cov_matrix, n_assets):
    w = maximum_diversification(cov_matrix)
    assert w.sum() == pytest.approx(1.0, abs=1e-5)
    assert np.all(w >= -1e-6)


def test_black_litterman_shape(cov_matrix, expected_returns, n_assets):
    market_weights = np.ones(n_assets) / n_assets
    P = np.array([[1, 0, 0, 0, 0], [0, 1, -1, 0, 0]], dtype=float)
    Q = np.array([0.25, 0.05])
    mu_bl = black_litterman(cov_matrix, market_weights, P, Q)
    assert mu_bl.shape == (n_assets,)


def test_black_litterman_view_influence(cov_matrix, n_assets):
    market_weights = np.ones(n_assets) / n_assets
    # Strong bullish view on asset 0
    P = np.array([[1, 0, 0, 0, 0]], dtype=float)
    Q = np.array([0.50])
    mu_bl = black_litterman(cov_matrix, market_weights, P, Q)
    delta = 2.5
    pi = delta * cov_matrix @ market_weights
    # BL return for asset 0 should shift toward 50% view
    assert mu_bl[0] > pi[0]


def test_maximize_sharpe_constrained(expected_returns, cov_matrix, n_assets):
    from portfolio_optimiser.constraints import PortfolioConstraints
    c = PortfolioConstraints(max_weight=0.30, min_weight=0.00)
    result = maximize_sharpe(expected_returns, cov_matrix, n_assets, constraints_obj=c, risk_free_rate=0.02)
    w = result.x
    assert np.all(w <= 0.30 + 1e-5)
    assert np.all(w >= -1e-6)
    assert w.sum() == pytest.approx(1.0, abs=1e-4)


def test_compare_strategies_returns_all_six(expected_returns, cov_matrix, stock_names):
    strategies, points = compare_strategies(expected_returns, cov_matrix, stock_names, risk_free_rate=0.02)
    expected_keys = {"Max Sharpe", "Min Variance", "Equal Weight", "Risk Parity", "Max Diversification", "HRP"}
    assert set(strategies.keys()) == expected_keys
    for name, w in strategies.items():
        assert w.sum() == pytest.approx(1.0, abs=1e-4), f"{name} weights don't sum to 1"
