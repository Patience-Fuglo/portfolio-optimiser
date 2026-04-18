"""Tests for report module: CVaR, risk attribution, diversification ratio."""

import numpy as np
import pytest

from portfolio_optimiser.report import (
    risk_contribution,
    pct_risk_contribution,
    diversification_ratio,
    cvar,
    print_portfolio_summary,
)
from portfolio_optimiser.optimizer import max_sharpe_ratio
from portfolio_optimiser.data_loader import calculate_returns


def test_risk_contribution_sums_to_portfolio_vol(expected_returns, cov_matrix, n_assets):
    w = max_sharpe_ratio(expected_returns, cov_matrix, risk_free_rate=0.02)
    rc = risk_contribution(w, cov_matrix)
    port_vol = np.sqrt(w @ cov_matrix @ w)
    assert rc.sum() == pytest.approx(port_vol, rel=1e-4)


def test_pct_risk_contribution_sums_to_100(expected_returns, cov_matrix):
    w = max_sharpe_ratio(expected_returns, cov_matrix, risk_free_rate=0.02)
    pct = pct_risk_contribution(w, cov_matrix)
    assert pct.sum() == pytest.approx(100.0, abs=0.1)


def test_pct_risk_contribution_non_negative(expected_returns, cov_matrix):
    w = max_sharpe_ratio(expected_returns, cov_matrix, risk_free_rate=0.02)
    pct = pct_risk_contribution(w, cov_matrix)
    assert np.all(pct >= -1e-6)


def test_diversification_ratio_ew_gt_one(cov_matrix, n_assets):
    w = np.ones(n_assets) / n_assets
    dr = diversification_ratio(w, cov_matrix)
    assert dr > 1.0


def test_diversification_ratio_single_asset():
    cov = np.array([[0.04]])
    w = np.array([1.0])
    dr = diversification_ratio(w, cov)
    assert dr == pytest.approx(1.0, abs=1e-6)


def test_cvar_daily_positive(sample_returns):
    w = np.ones(sample_returns.shape[1]) / sample_returns.shape[1]
    port_ret = sample_returns.values @ w
    daily_cvar, ann_cvar = cvar(port_ret, confidence=0.95)
    assert daily_cvar > 0
    assert ann_cvar > daily_cvar


def test_cvar_annualised_scaling(sample_returns):
    w = np.ones(sample_returns.shape[1]) / sample_returns.shape[1]
    port_ret = sample_returns.values @ w
    d, a = cvar(port_ret, confidence=0.95)
    assert a == pytest.approx(d * np.sqrt(252), rel=1e-5)


def test_cvar_95_more_conservative_than_90(sample_returns):
    w = np.ones(sample_returns.shape[1]) / sample_returns.shape[1]
    port_ret = sample_returns.values @ w
    _, cvar_95 = cvar(port_ret, confidence=0.95)
    _, cvar_90 = cvar(port_ret, confidence=0.90)
    assert cvar_95 >= cvar_90


def test_print_portfolio_summary_keys(expected_returns, cov_matrix, stock_names):
    w = max_sharpe_ratio(expected_returns, cov_matrix, risk_free_rate=0.02)
    result = print_portfolio_summary(w, expected_returns, cov_matrix, stock_names)
    assert "pct_risk_contributions" in result
    assert "diversification_ratio" in result
    assert result["diversification_ratio"] > 0
