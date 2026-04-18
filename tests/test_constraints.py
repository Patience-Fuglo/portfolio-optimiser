"""Tests for PortfolioConstraints: weight bounds and sector limits."""

import numpy as np
import pytest

from portfolio_optimiser.constraints import PortfolioConstraints
from portfolio_optimiser.optimizer import maximize_sharpe, portfolio_return, portfolio_volatility


def test_constraints_default_bounds(n_assets):
    c = PortfolioConstraints(max_weight=0.40, min_weight=0.05)
    bounds = c.get_bounds(n_assets)
    assert len(bounds) == n_assets
    for lo, hi in bounds:
        assert lo == pytest.approx(0.05)
        assert hi == pytest.approx(0.40)


def test_constraints_respected_in_optimizer(expected_returns, cov_matrix, n_assets):
    c = PortfolioConstraints(max_weight=0.30, min_weight=0.00)
    result = maximize_sharpe(expected_returns, cov_matrix, n_assets, constraints_obj=c, risk_free_rate=0.02)
    w = result.x
    assert np.all(w <= 0.30 + 1e-5)
    assert np.all(w >= -1e-6)


def test_sector_limits(expected_returns, cov_matrix, n_assets):
    sectors = ["Tech", "Tech", "Tech", "Consumer", "Auto"]
    c = PortfolioConstraints(max_weight=0.40, min_weight=0.00, sector_limits={"Tech": 0.70})
    result = maximize_sharpe(
        expected_returns, cov_matrix, n_assets,
        constraints_obj=c, asset_sectors=sectors, risk_free_rate=0.02,
    )
    w = result.x
    tech_exposure = sum(w[i] for i, s in enumerate(sectors) if s == "Tech")
    assert tech_exposure <= 0.70 + 1e-4


def test_constrained_sharpe_le_unconstrained(expected_returns, cov_matrix, n_assets):
    c = PortfolioConstraints(max_weight=0.30, min_weight=0.00)
    unc = maximize_sharpe(expected_returns, cov_matrix, n_assets, constraints_obj=None, risk_free_rate=0.02)
    con = maximize_sharpe(expected_returns, cov_matrix, n_assets, constraints_obj=c, risk_free_rate=0.02)
    unc_sharpe = (portfolio_return(unc.x, expected_returns) - 0.02) / portfolio_volatility(unc.x, cov_matrix)
    con_sharpe = (portfolio_return(con.x, expected_returns) - 0.02) / portfolio_volatility(con.x, cov_matrix)
    assert unc_sharpe >= con_sharpe - 1e-4


def test_weights_sum_to_one_constrained(expected_returns, cov_matrix, n_assets):
    c = PortfolioConstraints(max_weight=0.30, min_weight=0.00)
    result = maximize_sharpe(expected_returns, cov_matrix, n_assets, constraints_obj=c, risk_free_rate=0.02)
    assert result.x.sum() == pytest.approx(1.0, abs=1e-4)
