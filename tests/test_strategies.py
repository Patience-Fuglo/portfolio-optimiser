"""Tests for all 6 strategies, HRP, and Factor Risk Model."""

import numpy as np
import pytest

from portfolio_optimiser.optimizer import (
    max_sharpe_ratio, min_variance, equal_weight, risk_parity,
    maximum_diversification, portfolio_volatility, portfolio_return,
)
from portfolio_optimiser.hrp import hrp_weights
from portfolio_optimiser.factor_model import (
    estimate_factor_model, systematic_vs_idiosyncratic, print_factor_model_summary,
)
from portfolio_optimiser.data_loader import calculate_annual_volatility
import pandas as pd


# ── HRP ──────────────────────────────────────────────────────────────────────

def test_hrp_weights_sum(cov_matrix, sample_returns):
    corr = np.corrcoef(sample_returns.values, rowvar=False)
    w = hrp_weights(cov_matrix, corr)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_hrp_weights_non_negative(cov_matrix, sample_returns):
    corr = np.corrcoef(sample_returns.values, rowvar=False)
    w = hrp_weights(cov_matrix, corr)
    assert np.all(w >= -1e-8)


def test_hrp_without_corr_arg(cov_matrix):
    w = hrp_weights(cov_matrix)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_hrp_diversified(cov_matrix, sample_returns, n_assets):
    corr = np.corrcoef(sample_returns.values, rowvar=False)
    w = hrp_weights(cov_matrix, corr)
    # HRP should not concentrate everything in one asset
    assert w.max() < 0.80


# ── Strategy volatility ordering ─────────────────────────────────────────────

def test_min_variance_lower_vol_than_equal_weight(expected_returns, cov_matrix, n_assets):
    w_mv = min_variance(cov_matrix)
    w_ew = equal_weight(n_assets)
    assert portfolio_volatility(w_mv, cov_matrix) <= portfolio_volatility(w_ew, cov_matrix) + 1e-4


def test_max_sharpe_highest_sharpe(expected_returns, cov_matrix, n_assets):
    rfr = 0.02
    w_ms = max_sharpe_ratio(expected_returns, cov_matrix, risk_free_rate=rfr)
    candidates = [
        min_variance(cov_matrix),
        equal_weight(n_assets),
        risk_parity(cov_matrix),
        maximum_diversification(cov_matrix),
    ]
    sharpe_ms = (portfolio_return(w_ms, expected_returns) - rfr) / portfolio_volatility(w_ms, cov_matrix)
    for w in candidates:
        s = (portfolio_return(w, expected_returns) - rfr) / portfolio_volatility(w, cov_matrix)
        assert sharpe_ms >= s - 1e-4


# ── Factor Risk Model ─────────────────────────────────────────────────────────

@pytest.fixture
def factor_model(sample_returns):
    mkt = sample_returns.mean(axis=1)
    ranked = sample_returns.rank(axis=1)
    n = len(sample_returns.columns)
    smb = (sample_returns.loc[:, ranked.iloc[-1] <= n // 2].mean(axis=1)
           - sample_returns.loc[:, ranked.iloc[-1] > n // 2].mean(axis=1))
    vol_rank = calculate_annual_volatility(sample_returns)
    hml = (sample_returns[vol_rank.nsmallest(max(1, n // 2)).index].mean(axis=1)
           - sample_returns[vol_rank.nlargest(max(1, n // 2)).index].mean(axis=1))
    factors = pd.DataFrame({"Mkt-RF": mkt, "SMB": smb, "HML": hml}, index=sample_returns.index).dropna()
    return estimate_factor_model(sample_returns, factors)


def test_factor_model_loadings_shape(factor_model, n_assets):
    assert factor_model.loadings.shape == (n_assets, 3)


def test_factor_model_r_squared_range(factor_model):
    assert np.all(factor_model.r_squared >= 0)
    assert np.all(factor_model.r_squared <= 1.0 + 1e-6)


def test_factor_cov_matrix_symmetric(factor_model, n_assets):
    F = factor_model.factor_cov_matrix
    assert F.shape == (n_assets, n_assets)
    assert np.allclose(F, F.T, atol=1e-8)


def test_factor_cov_matrix_psd(factor_model):
    eigvals = np.linalg.eigvalsh(factor_model.factor_cov_matrix)
    assert np.all(eigvals >= -1e-8)


def test_systematic_idio_sums_to_100(factor_model, expected_returns, cov_matrix):
    w = max_sharpe_ratio(expected_returns, cov_matrix, risk_free_rate=0.02)
    decomp = systematic_vs_idiosyncratic(w, factor_model)
    total = decomp["pct_systematic"] + decomp["pct_idiosyncratic"]
    assert total == pytest.approx(100.0, abs=0.1)


def test_systematic_pct_positive(factor_model, expected_returns, cov_matrix):
    w = max_sharpe_ratio(expected_returns, cov_matrix, risk_free_rate=0.02)
    decomp = systematic_vs_idiosyncratic(w, factor_model)
    assert decomp["pct_systematic"] > 0
    assert decomp["pct_idiosyncratic"] > 0


def test_total_volatility_positive(factor_model, expected_returns, cov_matrix):
    w = max_sharpe_ratio(expected_returns, cov_matrix, risk_free_rate=0.02)
    decomp = systematic_vs_idiosyncratic(w, factor_model)
    assert decomp["total_volatility"] > 0
