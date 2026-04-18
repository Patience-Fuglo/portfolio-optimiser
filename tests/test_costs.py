"""Tests for TransactionCostModel: commission, turnover, net return."""

import pytest

from portfolio_optimiser.costs import TransactionCostModel


@pytest.fixture
def model():
    return TransactionCostModel(commission_rate=0.001, spread_cost=0.0005, min_commission=1.0)


def test_zero_trade_no_cost(model):
    assert model.trade_cost(0.0) == 0.0


def test_zero_trade_below_epsilon(model):
    assert model.trade_cost(1e-10) == 0.0


def test_trade_cost_variable(model):
    cost = model.trade_cost(10_000)
    expected = 10_000 * (0.001 + 0.0005)
    assert cost == pytest.approx(expected, rel=1e-6)


def test_trade_cost_min_commission_enforced(model):
    cost = model.trade_cost(10)  # variable = 10 * 0.0015 = 0.015 < min_commission=1.0
    assert cost == pytest.approx(1.0)


def test_turnover_no_change(model):
    w = [0.25, 0.25, 0.25, 0.25]
    assert model.turnover(w, w) == pytest.approx(0.0, abs=1e-8)


def test_turnover_full_rebalance(model):
    current = [1.0, 0.0, 0.0, 0.0]
    target = [0.0, 1.0, 0.0, 0.0]
    assert model.turnover(current, target) == pytest.approx(1.0, abs=1e-6)


def test_rebalance_cost_positive(model):
    current = [0.25, 0.25, 0.25, 0.25]
    target = [0.40, 0.30, 0.20, 0.10]
    cost = model.rebalance_cost(current, target, portfolio_value=100_000)
    assert cost > 0


def test_net_return_less_than_gross(model):
    current = [0.25, 0.25, 0.25, 0.25]
    target = [0.40, 0.30, 0.20, 0.10]
    gross = 0.12
    cost = model.rebalance_cost(current, target, portfolio_value=100_000)
    net = model.net_return(gross, cost, portfolio_value=100_000)
    assert net < gross


def test_net_return_formula(model):
    gross = 0.10
    cost = 50.0
    pv = 100_000
    expected = gross - cost / pv
    assert model.net_return(gross, cost, pv) == pytest.approx(expected, rel=1e-8)
