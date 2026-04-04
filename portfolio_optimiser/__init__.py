"""
Portfolio Optimiser
===================

A portfolio optimization library implementing Modern Portfolio Theory
with constraints, transaction costs, backtesting, and risk attribution.

Modules
-------
- data_loader: Price data loading, returns calculation, covariance estimation
- optimizer: Portfolio optimization strategies (Max Sharpe, Min Variance, Risk Parity)
- constraints: Position limits and sector exposure constraints
- costs: Transaction cost modeling
- backtester: Walk-forward backtesting engine
- report: Risk attribution and visualization

Example
-------
>>> from portfolio_optimiser import load_prices, calculate_returns, max_sharpe_ratio
>>> prices = load_prices("data/sample_price_data.csv")
>>> returns = calculate_returns(prices)
>>> weights = max_sharpe_ratio(returns.mean().values * 252, returns.cov().values * 252)
"""

__version__ = "1.0.0"
__author__ = "Patience Fuglo"

# Data loading utilities
from portfolio_optimiser.data_loader import (
    load_prices,
    calculate_returns,
    annualize_returns,
    calculate_covariance,
    calculate_correlation,
    calculate_annual_volatility,
)

# Optimization functions
from portfolio_optimiser.optimizer import (
    portfolio_return,
    portfolio_volatility,
    efficient_frontier,
    max_sharpe_ratio,
    min_variance,
    equal_weight,
    risk_parity,
    maximize_sharpe,
    compare_strategies,
)

# Constraints
from portfolio_optimiser.constraints import PortfolioConstraints

# Transaction costs
from portfolio_optimiser.costs import TransactionCostModel

# Backtesting
from portfolio_optimiser.backtester import PortfolioBacktester

# Reporting
from portfolio_optimiser.report import (
    risk_contribution,
    pct_risk_contribution,
    diversification_ratio,
    print_portfolio_summary,
    plot_risk_pie,
    plot_correlation_heatmap,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Data loader
    "load_prices",
    "calculate_returns",
    "annualize_returns",
    "calculate_covariance",
    "calculate_correlation",
    "calculate_annual_volatility",
    # Optimizer
    "portfolio_return",
    "portfolio_volatility",
    "efficient_frontier",
    "max_sharpe_ratio",
    "min_variance",
    "equal_weight",
    "risk_parity",
    "maximize_sharpe",
    "compare_strategies",
    # Constraints
    "PortfolioConstraints",
    # Costs
    "TransactionCostModel",
    # Backtester
    "PortfolioBacktester",
    # Report
    "risk_contribution",
    "pct_risk_contribution",
    "diversification_ratio",
    "print_portfolio_summary",
    "plot_risk_pie",
    "plot_correlation_heatmap",
]
