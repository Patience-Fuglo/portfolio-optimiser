"""
Portfolio Optimiser
===================

A portfolio optimization library implementing Modern Portfolio Theory
with constraints, transaction costs, backtesting, and risk attribution.

Modules
-------
- data_loader  : Price data loading, returns, Ledoit-Wolf covariance, live data (yfinance/FF3)
- optimizer    : Max Sharpe, Min Variance, Equal Weight, Risk Parity, Max Diversification,
                 Black-Litterman
- hrp          : Hierarchical Risk Parity (López de Prado 2016)
- factor_model : FF3 factor risk model — loadings, factor covariance, systematic/idio decomp
- constraints  : Per-asset weight bounds and sector exposure caps
- costs        : Transaction cost modeling
- backtester   : Walk-forward engine with Sharpe/Sortino/Calmar/CVaR/Alpha/Beta/IR
- report       : CVaR, risk attribution, correlation heatmap

Example
-------
>>> from portfolio_optimiser import load_prices, calculate_returns, max_sharpe_ratio
>>> prices = load_prices("data/sample_price_data.csv")
>>> returns = calculate_returns(prices)
>>> weights = max_sharpe_ratio(returns.mean().values * 252, returns.cov().values * 252)
"""

__version__ = "3.0.0"
__author__ = "Patience Fuglo"

# Data loading utilities
from portfolio_optimiser.data_loader import (
    load_prices,
    fetch_prices,
    fetch_ff3_factors,
    calculate_returns,
    annualize_returns,
    calculate_covariance,
    calculate_covariance_shrunk,
    calculate_correlation,
    calculate_annual_volatility,
)

# Hierarchical Risk Parity
from portfolio_optimiser.hrp import hrp_weights, plot_hrp_dendrogram

# Factor Risk Model
from portfolio_optimiser.factor_model import (
    estimate_factor_model,
    systematic_vs_idiosyncratic,
    print_factor_model_summary,
    plot_factor_loadings,
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
    maximum_diversification,
    black_litterman,
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
    cvar,
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
    "fetch_prices",
    "fetch_ff3_factors",
    "calculate_returns",
    "annualize_returns",
    "calculate_covariance",
    "calculate_covariance_shrunk",
    "calculate_correlation",
    "calculate_annual_volatility",
    # HRP
    "hrp_weights",
    "plot_hrp_dendrogram",
    # Factor model
    "estimate_factor_model",
    "systematic_vs_idiosyncratic",
    "print_factor_model_summary",
    "plot_factor_loadings",
    # Optimizer
    "portfolio_return",
    "portfolio_volatility",
    "efficient_frontier",
    "max_sharpe_ratio",
    "min_variance",
    "equal_weight",
    "risk_parity",
    "maximum_diversification",
    "black_litterman",
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
    "cvar",
    "print_portfolio_summary",
    "plot_risk_pie",
    "plot_correlation_heatmap",
]
