# Portfolio Optimiser

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-3.0.0-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Version](https://img.shields.io/badge/version-3.0.0-green.svg)]()
[![Research Report](https://img.shields.io/badge/Research%20Report-Live-brightgreen.svg)](https://patience-fuglo.github.io/portfolio-optimiser/)

**[View Full Research Report](https://patience-fuglo.github.io/portfolio-optimiser/)**

A professional-grade portfolio optimization system built in Python, implementing Modern Portfolio Theory with institutional-level techniques: Ledoit-Wolf covariance shrinkage, Black-Litterman view blending, Maximum Diversification, tail-risk metrics (CVaR), and full performance attribution.

<p align="center">
  <img src="outputs/frontier_with_strategies.png" alt="Efficient Frontier with Strategies" width="650">
</p>

---

## Overview

This project implements the core quantitative portfolio management techniques used in industry and academic research:

| Feature | Description |
|---|---|
| **Efficient Frontier** | Mean-variance optimization via SLSQP |
| **6 Portfolio Strategies** | Max Sharpe, Min Variance, Equal Weight, Risk Parity, Max Diversification, HRP |
| **Hierarchical Risk Parity** | López de Prado (2016) — clustering-based, no matrix inversion |
| **Black-Litterman** | Bayesian blending of CAPM equilibrium returns with investor views |
| **Factor Risk Model** | FF3-style factor loadings, factor covariance, systematic/idio decomposition |
| **Live Data** | Fetch prices via `yfinance`; FF3 factors via `pandas-datareader` |
| **Ledoit-Wolf Shrinkage** | Well-conditioned covariance estimation for small-sample regimes |
| **Portfolio Constraints** | Per-asset weight bounds, sector exposure caps |
| **Transaction Cost Modeling** | Commission, bid-ask spread, minimum trade thresholds |
| **Walk-Forward Backtesting** | Monthly rebalancing, look-ahead bias prevention |
| **Full Metrics Suite** | Sharpe, Sortino, Calmar, CVaR (95%), Alpha, Beta, Information Ratio |
| **Risk Attribution** | Marginal risk contribution per asset, diversification ratio |

---

## Project Structure

```
portfolio_optimiser/
├── __init__.py          # Package exports (v3.0.0)
├── data_loader.py       # Price data, Ledoit-Wolf covariance, yfinance, FF3 factors
├── optimizer.py         # 5 MVO strategies + Black-Litterman + Max Diversification
├── hrp.py               # Hierarchical Risk Parity (López de Prado 2016)
├── factor_model.py      # FF3-style factor risk model
├── constraints.py       # Weight and sector constraints
├── costs.py             # Transaction cost modeling
├── backtester.py        # Walk-forward engine + full metrics + rolling Sharpe plot
└── report.py            # CVaR, risk attribution, visualizations

tests/
├── test_optimizer.py    # Efficient frontier, optimization algorithms
├── test_constraints.py  # Constraint tests
├── test_costs.py        # Transaction cost tests
├── test_backtester.py   # Backtest engine tests
├── test_report.py       # Risk attribution tests
└── test_strategies.py   # Strategy comparison tests

data/
└── sample_price_data.csv

outputs/                 # Generated visualizations
├── efficient_frontier.png
├── frontier_with_strategies.png
├── constrained_vs_unconstrained.png
├── backtest.png
├── portfolio_risk_contribution.png
└── correlation_heatmap.png

reports/
└── portfolio_research_report.html   # Self-contained research report
```

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

```bash
git clone https://github.com/Patience-Fuglo/portfolio-optimiser.git
cd portfolio-optimiser

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Run Full Analysis

```bash
python main.py
```

Pipeline:
1. Load 5-stock price data with Ledoit-Wolf covariance estimation
2. Compare all 5 strategies (Max Sharpe, Min Variance, EW, Risk Parity, Max Diversification)
3. Run Black-Litterman demo with absolute and relative views
4. Walk-forward backtest with full metric suite
5. Risk attribution report including CVaR
6. Constrained vs unconstrained comparison
7. Save all visualizations to `outputs/`

### Demo Notebook

```bash
jupyter notebook examples/demo.ipynb
```

---

## Testing

```bash
pytest                                              # full suite
pytest -v                                          # verbose
pytest --cov=portfolio_optimiser --cov-report=term-missing  # with coverage
pytest tests/test_optimizer.py                     # single module
pytest -k "sharpe"                                 # by name pattern
```

| Test File | What It Covers |
|---|---|
| `test_optimizer.py` | Efficient frontier, all 5 optimization strategies |
| `test_constraints.py` | Weight bounds, sector limits |
| `test_costs.py` | Transaction cost, turnover, net return |
| `test_backtester.py` | Walk-forward engine, metrics |
| `test_report.py` | CVaR, risk attribution, diversification ratio |
| `test_strategies.py` | Strategy comparison across regimes |

---

## Sample Visualizations

<table>
  <tr>
    <td><img src="outputs/frontier_with_strategies.png" alt="Frontier with Strategies" width="400"></td>
    <td><img src="outputs/backtest.png" alt="Backtest + Rolling Sharpe" width="400"></td>
  </tr>
  <tr>
    <td><img src="outputs/correlation_heatmap.png" alt="Correlation Heatmap" width="400"></td>
    <td><img src="outputs/portfolio_risk_contribution.png" alt="Risk Contribution" width="400"></td>
  </tr>
</table>

---

## Key Features

### 1. Ledoit-Wolf Covariance Shrinkage
The sample covariance matrix becomes ill-conditioned when history is short or asset count is large. Ledoit-Wolf analytically computes the optimal shrinkage toward a structured target, improving optimizer stability.

```python
from portfolio_optimiser import calculate_covariance_shrunk
cov = calculate_covariance_shrunk(daily_returns)
```

### 2. Portfolio Strategies

| Strategy | Objective | Key Property |
|---|---|---|
| **Max Sharpe** | Maximize (μ − r_f) / σ | Best risk-adjusted return |
| **Min Variance** | Minimize σ² | Lowest volatility |
| **Equal Weight** | w_i = 1/N | Naïve benchmark |
| **Risk Parity** | Equalize RC_i | Balanced risk exposure |
| **Max Diversification** | Maximize Σ(w·σ) / σ_p | Maximum DR benefit |
| **HRP** | Cluster then bisect | No matrix inversion |

### 3. Black-Litterman Model
Blends CAPM-implied equilibrium returns with investor views using Bayes' theorem. Supports both absolute views (asset return forecast) and relative views (asset A outperforms asset B).

```python
from portfolio_optimiser import black_litterman, max_sharpe_ratio

# P: pick matrix  |  Q: view returns
P = np.array([[1, 0, 0, 0, 0],    # absolute: AAPL at 25%
              [0, 1, -1, 0, 0]])   # relative: MSFT vs GOOGL +5%
Q = np.array([0.25, 0.05])

mu_bl = black_litterman(cov, market_weights, P, Q)
weights = max_sharpe_ratio(mu_bl, cov)
```

### 4. Backtest Metrics

| Metric | Definition |
|---|---|
| Sharpe Ratio | (R_p − r_f) / σ_p |
| **Sortino Ratio** | (R_p − r_f) / σ_downside |
| **Calmar Ratio** | R_p / \|Max Drawdown\| |
| **CVaR (95%)** | Expected loss on worst 5% of days, annualised |
| **Beta** | Cov(R_p, R_b) / Var(R_b) |
| **Alpha** | R_p − (r_f + β(R_b − r_f)) |
| **Information Ratio** | Active return / Tracking error |

### 5. Hierarchical Risk Parity (HRP)

No matrix inversion — avoids ill-conditioning entirely. Groups correlated assets by clustering, then allocates inversely to cluster variance via recursive bisection.

```python
from portfolio_optimiser import hrp_weights, plot_hrp_dendrogram

w = hrp_weights(cov_matrix, corr_matrix)
plot_hrp_dendrogram(cov_matrix, asset_names)
```

### 6. Factor Risk Model (FF3-style)

Decomposes portfolio risk into **systematic** (factor-driven) and **idiosyncratic** components. The factor-based covariance matrix `Σ_f = B·F·Bᵀ + D` can replace sample or Ledoit-Wolf covariance in any optimizer.

```python
from portfolio_optimiser import (
    fetch_ff3_factors, estimate_factor_model,
    systematic_vs_idiosyncratic, plot_factor_loadings,
)

factors = fetch_ff3_factors(start="2022-01-01")          # Mkt-RF, SMB, HML
fm = estimate_factor_model(asset_returns, factors)
decomp = systematic_vs_idiosyncratic(weights, fm)
# decomp["pct_systematic"]    → e.g. 93.4%
# decomp["pct_idiosyncratic"] → e.g.  6.6%

# Use factor covariance in optimizer
w = max_sharpe_ratio(mu, fm.factor_cov_matrix)
```

### 7. Live Data via yfinance

```python
from portfolio_optimiser import fetch_prices

prices = fetch_prices(["AAPL", "MSFT", "GOOGL"], start="2022-01-01")
```

### 8. CVaR / Expected Shortfall
More sensitive to tail risk than VaR; required under Basel III for institutional risk reporting.

```python
from portfolio_optimiser import cvar
daily_cvar, annualized_cvar = cvar(daily_returns, confidence=0.95)
```

---

## Sample Results

### Strategy Comparison (Ledoit-Wolf covariance, 2% risk-free rate)

```
Strategy            Return   Volatility   Sharpe
Max Sharpe          0.3981       0.1670   2.2643
Min Variance        0.2856       0.1423   1.8675
Equal Weight        0.3833       0.2392   1.5188
Risk Parity         0.3245       0.1856   1.6392
Max Diversification 0.3102       0.1590   1.8250
```

### Backtest Metrics (Max Sharpe, monthly rebalancing, $100k initial)

```
Annualized Return    32.16%
Annualized Volatility 20.51%
Sharpe Ratio         1.4704
Sortino Ratio        2.2533
Calmar Ratio         3.0713
Max Drawdown        -10.47%
CVaR (95%)          43.92%
Beta                 0.8252
Alpha               -5.32%
Information Ratio   -1.3541
Final Value         $130,207
```

### Risk Attribution (Max Sharpe Portfolio)

```
Asset      Weight   Exp Return   Volatility   Risk %
AAPL       0.xxxx       0.xxxx       0.xxxx   xx.xx%
...
Diversification Ratio: 1.13
CVaR (95%, annualised): 37.10%
```

---

## Configuration

### Black-Litterman

```python
mu_bl = black_litterman(
    cov_matrix,            # Annualized covariance (Ledoit-Wolf recommended)
    market_weights,        # Equilibrium reference weights
    view_picks,            # P matrix (k × n): absolute or relative views
    view_returns,          # Q vector (k,): expected return per view
    tau=0.025,             # Prior uncertainty (typically 0.01–0.05)
    risk_aversion=2.5,     # Market risk aversion δ
)
```

### Backtester

```python
backtester = PortfolioBacktester(
    optimizer_func=optimizer_wrapper,
    rebalance_months=1,
    cost_model=TransactionCostModel(commission_rate=0.001, spread_cost=0.0005),
    starting_value=100_000,
    risk_free_rate=0.02,
    lookback_days=60,
)
backtester.run(prices)
backtester.calculate_metrics()
backtester.print_metrics()
backtester.plot_results(title="Max Sharpe Backtest", rolling_window=60)
```

### Constraints

```python
constraints = PortfolioConstraints(
    max_weight=0.30,
    min_weight=0.00,
    sector_limits={"Tech": 0.70},
)
```

---

## Technical Details

### Optimization Method
- **Algorithm**: Sequential Least Squares Programming (SLSQP) via `scipy.optimize.minimize`
- **Convergence**: All optimizers raise `RuntimeError` on failed convergence
- **Covariance**: Ledoit-Wolf shrinkage via `sklearn.covariance.LedoitWolf`

### Black-Litterman Posterior
```
π  = δ × Σ × w_market                     (equilibrium returns)
Ω  = τ × P × Σ × Pᵀ                       (Meucci proportional uncertainty)
μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ [(τΣ)⁻¹π + PᵀΩ⁻¹Q]
```

### Risk Contribution
```
RC_i = w_i × (Σ w)_i / σ_p
```

### Diversification Ratio
```
DR = Σ(w_i × σ_i) / σ_p      (> 1 means diversification benefit)
```

### CVaR (Expected Shortfall)
```
CVaR_α = -E[R | R ≤ VaR_α]   annualised as daily_CVaR × √252
```

---

## Technologies

| Category | Libraries / Concepts |
|---|---|
| **Quantitative Finance** | MPT, Mean-Variance Optimization, Black-Litterman, Risk Parity, CVaR |
| **Mathematical Optimization** | Constrained Optimization, SLSQP, Convex Optimization |
| **Statistics** | Covariance Shrinkage, Tail-Risk Estimation, Performance Attribution |
| **Python** | NumPy, Pandas, SciPy, Matplotlib, scikit-learn, yfinance, pandas-datareader |

---

## Author

**Patience Fuglo**

- GitHub: [@Patience-Fuglo](https://github.com/Patience-Fuglo)
- LinkedIn: [Patience Fuglo](https://www.linkedin.com/in/patience-fuglo/)

---

<p align="center">Star this repo if you find it useful!</p>
