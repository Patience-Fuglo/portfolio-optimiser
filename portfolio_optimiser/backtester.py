"""
Backtester Module
=================

Walk-forward backtesting engine with transaction cost support.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt

from portfolio_optimiser.data_loader import calculate_returns, annualize_returns, calculate_covariance
from portfolio_optimiser.costs import TransactionCostModel


Weights = npt.NDArray[np.float64]
OptimizerFunc = Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], Weights]


class PortfolioBacktester:
    """
    Walk-forward backtesting engine for portfolio strategies.

    Features:
    - Configurable rebalancing frequency
    - Transaction cost modeling
    - Benchmark comparison (equal-weight buy-and-hold)
    - Performance metrics (Sharpe, drawdown, etc.)

    Example:
        >>> backtester = PortfolioBacktester(
        ...     optimizer_func=max_sharpe_ratio,
        ...     rebalance_months=1,
        ...     cost_model=TransactionCostModel(),
        ... )
        >>> backtester.run(prices)
        >>> backtester.calculate_metrics()
        >>> backtester.print_metrics()
    """

    def __init__(
        self,
        optimizer_func: OptimizerFunc,
        rebalance_months: int = 1,
        cost_model: TransactionCostModel | None = None,
        starting_value: float = 100000,
        risk_free_rate: float = 0.02,
        lookback_days: int = 60,
    ) -> None:
        """
        Initialize the backtester.

        Args:
            optimizer_func: Function that takes (expected_returns, cov_matrix) and returns weights.
            rebalance_months: Rebalancing frequency in months.
            cost_model: Transaction cost model (optional).
            starting_value: Initial portfolio value in dollars.
            risk_free_rate: Annual risk-free rate for Sharpe calculation.
            lookback_days: Number of trailing days for parameter estimation.
        """
        self.optimizer_func = optimizer_func
        self.rebalance_months = rebalance_months
        self.cost_model = cost_model
        self.starting_value = starting_value
        self.risk_free_rate = risk_free_rate
        self.lookback_days = lookback_days

        self.portfolio_history: list[tuple[datetime, float]] = []
        self.benchmark_history: list[tuple[datetime, float]] = []
        self.portfolio_daily_returns: list[float] = []
        self.benchmark_daily_returns: list[float] = []
        self.metrics: dict[str, float] = {}
        self.benchmark_metrics: dict[str, float] = {}

    def _is_rebalance_date(self, current_date: datetime, previous_date: datetime | None) -> bool:
        """
        Check if current date is a rebalancing date.

        Args:
            current_date: Current trading date.
            previous_date: Previous rebalancing date.

        Returns:
            True if rebalancing should occur.
        """
        if previous_date is None:
            return True

        month_diff = (current_date.year - previous_date.year) * 12 + (current_date.month - previous_date.month)
        return month_diff >= self.rebalance_months

    def run(self, prices: pd.DataFrame) -> tuple[list[tuple[datetime, float]], list[tuple[datetime, float]]]:
        """
        Run the backtest.

        Args:
            prices: DataFrame of daily prices indexed by date.

        Returns:
            Tuple of (portfolio_history, benchmark_history).
        """
        prices = prices.sort_index().copy()
        returns = calculate_returns(prices)

        n_assets = prices.shape[1]
        current_weights = np.array([1 / n_assets] * n_assets)
        benchmark_weights = np.array([1 / n_assets] * n_assets)

        portfolio_value = self.starting_value
        benchmark_value = self.starting_value

        self.portfolio_history = []
        self.benchmark_history = []
        self.portfolio_daily_returns = []
        self.benchmark_daily_returns = []

        previous_date = None

        for i, date in enumerate(returns.index):
            daily_ret_vector = returns.loc[date].values

            # rebalance using only past data up to this date
            if i >= self.lookback_days and self._is_rebalance_date(date, previous_date):
                historical_returns = returns.iloc[i - self.lookback_days:i]

                expected_returns = annualize_returns(historical_returns).values
                cov_matrix = calculate_covariance(historical_returns).values

                target_weights = self.optimizer_func(expected_returns, cov_matrix)

                if self.cost_model is not None:
                    total_cost = self.cost_model.rebalance_cost(
                        current_weights,
                        target_weights,
                        portfolio_value,
                    )
                    portfolio_value -= total_cost

                current_weights = target_weights.copy()
                previous_date = date

            # portfolio daily return
            portfolio_day_return = np.dot(current_weights, daily_ret_vector)
            portfolio_value *= (1 + portfolio_day_return)

            # benchmark: equal weight buy-and-hold approximation per task
            benchmark_day_return = np.dot(benchmark_weights, daily_ret_vector)
            benchmark_value *= (1 + benchmark_day_return)

            self.portfolio_history.append((date, portfolio_value))
            self.benchmark_history.append((date, benchmark_value))
            self.portfolio_daily_returns.append(portfolio_day_return)
            self.benchmark_daily_returns.append(benchmark_day_return)

        return self.portfolio_history, self.benchmark_history

    def calculate_metrics(self):
        portfolio_values = pd.Series(
            [value for _, value in self.portfolio_history],
            index=[date for date, _ in self.portfolio_history],
        )

        benchmark_values = pd.Series(
            [value for _, value in self.benchmark_history],
            index=[date for date, _ in self.benchmark_history],
        )

        self.metrics = self._compute_metrics_from_series(portfolio_values, self.portfolio_daily_returns)
        self.benchmark_metrics = self._compute_metrics_from_series(benchmark_values, self.benchmark_daily_returns)

        return self.metrics, self.benchmark_metrics

    def _compute_metrics_from_series(self, values, daily_returns):
        initial_value = values.iloc[0]
        final_value = values.iloc[-1]
        n_days = len(daily_returns)

        total_return = (final_value / initial_value) - 1
        annualized_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0

        daily_returns = np.array(daily_returns)
        annualized_volatility = np.std(daily_returns) * np.sqrt(252)

        if annualized_volatility == 0:
            sharpe_ratio = 0
        else:
            sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_volatility

        running_max = values.cummax()
        drawdown = (values / running_max) - 1
        max_drawdown = drawdown.min()

        return {
            "Total Return": total_return,
            "Annualized Return": annualized_return,
            "Annualized Volatility": annualized_volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown,
            "Final Value": final_value,
        }

    def plot_results(self, title="Portfolio Backtest"):
        portfolio_df = pd.DataFrame(self.portfolio_history, columns=["Date", "Value"]).set_index("Date")
        benchmark_df = pd.DataFrame(self.benchmark_history, columns=["Date", "Value"]).set_index("Date")

        sharpe = self.metrics.get("Sharpe Ratio", None)
        if sharpe is not None:
            chart_title = f"{title} | Sharpe Ratio: {sharpe:.2f}"
        else:
            chart_title = title

        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_df.index, portfolio_df["Value"], label="Optimized Portfolio", linewidth=2)
        plt.plot(benchmark_df.index, benchmark_df["Value"], label="Equal-Weight Benchmark", linewidth=2, linestyle="--")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.title(chart_title)
        plt.legend()
        plt.grid(True)
        plt.savefig("outputs/backtest.png", dpi=150, bbox_inches="tight")
        plt.show()

    def print_metrics(self):
        print("\n=== OPTIMIZED PORTFOLIO METRICS ===")
        for key, value in self.metrics.items():
            if "Value" in key:
                print(f"{key}: ${value:,.2f}")
            else:
                print(f"{key}: {value:.4f} ({value * 100:.2f}%)" if key != "Sharpe Ratio" else f"{key}: {value:.4f}")

        print("\n=== BENCHMARK METRICS ===")
        for key, value in self.benchmark_metrics.items():
            if "Value" in key:
                print(f"{key}: ${value:,.2f}")
            else:
                print(f"{key}: {value:.4f} ({value * 100:.2f}%)" if key != "Sharpe Ratio" else f"{key}: {value:.4f}")
