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

            # benchmark: equal-weight continuously rebalanced
            benchmark_day_return = np.dot(benchmark_weights, daily_ret_vector)
            benchmark_value *= (1 + benchmark_day_return)

            self.portfolio_history.append((date, portfolio_value))
            self.benchmark_history.append((date, benchmark_value))
            self.portfolio_daily_returns.append(portfolio_day_return)
            self.benchmark_daily_returns.append(benchmark_day_return)

        return self.portfolio_history, self.benchmark_history

    def calculate_metrics(self) -> tuple[dict, dict]:
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

        # Comparative metrics require both series
        port_arr = np.array(self.portfolio_daily_returns)
        bench_arr = np.array(self.benchmark_daily_returns)

        bench_var = np.var(bench_arr, ddof=1)
        cov_pb = np.cov(port_arr, bench_arr)[0, 1]
        beta = cov_pb / bench_var if bench_var > 1e-10 else 0.0

        alpha = self.metrics["Annualized Return"] - (
            self.risk_free_rate
            + beta * (self.benchmark_metrics["Annualized Return"] - self.risk_free_rate)
        )

        active_daily = port_arr - bench_arr
        tracking_error = np.std(active_daily, ddof=1) * np.sqrt(252)
        active_return = self.metrics["Annualized Return"] - self.benchmark_metrics["Annualized Return"]
        information_ratio = active_return / tracking_error if tracking_error > 1e-10 else 0.0

        self.metrics["Beta"] = beta
        self.metrics["Alpha"] = alpha
        self.metrics["Information Ratio"] = information_ratio
        self.metrics["Tracking Error"] = tracking_error

        return self.metrics, self.benchmark_metrics

    def _compute_metrics_from_series(self, values: pd.Series, daily_returns: list[float]) -> dict:
        initial_value = values.iloc[0]
        final_value = values.iloc[-1]
        n_days = len(daily_returns)

        total_return = (final_value / initial_value) - 1
        annualized_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0.0

        r = np.array(daily_returns)
        annualized_volatility = np.std(r, ddof=1) * np.sqrt(252)

        sharpe_ratio = (
            (annualized_return - self.risk_free_rate) / annualized_volatility
            if annualized_volatility > 1e-10 else 0.0
        )

        # Sortino: penalise only downside returns
        downside = r[r < 0]
        downside_vol = np.std(downside, ddof=1) * np.sqrt(252) if len(downside) > 1 else 0.0
        sortino_ratio = (
            (annualized_return - self.risk_free_rate) / downside_vol
            if downside_vol > 1e-10 else 0.0
        )

        running_max = values.cummax()
        drawdown = (values / running_max) - 1
        max_drawdown = drawdown.min()

        # Calmar: annualised return / absolute max drawdown
        calmar_ratio = annualized_return / abs(max_drawdown) if abs(max_drawdown) > 1e-10 else 0.0

        # CVaR (Expected Shortfall) at 95% — expected loss on the worst 5% of days
        sorted_r = np.sort(r)
        cutoff = max(1, int(len(sorted_r) * 0.05))
        daily_cvar = float(-np.mean(sorted_r[:cutoff]))
        annualized_cvar = daily_cvar * np.sqrt(252)

        return {
            "Total Return": total_return,
            "Annualized Return": annualized_return,
            "Annualized Volatility": annualized_volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Sortino Ratio": sortino_ratio,
            "Calmar Ratio": calmar_ratio,
            "Max Drawdown": max_drawdown,
            "CVaR (95%)": annualized_cvar,
            "Final Value": final_value,
        }

    def plot_results(self, title: str = "Portfolio Backtest", rolling_window: int = 60) -> None:
        portfolio_df = pd.DataFrame(self.portfolio_history, columns=["Date", "Value"]).set_index("Date")
        benchmark_df = pd.DataFrame(self.benchmark_history, columns=["Date", "Value"]).set_index("Date")

        sharpe = self.metrics.get("Sharpe Ratio")
        sortino = self.metrics.get("Sortino Ratio")
        header = f"{title}"
        if sharpe is not None and sortino is not None:
            header += f"  |  Sharpe: {sharpe:.2f}  Sortino: {sortino:.2f}"

        dates = [d for d, _ in self.portfolio_history]
        port_ret_series = pd.Series(self.portfolio_daily_returns, index=dates)
        rolling_sharpe = (
            port_ret_series.rolling(rolling_window).mean() * 252 - self.risk_free_rate
        ) / (port_ret_series.rolling(rolling_window).std(ddof=1) * np.sqrt(252))

        fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})

        ax1 = axes[0]
        ax1.plot(portfolio_df.index, portfolio_df["Value"], label="Optimized Portfolio", linewidth=2)
        ax1.plot(benchmark_df.index, benchmark_df["Value"],
                 label="EW Rebalanced Benchmark", linewidth=2, linestyle="--")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.set_title(header)
        ax1.legend()
        ax1.grid(True)

        ax2 = axes[1]
        ax2.plot(rolling_sharpe.index, rolling_sharpe,
                 label=f"{rolling_window}-Day Rolling Sharpe", color="steelblue", linewidth=1.5)
        ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax2.set_ylabel("Rolling Sharpe")
        ax2.set_xlabel("Date")
        ax2.legend(fontsize=9)
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig("outputs/backtest.png", dpi=150, bbox_inches="tight")
        plt.show()

    def print_metrics(self) -> None:
        _RATIO_KEYS = {"Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
                       "Information Ratio", "Beta"}
        _VALUE_KEYS = {"Final Value"}

        def _fmt(key: str, value: float) -> str:
            if key in _VALUE_KEYS:
                return f"${value:,.2f}"
            if key in _RATIO_KEYS:
                return f"{value:.4f}"
            return f"{value:.4f}  ({value * 100:.2f}%)"

        print("\n=== OPTIMIZED PORTFOLIO METRICS ===")
        for key, value in self.metrics.items():
            print(f"  {key:<22} {_fmt(key, value)}")

        print("\n=== BENCHMARK METRICS (EW Rebalanced) ===")
        bench_display = {k: v for k, v in self.benchmark_metrics.items()
                         if k not in {"Beta", "Alpha", "Information Ratio", "Tracking Error"}}
        for key, value in bench_display.items():
            print(f"  {key:<22} {_fmt(key, value)}")
