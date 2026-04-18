"""
Transaction Costs Module
========================

Model transaction costs including commissions, spreads, and minimum trade sizes.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


Weights = npt.NDArray[np.float64]


class TransactionCostModel:
    """
    Model transaction costs for portfolio rebalancing.

    Includes:
    - Commission rate (broker fee)
    - Bid-ask spread cost
    - Minimum commission per trade

    Example:
        >>> cost_model = TransactionCostModel(commission_rate=0.001, spread_cost=0.0005)
        >>> cost = cost_model.rebalance_cost(old_weights, new_weights, portfolio_value=100000)
    """

    def __init__(
        self,
        commission_rate: float = 0.001,
        spread_cost: float = 0.0005,
        min_commission: float = 1.0,
    ) -> None:
        """
        Initialize transaction cost model.

        Args:
            commission_rate: Broker commission as fraction (e.g., 0.001 = 0.1%).
            spread_cost: Bid-ask spread cost as fraction (e.g., 0.0005 = 0.05%).
            min_commission: Minimum commission per trade in dollars.
        """
        self.commission_rate = commission_rate
        self.spread_cost = spread_cost
        self.min_commission = min_commission

    def trade_cost(self, trade_value: float) -> float:
        """
        Calculate cost of a single trade.

        Args:
            trade_value: Absolute dollar value of the trade.

        Returns:
            Transaction cost in dollars (always positive), or 0 if no trade.
        """
        if abs(trade_value) < 1e-8:
            return 0.0
        variable_cost = abs(trade_value) * (self.commission_rate + self.spread_cost)
        return max(variable_cost, self.min_commission)

    def rebalance_cost(
        self,
        current_weights: Weights,
        target_weights: Weights,
        portfolio_value: float,
    ) -> float:
        """
        Calculate total cost to rebalance portfolio.

        Args:
            current_weights: Current portfolio weights.
            target_weights: Target portfolio weights after rebalancing.
            portfolio_value: Total portfolio value in dollars.

        Returns:
            Total transaction cost in dollars.
        """
        total_cost = 0.0

        for current_weight, target_weight in zip(current_weights, target_weights):
            trade_value = abs(target_weight - current_weight) * portfolio_value
            total_cost += self.trade_cost(trade_value)

        return total_cost

    def turnover(self, current_weights: Weights, target_weights: Weights) -> float:
        """
        Calculate portfolio turnover.

        Turnover is the fraction of the portfolio that changes hands.
        Divided by 2 so buys and sells are not double-counted.

        Args:
            current_weights: Current portfolio weights.
            target_weights: Target portfolio weights.

        Returns:
            Turnover as a fraction (0 to 1).
        """
        return sum(abs(target - current) for current, target in zip(current_weights, target_weights)) / 2

    def net_return(
        self,
        gross_return: float,
        total_cost: float,
        portfolio_value: float,
    ) -> float:
        """
        Calculate net return after transaction costs.

        Args:
            gross_return: Gross return as a fraction.
            total_cost: Total transaction costs in dollars.
            portfolio_value: Portfolio value in dollars.

        Returns:
            Net return as a fraction.
        """
        return gross_return - (total_cost / portfolio_value)
