"""
Constraints Module
==================

Portfolio constraints for position limits and sector exposure.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


class PortfolioConstraints:
    """
    Portfolio constraints for optimization.

    Supports:
    - Per-asset weight bounds (min/max)
    - Sector exposure limits

    Example:
        >>> constraints = PortfolioConstraints(max_weight=0.30, min_weight=0.05)
        >>> constraints.get_bounds(5)  # For 5 assets
        [(0.05, 0.30), (0.05, 0.30), ...]
    """

    def __init__(
        self,
        max_weight: float | None = None,
        min_weight: float | None = None,
        sector_limits: dict[str, float] | None = None,
    ) -> None:
        """
        Initialize portfolio constraints.

        Args:
            max_weight: Maximum weight per asset (e.g., 0.30 for 30%).
            min_weight: Minimum weight per asset (e.g., 0.05 for 5%).
            sector_limits: Dict mapping sector names to max exposure (e.g., {'Tech': 0.50}).
        """
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.sector_limits = sector_limits or {}

    def get_bounds(self, n_assets: int) -> list[tuple[float, float]]:
        """
        Return weight bounds for scipy optimizer.

        Args:
            n_assets: Number of assets in the portfolio.

        Returns:
            List of (min, max) tuples for each asset.
        """
        min_w = 0.0 if self.min_weight is None else self.min_weight
        max_w = 1.0 if self.max_weight is None else self.max_weight
        return [(min_w, max_w) for _ in range(n_assets)]

    def get_sector_constraint(
        self,
        asset_sectors: list[str],
        sector: str,
        limit: float,
    ) -> dict[str, Any]:
        """
        Create scipy inequality constraint for one sector.

        Args:
            asset_sectors: List of sector labels for each asset.
            sector: Sector name to constrain.
            limit: Maximum allowed weight for this sector.

        Returns:
            SciPy constraint dict: total sector weight <= limit.
        """
        sector_indices = [i for i, s in enumerate(asset_sectors) if s == sector]

        return {
            "type": "ineq",
            "fun": lambda w, idx=sector_indices, lim=limit: lim - np.sum([w[i] for i in idx])
        }

    def get_all_constraints(
        self,
        asset_sectors: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get all constraints for scipy optimizer.

        Args:
            asset_sectors: Optional list of sector labels for each asset.

        Returns:
            List of constraint dicts (weights sum to 1, plus sector limits).
        """
        constraints: list[dict[str, Any]] = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        ]

        if asset_sectors is not None:
            for sector, limit in self.sector_limits.items():
                constraints.append(self.get_sector_constraint(asset_sectors, sector, limit))

        return constraints
