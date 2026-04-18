"""
Pytest configuration and fixtures for Portfolio Optimiser tests.
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from portfolio_optimiser.data_loader import load_prices, calculate_returns, annualize_returns, calculate_covariance


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Load sample price data for testing."""
    filepath = project_root / "data" / "sample_price_data.csv"
    prices = load_prices(str(filepath))
    return prices[["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]]


@pytest.fixture
def sample_returns(sample_prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate returns from sample prices."""
    return calculate_returns(sample_prices)


@pytest.fixture
def expected_returns(sample_returns: pd.DataFrame) -> np.ndarray:
    """Annualized expected returns."""
    return annualize_returns(sample_returns).values


@pytest.fixture
def cov_matrix(sample_returns: pd.DataFrame) -> np.ndarray:
    """Annualized covariance matrix."""
    return calculate_covariance(sample_returns).values


@pytest.fixture
def stock_names(sample_returns: pd.DataFrame) -> list[str]:
    """List of stock ticker symbols."""
    return sample_returns.columns.tolist()


@pytest.fixture
def n_assets(stock_names: list[str]) -> int:
    """Number of assets."""
    return len(stock_names)
