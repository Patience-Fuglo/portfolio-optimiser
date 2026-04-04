"""
Data Loader Module
==================

Functions for loading price data and calculating returns, covariance, and correlations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS: int = 252


def load_prices(filepath: str) -> pd.DataFrame:
    """
    Load price data from a CSV file.

    Args:
        filepath: Path to CSV file with columns [Date, TICKER1, TICKER2, ...]

    Returns:
        DataFrame indexed by Date with stock prices as columns.

    Example:
        >>> prices = load_prices("data/sample_price_data.csv")
        >>> prices.head()
    """
    prices = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
    prices = prices.sort_index()
    return prices


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily percentage returns from price data.

    Args:
        prices: DataFrame of daily prices indexed by date.

    Returns:
        DataFrame of daily returns (e.g., 0.02 means +2% that day).

    Example:
        >>> returns = calculate_returns(prices)
        >>> returns.mean()  # Average daily returns
    """
    returns = prices.pct_change().dropna()
    return returns


def annualize_returns(daily_returns: pd.DataFrame) -> pd.Series:
    """
    Convert mean daily returns to annualized expected returns.

    Args:
        daily_returns: DataFrame of daily returns.

    Returns:
        Series of annualized returns (assuming 252 trading days).
    """
    return daily_returns.mean() * TRADING_DAYS


def calculate_covariance(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the annualized covariance matrix.

    Args:
        daily_returns: DataFrame of daily returns.

    Returns:
        Annualized covariance matrix as DataFrame.
    """
    return daily_returns.cov() * TRADING_DAYS


def calculate_correlation(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the correlation matrix.

    Args:
        daily_returns: DataFrame of daily returns.

    Returns:
        Correlation matrix as DataFrame.
    """
    return daily_returns.corr()


def calculate_annual_volatility(daily_returns: pd.DataFrame) -> pd.Series:
    """
    Calculate annualized volatility for each asset.

    Args:
        daily_returns: DataFrame of daily returns.

    Returns:
        Series of annualized volatilities.
    """
    return daily_returns.std() * np.sqrt(TRADING_DAYS)


def print_summary(returns: pd.DataFrame) -> None:
    """
    Print:
    - annual return for each stock
    - annual volatility for each stock
    - full correlation matrix
    """
    annual_returns = annualize_returns(returns)
    annual_volatility = calculate_annual_volatility(returns)
    correlation_matrix = calculate_correlation(returns)

    print("\n=== PORTFOLIO SUMMARY ===\n")

    for stock in returns.columns:
        print(f"{stock}:")
        print(f"  Average Annual Return: {annual_returns[stock]:.4f} ({annual_returns[stock] * 100:.2f}%)")
        print(f"  Annual Volatility:     {annual_volatility[stock]:.4f} ({annual_volatility[stock] * 100:.2f}%)")
        print()

    print("=== CORRELATION MATRIX ===")
    print(correlation_matrix.round(4))


def find_highest_return_stock(returns: pd.DataFrame) -> tuple[str, float]:
    """
    Return the stock with the highest annualized return.
    """
    annual_returns = annualize_returns(returns)
    best_stock = annual_returns.idxmax()
    best_value = annual_returns.max()
    return best_stock, best_value


def find_lowest_correlation_pair(returns: pd.DataFrame) -> tuple[tuple[str, str], float]:
    """
    Return the pair of stocks with the lowest correlation.
    This can suggest the best diversification pair.
    """
    corr = calculate_correlation(returns)

    lowest_pair = None
    lowest_value = float("inf")
    columns = corr.columns

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            value = corr.iloc[i, j]
            if value < lowest_value:
                lowest_value = value
                lowest_pair = (columns[i], columns[j])

    return lowest_pair, lowest_value


if __name__ == "__main__":
    filepath = "data/sample_price_data.csv"

    prices = load_prices(filepath)
    returns = calculate_returns(prices)

    print_summary(returns)

    best_stock, best_return = find_highest_return_stock(returns)
    best_pair, lowest_corr = find_lowest_correlation_pair(returns)

    print("\n=== ANSWERS ===")
    print(f"Highest return stock: {best_stock} ({best_return * 100:.2f}%)")
    print(f"Lowest correlation pair: {best_pair[0]} and {best_pair[1]} ({lowest_corr:.4f})")