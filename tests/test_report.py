import sys; sys.path.insert(0, ".")
from portfolio_optimiser.data_loader import (
    load_prices,
    calculate_returns,
    annualize_returns,
    calculate_covariance,
    calculate_correlation,
)
from portfolio_optimiser.optimizer import max_sharpe_ratio
from portfolio_optimiser.report import (
    print_portfolio_summary,
    plot_risk_pie,
    plot_correlation_heatmap,
)


def main():
    filepath = "data/sample_price_data.csv"

    prices = load_prices(filepath)
    prices = prices[["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]]

    returns = calculate_returns(prices)

    expected_returns = annualize_returns(returns).values
    cov_matrix = calculate_covariance(returns).values
    corr_matrix = calculate_correlation(returns).values
    asset_names = returns.columns.tolist()

    weights = max_sharpe_ratio(expected_returns, cov_matrix, risk_free_rate=0.02)

    report = print_portfolio_summary(
        weights,
        expected_returns,
        cov_matrix,
        asset_names,
    )

    pct_contributions = report["pct_risk_contributions"]

    max_risk_idx = pct_contributions.argmax()
    print(f"\nBiggest Risk Contributor: {asset_names[max_risk_idx]} ({pct_contributions[max_risk_idx]:.2f}%)")

    if report["diversification_ratio"] > 1:
        print("Diversification Ratio is above 1: diversification is reducing portfolio risk.")
    else:
        print("Diversification Ratio is not above 1: limited diversification benefit.")

    plot_risk_pie(asset_names, pct_contributions)
    plot_correlation_heatmap(corr_matrix, asset_names)


if __name__ == "__main__":
    main()
