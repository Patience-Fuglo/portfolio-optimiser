import numpy as np
import matplotlib.pyplot as plt

from portfolio_optimiser.optimizer import portfolio_volatility, portfolio_return


def risk_contribution(weights, cov_matrix):
    """
    Risk contribution of each asset:
    RC_i = w_i * (cov_matrix @ weights)[i] / portfolio_vol
    """
    portfolio_vol = portfolio_volatility(weights, cov_matrix)

    if portfolio_vol == 0:
        return np.zeros(len(weights))

    marginal_contrib = np.dot(cov_matrix, weights)
    rc = weights * marginal_contrib / portfolio_vol
    return rc


def pct_risk_contribution(risk_contributions, portfolio_vol):
    """
    Convert risk contributions into percentage of total portfolio risk.
    They should sum to about 100%.
    """
    if portfolio_vol == 0:
        return np.zeros(len(risk_contributions))

    return (risk_contributions / portfolio_vol) * 100


def diversification_ratio(weights, individual_vols, portfolio_vol):
    """
    Diversification Ratio = sum(weight_i * vol_i) / portfolio_vol
    Ratio > 1 means diversification benefit exists.
    """
    if portfolio_vol == 0:
        return 0

    return np.sum(weights * individual_vols) / portfolio_vol


def print_portfolio_summary(weights, expected_returns, cov_matrix, asset_names):
    """
    Print portfolio summary table:
    Asset, Weight, Expected Return, Volatility, Risk Contribution %
    """
    portfolio_vol = portfolio_volatility(weights, cov_matrix)
    portfolio_ret = portfolio_return(weights, expected_returns)
    individual_vols = np.sqrt(np.diag(cov_matrix))

    rc = risk_contribution(weights, cov_matrix)
    pct_rc = pct_risk_contribution(rc, portfolio_vol)
    div_ratio = diversification_ratio(weights, individual_vols, portfolio_vol)

    print("\n=== PORTFOLIO RISK REPORT ===\n")
    print(f"{'Asset':<10} {'Weight':>10} {'Exp Return':>12} {'Volatility':>12} {'Risk %':>12}")
    print("-" * 60)

    for i, asset in enumerate(asset_names):
        print(
            f"{asset:<10} "
            f"{weights[i]:>10.4f} "
            f"{expected_returns[i]:>12.4f} "
            f"{individual_vols[i]:>12.4f} "
            f"{pct_rc[i]:>11.2f}%"
        )

    print("-" * 60)
    print(
        f"{'PORTFOLIO':<10} "
        f"{np.sum(weights):>10.4f} "
        f"{portfolio_ret:>12.4f} "
        f"{portfolio_vol:>12.4f} "
        f"{np.sum(pct_rc):>11.2f}%"
    )
    print(f"\nDiversification Ratio: {div_ratio:.4f}")

    return {
        "portfolio_return": portfolio_ret,
        "portfolio_volatility": portfolio_vol,
        "risk_contributions": rc,
        "pct_risk_contributions": pct_rc,
        "individual_vols": individual_vols,
        "diversification_ratio": div_ratio,
    }


def plot_risk_pie(asset_names, pct_contributions):
    """
    Pie chart of percentage risk contributions.
    """
    plt.figure(figsize=(8, 8))
    plt.pie(
        pct_contributions,
        labels=asset_names,
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.title("Portfolio Risk Contribution")
    plt.axis("equal")
    plt.savefig("outputs/portfolio_risk_contribution.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_correlation_heatmap(correlation_matrix, asset_names):
    """
    Correlation heatmap using matplotlib imshow.
    """
    plt.figure(figsize=(8, 6))
    im = plt.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)

    plt.xticks(range(len(asset_names)), asset_names, rotation=45)
    plt.yticks(range(len(asset_names)), asset_names)

    plt.title("Correlation Heatmap")
    plt.colorbar(im, label="Correlation")

    for i in range(len(asset_names)):
        for j in range(len(asset_names)):
            plt.text(
                j,
                i,
                f"{correlation_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig("outputs/correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
