import sys; sys.path.insert(0, ".")
from portfolio_optimiser.costs import TransactionCostModel


def main():
    model = TransactionCostModel(
        commission_rate=0.001,   # 0.10%
        spread_cost=0.0005,      # 0.05%
        min_commission=1.0
    )

    portfolio_value = 100000

    # Start: equal weights across 4 stocks
    current_weights = [0.25, 0.25, 0.25, 0.25]

    # Rebalance to target max Sharpe example
    target_weights = [0.40, 0.30, 0.20, 0.10]

    turnover = model.turnover(current_weights, target_weights)
    total_cost = model.rebalance_cost(current_weights, target_weights, portfolio_value)

    gross_return = 0.12   # example 12% gross return
    net_return = model.net_return(gross_return, total_cost, portfolio_value)

    print("=== TRANSACTION COST MODEL TEST ===")
    print(f"Current weights: {current_weights}")
    print(f"Target weights:  {target_weights}")
    print(f"Portfolio value: ${portfolio_value:,.2f}")
    print(f"Turnover:        {turnover:.4f} ({turnover * 100:.2f}%)")
    print(f"Total cost:      ${total_cost:,.2f}")
    print(f"Gross return:    {gross_return:.4f} ({gross_return * 100:.2f}%)")
    print(f"Net return:      {net_return:.4f} ({net_return * 100:.2f}%)")


if __name__ == "__main__":
    main()
