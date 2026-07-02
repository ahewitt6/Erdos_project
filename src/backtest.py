"""Backtest a short-call portfolio with configurable hedge deltas."""

import numpy as np
import pandas as pd

from src.portfolio import stock_hedge_from_delta


def run_hedge_backtest(options, delta_column, strategy_name, config):
    """Track cash, stock hedge, marked option value, P&L, costs, and turnover."""
    data = options.sort_values("date").reset_index(drop=True).copy()
    cash = float(config.initial_cash)
    old_stock_position = 0.0
    previous_option_price = None
    previous_stock_price = None
    previous_value = None
    rows = []

    for index, row in data.iterrows():
        stock_price = float(row["underlying_price"])
        option_price = float(row["mid_price"])
        should_rebalance = config.rebalance_frequency == "daily" or index % 5 == 0
        if should_rebalance:
            new_stock_position = float(
                stock_hedge_from_delta(
                    row[delta_column], config.option_quantity, config.contract_multiplier
                )
            )
        else:
            new_stock_position = old_stock_position
        trade_size = new_stock_position - old_stock_position
        transaction_cost = (
            abs(trade_size) * stock_price * config.transaction_cost_bps / 10000
        )
        cash -= trade_size * stock_price + transaction_cost
        option_value = config.option_quantity * option_price * config.contract_multiplier
        portfolio_value = option_value + new_stock_position * stock_price + cash

        if previous_value is None:
            daily_pnl = 0.0
            hedging_error = 0.0
        else:
            daily_pnl = portfolio_value - previous_value
            option_pnl = (
                config.option_quantity
                * (option_price - previous_option_price)
                * config.contract_multiplier
            )
            stock_pnl = old_stock_position * (stock_price - previous_stock_price)
            hedging_error = option_pnl + stock_pnl - transaction_cost

        rows.append({
            "date": row["date"],
            "strategy": strategy_name,
            "option_price": option_price,
            "stock_price": stock_price,
            "option_delta": row[delta_column],
            "stock_position": new_stock_position,
            "cash": cash,
            "portfolio_value": portfolio_value,
            "daily_pnl": daily_pnl,
            "hedging_error": hedging_error,
            "transaction_cost": transaction_cost,
            "turnover": abs(trade_size),
        })
        old_stock_position = new_stock_position
        previous_stock_price = stock_price
        previous_option_price = option_price
        previous_value = portfolio_value

    result = pd.DataFrame(rows)
    result["cumulative_pnl"] = result["daily_pnl"].cumsum()
    return result
