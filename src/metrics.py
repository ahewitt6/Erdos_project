"""Performance metrics for hedge-strategy comparison."""

import numpy as np
import pandas as pd


def maximum_drawdown(values):
    """Return the largest peak-to-trough loss in currency units."""
    values = pd.Series(values, dtype=float)
    return float((values - values.cummax()).min())


def strategy_metrics(backtest):
    """Calculate P&L, hedge error, drawdown, cost, and turnover metrics."""
    pnl = backtest["daily_pnl"]
    errors = backtest["hedging_error"]
    volatility = pnl.std(ddof=1)
    sharpe = np.sqrt(252) * pnl.mean() / volatility if volatility > 0 else 0.0
    return {
        "model": backtest["strategy"].iloc[0],
        "final_value": backtest["portfolio_value"].iloc[-1],
        "total_pnl": pnl.sum(),
        "mean_daily_pnl": pnl.mean(),
        "pnl_volatility": volatility,
        "sharpe": sharpe,
        "mean_abs_hedging_error": errors.abs().mean(),
        "rmse_hedging_error": np.sqrt(np.mean(errors**2)),
        "max_drawdown": maximum_drawdown(backtest["portfolio_value"]),
        "transaction_costs": backtest["transaction_cost"].sum(),
        "turnover": backtest["turnover"].sum(),
    }


def compare_strategies(*backtests):
    """Create one summary row per strategy."""
    return pd.DataFrame([strategy_metrics(result) for result in backtests])
