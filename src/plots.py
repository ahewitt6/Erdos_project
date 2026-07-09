"""Matplotlib-only visualizations for the hedge comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def _line_plot(backtests, column, ylabel, title, filename, output_dir):
    plt.figure(figsize=(10, 5))
    for data in backtests:
        plt.plot(data["date"], data[column], label=data["strategy"].iloc[0])
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    _save(Path(output_dir) / filename)


def _spy_buy_and_hold_benchmark(reference_result):
    """Create a same-scale SPY buy-and-hold benchmark from backtest stock prices."""
    data = reference_result[["date", "stock_price"]].dropna().sort_values("date").copy()
    initial_value = abs(float(reference_result["portfolio_value"].iloc[0]))
    initial_stock_price = float(data["stock_price"].iloc[0])
    shares = initial_value / initial_stock_price if initial_stock_price else 0.0
    data["strategy"] = "SPY Buy-and-Hold"
    data["portfolio_value"] = shares * data["stock_price"]
    data["cumulative_pnl"] = data["portfolio_value"] - data["portfolio_value"].iloc[0]
    data["stock_position"] = shares
    data["transaction_cost"] = 0.0
    return data


def create_strategy_plots(dupire_result, nn_result, output_dir="results/plots"):
    """Save all requested backtest plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [dupire_result, nn_result]
    results_with_spy = results + [_spy_buy_and_hold_benchmark(dupire_result)]

    _line_plot(results_with_spy, "portfolio_value", "Portfolio value", "Portfolio Value", "portfolio_value.png", output_dir)
    _line_plot(results_with_spy, "cumulative_pnl", "Cumulative P&L", "Cumulative P&L", "cumulative_pnl.png", output_dir)
    _line_plot(results, "hedging_error", "Hedging error", "Daily Hedging Error", "daily_hedging_error.png", output_dir)
    _line_plot(results_with_spy, "stock_position", "Shares", "Stock Position", "stock_hedge_position.png", output_dir)
    _line_plot(results_with_spy, "transaction_cost", "Transaction cost", "Transaction Costs", "transaction_costs.png", output_dir)

    plt.figure(figsize=(9, 5))
    for data in results:
        plt.hist(data["hedging_error"], bins=30, alpha=0.55, label=data["strategy"].iloc[0])
    plt.xlabel("Hedging error")
    plt.ylabel("Frequency")
    plt.title("Hedging Error Distribution")
    plt.legend()
    _save(output_dir / "hedging_error_histogram.png")


def plot_volatility_surfaces(surface_options, local_vol_model, output_dir="results/plots"):
    """Save educational implied- and local-volatility surface plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9, 6))
    axis = fig.add_subplot(111, projection="3d")
    axis.scatter(
        surface_options["log_moneyness"],
        surface_options["T"],
        surface_options["impliedVolatility"],
        s=16,
    )
    axis.set_xlabel("Log moneyness")
    axis.set_ylabel("T")
    axis.set_zlabel("Implied volatility")
    axis.set_title("Implied Volatility Surface Points")
    _save(output_dir / "implied_volatility_surface.png")

    X, T = np.meshgrid(local_vol_model.log_moneyness_grid, local_vol_model.T_grid)
    fig = plt.figure(figsize=(9, 6))
    axis = fig.add_subplot(111, projection="3d")
    axis.plot_surface(X, T, local_vol_model.local_vol_grid, cmap="viridis")
    axis.set_xlabel("Log moneyness")
    axis.set_ylabel("T")
    axis.set_zlabel("Local volatility")
    axis.set_title("Dupire-Inspired Local Volatility Surface")
    _save(output_dir / "local_volatility_surface.png")
