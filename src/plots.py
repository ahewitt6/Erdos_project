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


def create_strategy_plots(dupire_result, nn_result, output_dir="results/plots"):
    """Save all requested backtest plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [dupire_result, nn_result]
    _line_plot(results, "portfolio_value", "Portfolio value", "Portfolio Value", "portfolio_value.png", output_dir)
    _line_plot(results, "cumulative_pnl", "Cumulative P&L", "Cumulative P&L", "cumulative_pnl.png", output_dir)
    _line_plot(results, "hedging_error", "Hedging error", "Daily Hedging Error", "daily_hedging_error.png", output_dir)
    _line_plot(results, "stock_position", "Shares", "Stock Hedge Position", "stock_hedge_position.png", output_dir)
    _line_plot(results, "transaction_cost", "Transaction cost", "Transaction Costs", "transaction_costs.png", output_dir)

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
