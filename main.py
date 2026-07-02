"""Run the Dupire delta hedge versus neural-network delta hedge project."""

from pathlib import Path

import pandas as pd

from config import CONFIG
from src.backtest import run_hedge_backtest
from src.data_loader import load_all_data
from src.dupire import compute_dupire_delta_for_options, compute_local_vol_surface
from src.features import add_market_features
from src.metrics import compare_strategies
from src.nn_hedge import prepare_nn_dataset, predict_nn_delta, train_nn_hedge_model
from src.plots import create_strategy_plots, plot_volatility_surfaces
from src.portfolio import select_backtest_portfolio
from src.utils import ensure_directories, set_random_seeds


def main():
    ensure_directories()
    set_random_seeds(CONFIG.random_seed)

    print("Loading stock and option data...")
    stock, surface_options, historical_options, mode = load_all_data(CONFIG)
    print(f"Data mode: {mode}")
    if mode == "synthetic_backtest":
        print(
            "The hedge comparison uses synthetic historical calls. "
            "Add data/historical_options.csv for a true historical-options study."
        )
    surface_options.to_csv("data/surface_options_used.csv", index=False)
    historical_options.to_csv("data/backtest_options_used.csv", index=False)

    print("\nFitting educational Dupire/local-volatility surface...")
    local_vol_model = compute_local_vol_surface(surface_options, CONFIG.risk_free_rate)
    plot_volatility_surfaces(surface_options, local_vol_model)

    print("\nPreparing neural-network hedge data...")
    nn_dataset = prepare_nn_dataset(historical_options, stock, CONFIG.risk_free_rate)
    nn_dataset.to_csv("data/nn_training_dataset.csv", index=False)
    model, scaler, test_start_date = train_nn_hedge_model(
        nn_dataset,
        epochs=CONFIG.nn_epochs,
        learning_rate=CONFIG.nn_learning_rate,
        patience=CONFIG.nn_patience,
        random_seed=CONFIG.random_seed,
    )

    featured_options = add_market_features(
        historical_options, stock, CONFIG.risk_free_rate
    )
    featured_options = compute_dupire_delta_for_options(
        featured_options, CONFIG.risk_free_rate, local_vol_model
    )

    out_of_sample_features = featured_options[
        featured_options["date"] >= pd.Timestamp(test_start_date)
    ].copy()
    if len(out_of_sample_features) < 5:
        raise ValueError(
            "Fewer than five out-of-sample option rows remain. "
            "Provide a longer historical option dataset."
        )

    portfolio = select_backtest_portfolio(out_of_sample_features, CONFIG.portfolio_type)
    portfolio["nn_delta"] = predict_nn_delta(model, scaler, portfolio)
    portfolio.to_csv("data/out_of_sample_portfolio.csv", index=False)
    print(
        f"Running both strategies on {len(portfolio)} out-of-sample dates "
        f"from {portfolio['date'].min().date()} onward."
    )

    dupire_result = run_hedge_backtest(
        portfolio, "dupire_delta", "Dupire Delta Hedge", CONFIG
    )
    nn_result = run_hedge_backtest(
        portfolio, "nn_delta", "Neural-Network Delta Hedge", CONFIG
    )
    dupire_result.to_csv("results/dupire_backtest.csv", index=False)
    nn_result.to_csv("results/nn_backtest.csv", index=False)

    comparison = compare_strategies(dupire_result, nn_result)
    comparison.to_csv("results/strategy_comparison.csv", index=False)
    create_strategy_plots(dupire_result, nn_result)

    print("\nOut-of-sample strategy comparison:")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nSaved intermediate data, backtests, model, metrics, and plots.")


if __name__ == "__main__":
    main()
