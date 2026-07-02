"""Portfolio selection and hedge-position helpers."""

import pandas as pd


def select_backtest_portfolio(options, portfolio_type="short_atm_call"):
    """Select a consistently observed call contract for the backtest."""
    if options.empty:
        raise ValueError("No options are available for portfolio construction.")

    data = options.copy()
    data["distance_from_atm"] = (data["moneyness"] - 1.0).abs()
    if portfolio_type == "short_atm_call":
        counts = data.groupby(["expiration", "strike"]).size().sort_values(ascending=False)
        best_count = counts.iloc[0]
        candidates = counts[counts == best_count].index
        expiration, strike = min(
            candidates,
            key=lambda key: data.loc[
                (data["expiration"] == key[0]) & (data["strike"] == key[1]),
                "distance_from_atm",
            ].mean(),
        )
        selected = data[
            (data["expiration"] == expiration) & (data["strike"] == strike)
        ].copy()
    elif portfolio_type == "short_call_basket":
        selected = (
            data.sort_values(["date", "distance_from_atm"])
            .groupby("date", as_index=False)
            .head(3)
            .groupby("date", as_index=False)
            .first()
        )
        print("Basket mode uses an equal-weight ATM representative in this educational version.")
    else:
        raise ValueError(f"Unknown portfolio_type: {portfolio_type}")

    return selected.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def stock_hedge_from_delta(option_delta, option_quantity=-1, contract_multiplier=100):
    """Convert an option delta into shares needed to delta hedge the position."""
    portfolio_delta = option_quantity * option_delta * contract_multiplier
    return -portfolio_delta
