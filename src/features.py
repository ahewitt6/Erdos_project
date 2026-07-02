"""Build features and realized-delta targets."""

import numpy as np
import pandas as pd

from src.black_scholes import bs_call_delta


FEATURE_COLUMNS = [
    "moneyness", "log_moneyness", "T", "impliedVolatility", "realized_vol_20",
    "realized_vol_60", "option_delta_bs", "mid_price", "relative_spread",
    "volume", "openInterest", "recent_stock_return", "recent_absolute_return",
]


def add_market_features(options, stock, r):
    """Merge stock features onto option observations and calculate BS delta."""
    data = options.copy()
    stock_features = stock[
        ["date", "returns", "realized_vol_20", "realized_vol_60"]
    ].rename(columns={"returns": "recent_stock_return"})
    data = data.merge(stock_features, on="date", how="left")
    data["recent_absolute_return"] = data["recent_stock_return"].abs()
    data["option_delta_bs"] = bs_call_delta(
        data["underlying_price"], data["strike"], data["T"], r, data["impliedVolatility"]
    )
    data["recent_stock_return"] = data["recent_stock_return"].fillna(0)
    data["recent_absolute_return"] = data["recent_absolute_return"].fillna(0)
    data[["realized_vol_20", "realized_vol_60"]] = data[
        ["realized_vol_20", "realized_vol_60"]
    ].fillna(data["impliedVolatility"])
    return data.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


def add_realized_delta_target(options, minimum_stock_move=0.01):
    """Calculate next-period dC/dS within each strike/expiration contract."""
    data = options.sort_values(["expiration", "strike", "date"]).copy()
    groups = data.groupby(["expiration", "strike"], sort=False)
    data["next_option_price"] = groups["mid_price"].shift(-1)
    data["next_stock_price"] = groups["underlying_price"].shift(-1)
    dC = data["next_option_price"] - data["mid_price"]
    dS = data["next_stock_price"] - data["underlying_price"]
    data["realized_delta"] = (dC / dS).clip(-2, 2)
    return data.loc[dS.abs() >= minimum_stock_move].dropna(subset=["realized_delta"]).reset_index(drop=True)
