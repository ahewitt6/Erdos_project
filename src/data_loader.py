"""Load historical CSV data or use rate-limit-safe yfinance fallbacks."""

import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from src.black_scholes import bs_call_price
from src.utils import cache_is_fresh, retry_request


EXPECTED_OPTION_COLUMNS = [
    "date", "expiration", "option_type", "strike", "bid", "ask", "lastPrice",
    "volume", "openInterest", "impliedVolatility", "underlying_price",
]


def _synthetic_stock_prices(days=600, seed=42):
    """Create a reproducible stock path only when Yahoo and caches are unavailable."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    returns = rng.normal(0.0003, 0.012, len(dates))
    close = 500.0 * np.exp(np.cumsum(returns))
    print("WARNING: Using synthetic stock prices because Yahoo Finance is unavailable.")
    return pd.DataFrame({"date": dates, "close": close})


def load_stock_prices(config):
    """Load cached SPY history, refresh it carefully, or synthesize a fallback."""
    cache = Path("data") / f"{config.symbol.lower()}_stock_prices.csv"
    if cache_is_fresh(cache, config.cache_hours):
        print(f"Loading cached stock prices from {cache}.")
        stock = pd.read_csv(cache)
    else:
        try:
            ticker = yf.Ticker(config.symbol)
            history = retry_request(
                lambda: ticker.history(period=config.stock_history_period, auto_adjust=True),
                "Stock-history request",
                config.max_retries,
                config.retry_wait_seconds,
            )
            if history.empty:
                raise ValueError("Yahoo returned no stock history.")
            stock = history.reset_index()[["Date", "Close"]].rename(
                columns={"Date": "date", "Close": "close"}
            )
            stock.to_csv(cache, index=False)
        except Exception as exc:
            if cache.exists():
                print(f"Stock refresh failed ({exc}); using older cache.")
                stock = pd.read_csv(cache)
            else:
                stock = _synthetic_stock_prices(seed=config.random_seed)
                stock.to_csv(cache, index=False)

    stock["date"] = pd.to_datetime(stock["date"]).dt.tz_localize(None).dt.normalize()
    stock = stock.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    stock["returns"] = stock["close"].pct_change()
    stock["log_returns"] = np.log(stock["close"] / stock["close"].shift(1))
    stock["realized_vol_20"] = stock["log_returns"].rolling(20).std() * np.sqrt(252)
    stock["realized_vol_60"] = stock["log_returns"].rolling(60).std() * np.sqrt(252)
    stock[["realized_vol_20", "realized_vol_60"]] = stock[
        ["realized_vol_20", "realized_vol_60"]
    ].bfill().fillna(0.20)
    return stock.tail(config.backtest_days + 80).reset_index(drop=True)


def _current_underlying_price(ticker):
    """Use a short history request instead of yfinance fast_info."""
    history = ticker.history(period="5d", auto_adjust=True)
    if history.empty:
        raise ValueError("No current underlying price was returned.")
    return float(history["Close"].dropna().iloc[-1])


def download_current_option_snapshot(config):
    """Download only a few chains with delays, retries, and a persistent cache."""
    cache = Path("data") / f"current_options_{config.symbol}.csv"
    if cache_is_fresh(cache, config.cache_hours):
        print(f"Loading cached current option chains from {cache}.")
        return pd.read_csv(cache)

    try:
        ticker = yf.Ticker(config.symbol)
        expirations = retry_request(
            lambda: ticker.options,
            "Option expiration request",
            config.max_retries,
            config.retry_wait_seconds,
        )[: config.max_expirations]
        if not expirations:
            raise ValueError("Yahoo returned no option expirations.")
        spot = retry_request(
            lambda: _current_underlying_price(ticker),
            "Current-price request",
            config.max_retries,
            config.retry_wait_seconds,
        )
        frames = []
        for index, expiration in enumerate(expirations):
            try:
                calls = retry_request(
                    lambda expiration=expiration: ticker.option_chain(expiration).calls.copy(),
                    f"Option-chain request for {expiration}",
                    config.max_retries,
                    config.retry_wait_seconds,
                )
                calls["date"] = pd.Timestamp.today().normalize()
                calls["expiration"] = expiration
                calls["option_type"] = "call"
                calls["underlying_price"] = spot
                frames.append(calls)
            except Exception as exc:
                print(f"Skipping {expiration}: {exc}")
            if index < len(expirations) - 1:
                time.sleep(config.request_delay_seconds)
        if not frames:
            raise ValueError("No option chains were downloaded.")
        options = pd.concat(frames, ignore_index=True)
        options.to_csv(cache, index=False)
        return options
    except Exception as exc:
        if cache.exists():
            print(f"Yahoo option request failed ({exc}); using older option cache.")
            return pd.read_csv(cache)
        print(f"WARNING: Yahoo option request failed ({exc}).")
        return pd.DataFrame()


def clean_options(options):
    """Apply consistent call-option cleaning and feature calculations."""
    if options.empty:
        return options
    data = options.copy()
    for column in EXPECTED_OPTION_COLUMNS:
        if column not in data:
            data[column] = np.nan
    data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None).dt.normalize()
    data["expiration"] = pd.to_datetime(data["expiration"]).dt.tz_localize(None).dt.normalize()
    numeric = [
        "strike", "bid", "ask", "lastPrice", "volume", "openInterest",
        "impliedVolatility", "underlying_price",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["volume"] = data["volume"].fillna(0)
    data["mid_price"] = (data["bid"] + data["ask"]) / 2
    data["days_to_expiry"] = (data["expiration"] - data["date"]).dt.days
    data["T"] = data["days_to_expiry"] / 365.0
    data["moneyness"] = data["strike"] / data["underlying_price"]
    data["log_moneyness"] = np.log(data["moneyness"])
    data["bid_ask_spread"] = data["ask"] - data["bid"]
    data["relative_spread"] = data["bid_ask_spread"] / data["mid_price"]
    valid = (
        data["option_type"].astype(str).str.lower().eq("call")
        & (data["bid"] > 0)
        & (data["ask"] > 0)
        & (data["ask"] >= data["bid"])
        & (data["openInterest"] > 0)
        & (data["impliedVolatility"] > 0)
        & data["days_to_expiry"].between(7, 365)
        & data["moneyness"].between(0.8, 1.2)
    )
    return data.loc[valid].replace([np.inf, -np.inf], np.nan).dropna(
        subset=["mid_price", "T", "log_moneyness"]
    ).reset_index(drop=True)


def generate_synthetic_surface(stock, config):
    """Create a broad current surface when no usable current chains are available."""
    spot = float(stock["close"].iloc[-1])
    snapshot = stock["date"].iloc[-1]
    rows = []
    for days in [30, 60, 120, 240]:
        for moneyness in np.linspace(0.8, 1.2, 17):
            strike = spot * moneyness
            sigma = np.clip(0.18 + 0.30 * (moneyness - 1.0) ** 2 + 0.04 * np.sqrt(days / 365), 0.08, 0.80)
            price = float(bs_call_price(spot, strike, days / 365, config.risk_free_rate, sigma))
            spread = max(0.04, price * 0.015)
            rows.append({
                "date": snapshot, "expiration": snapshot + pd.Timedelta(days=days),
                "option_type": "call", "strike": strike, "bid": max(price - spread / 2, 0.01),
                "ask": price + spread / 2, "lastPrice": price, "volume": 100,
                "openInterest": 1000, "impliedVolatility": sigma, "underlying_price": spot,
            })
    print("WARNING: Using a synthetic option surface because no current chain is available.")
    return clean_options(pd.DataFrame(rows))


def generate_synthetic_option_history(stock, config):
    """Create one consistently tracked call contract for an end-to-end demonstration."""
    rng = np.random.default_rng(config.random_seed)
    data = stock.tail(config.backtest_days).copy().reset_index(drop=True)
    strike = round(float(data["close"].iloc[0]) / 5) * 5
    expiration = data["date"].iloc[-1] + pd.Timedelta(days=45)
    data["T"] = (expiration - data["date"]).dt.days / 365
    sigma = data["realized_vol_20"].clip(0.08, 0.80)
    theoretical = bs_call_price(data["close"], strike, data["T"], config.risk_free_rate, sigma)
    noise = rng.normal(0, 0.01, len(data)) * np.maximum(theoretical, 1.0)
    mid = np.maximum(theoretical + noise, np.maximum(data["close"] - strike, 0) + 0.01)
    spread = np.maximum(0.04, mid * 0.01)
    raw = pd.DataFrame({
        "date": data["date"], "expiration": expiration, "option_type": "call",
        "strike": strike, "bid": np.maximum(mid - spread / 2, 0.01),
        "ask": mid + spread / 2, "lastPrice": mid, "volume": 500,
        "openInterest": 5000, "impliedVolatility": sigma,
        "underlying_price": data["close"],
    })
    print("WARNING: No historical option CSV found; using synthetic historical calls for backtesting.")
    return clean_options(raw)


def load_all_data(config):
    """Return stock data, a surface snapshot, historical/backtest options, and mode."""
    stock = load_stock_prices(config)
    historical_path = Path("data/historical_options.csv")
    if historical_path.exists():
        historical = clean_options(pd.read_csv(historical_path))
        if historical.empty:
            raise ValueError("data/historical_options.csv exists but has no rows after cleaning.")
        mode = "historical_csv"
        surface = historical[historical["date"] == historical["date"].max()].copy()
    else:
        current = clean_options(download_current_option_snapshot(config))
        surface = current if not current.empty else generate_synthetic_surface(stock, config)
        historical = generate_synthetic_option_history(stock, config)
        mode = "synthetic_backtest"
    return stock, surface, historical, mode
