"""Educational Dupire/local-volatility approximation.

This is intentionally a clear prototype, not a production surface calibration.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RegularGridInterpolator

from src.black_scholes import bs_call_delta, bs_call_price


def fit_iv_surface(options_df):
    """Return a robust interpolated implied-volatility function."""
    points = options_df[["log_moneyness", "T"]].to_numpy()
    values = options_df["impliedVolatility"].clip(0.01, 2.0).to_numpy()
    nearest = NearestNDInterpolator(points, values)
    try:
        linear = LinearNDInterpolator(points, values, fill_value=np.nan)
    except Exception:
        linear = None

    def sigma_iv(log_moneyness, T):
        x, t = np.broadcast_arrays(log_moneyness, T)
        query = np.column_stack([x.ravel(), t.ravel()])
        result = (
            np.asarray(linear(query), dtype=float)
            if linear is not None
            else np.full(len(query), np.nan)
        )
        missing = ~np.isfinite(result)
        result[missing] = nearest(query[missing])
        return np.clip(result.reshape(x.shape), 0.01, 2.0)

    return sigma_iv


@dataclass
class LocalVolModel:
    """Interpolated local-vol surface plus a practical delta proxy."""

    log_moneyness_grid: np.ndarray
    T_grid: np.ndarray
    local_vol_grid: np.ndarray
    spot_reference: float

    def __post_init__(self):
        self._interpolator = RegularGridInterpolator(
            (self.T_grid, self.log_moneyness_grid),
            self.local_vol_grid,
            bounds_error=False,
            fill_value=None,
        )

    def get_local_vol(self, log_moneyness, T):
        x, t = np.broadcast_arrays(log_moneyness, T)
        query = np.column_stack([t.ravel(), x.ravel()])
        values = self._interpolator(query).reshape(x.shape)
        return np.clip(values, 0.01, 2.0)

    def dupire_call_delta(self, S, K, T, r):
        log_moneyness = np.log(np.asarray(K, dtype=float) / np.asarray(S, dtype=float))
        sigma = self.get_local_vol(log_moneyness, T)
        return bs_call_delta(S, K, T, r, sigma)


def compute_local_vol_surface(options_df, r):
    """Numerically apply the Dupire formula to BS prices on a smooth IV grid."""
    iv_surface = fit_iv_surface(options_df)
    spot = float(options_df["underlying_price"].median())
    x_min, x_max = options_df["log_moneyness"].quantile([0.02, 0.98])
    t_min, t_max = options_df["T"].quantile([0.02, 0.98])
    if x_max - x_min < 0.05:
        center = (x_min + x_max) / 2
        x_min, x_max = center - 0.15, center + 0.15
    if t_max - t_min < 0.05:
        center = max((t_min + t_max) / 2, 30 / 365)
        t_min, t_max = max(7 / 365, center - 0.08), center + 0.08
    x_grid = np.linspace(x_min, x_max, 35)
    T_grid = np.linspace(max(t_min, 7 / 365), max(t_max, t_min + 0.05), 30)
    X, TT = np.meshgrid(x_grid, T_grid)
    strikes = spot * np.exp(X)
    iv = iv_surface(X, TT)
    prices = bs_call_price(spot, strikes, TT, r, iv)

    dC_dT = np.gradient(prices, T_grid, axis=0)
    dC_dK = np.gradient(prices, strikes[0], axis=1)
    d2C_dK2 = np.gradient(dC_dK, strikes[0], axis=1)
    numerator = dC_dT + r * strikes * dC_dK
    denominator = 0.5 * strikes**2 * d2C_dK2
    local_variance = np.divide(
        numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 1e-8
    )
    fallback = iv**2
    local_variance = np.where(np.isfinite(local_variance) & (local_variance > 0), local_variance, fallback)
    local_vol = np.sqrt(np.clip(local_variance, 0.01**2, 2.0**2))
    return LocalVolModel(x_grid, T_grid, local_vol, spot)


def get_local_vol(model, log_moneyness, T):
    return model.get_local_vol(log_moneyness, T)


def dupire_call_delta(model, S, K, T, r):
    return model.dupire_call_delta(S, K, T, r)


def compute_dupire_delta_for_options(options_df, r, model):
    data = options_df.copy()
    data["dupire_delta"] = model.dupire_call_delta(
        data["underlying_price"], data["strike"], data["T"], r
    )
    return data
