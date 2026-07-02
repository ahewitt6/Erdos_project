"""Vectorized Black-Scholes prices and deltas."""

import numpy as np
from scipy.stats import norm


def _inputs(S, K, T, sigma):
    return (
        np.asarray(S, dtype=float),
        np.asarray(K, dtype=float),
        np.maximum(np.asarray(T, dtype=float), 1e-8),
        np.maximum(np.asarray(sigma, dtype=float), 1e-8),
    )


def _d1_d2(S, K, T, r, sigma):
    S, K, T, sigma = _inputs(S, K, T, sigma)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def bs_call_price(S, K, T, r, sigma):
    """Return European call prices."""
    S, K, T, sigma = _inputs(S, K, T, sigma)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return np.maximum(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), 0.0)


def bs_put_price(S, K, T, r, sigma):
    """Return European put prices."""
    S, K, T, sigma = _inputs(S, K, T, sigma)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return np.maximum(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1), 0.0)


def bs_call_delta(S, K, T, r, sigma):
    """Return European call deltas."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1)


def bs_put_delta(S, K, T, r, sigma):
    """Return European put deltas."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1) - 1.0
