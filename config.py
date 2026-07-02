"""Central configuration for the hedging comparison project."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    symbol: str = "SPY"
    risk_free_rate: float = 0.04
    contract_multiplier: int = 100
    option_quantity: int = -1
    portfolio_type: str = "short_atm_call"
    rebalance_frequency: str = "daily"
    initial_cash: float = 0.0
    transaction_cost_bps: float = 1.0
    random_seed: int = 42

    stock_history_period: str = "2y"
    backtest_days: int = 252

    # Conservative Yahoo settings. Cached files are reused for a full day.
    max_expirations: int = 4
    cache_hours: int = 24
    request_delay_seconds: int = 4
    retry_wait_seconds: int = 20
    max_retries: int = 2

    nn_epochs: int = 300
    nn_learning_rate: float = 1e-3
    nn_patience: int = 35


CONFIG = Config()
