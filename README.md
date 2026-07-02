# Dupire Delta Hedge vs Neural-Network Delta Hedge

This project asks whether a small neural network can learn option hedge ratios that
produce lower out-of-sample hedging error than an educational Dupire/local-volatility
model. SPY is the default underlying.

The project runs end to end without paid historical option data. When
`data/historical_options.csv` is absent, it uses current option chains to construct
the volatility surface when Yahoo Finance permits it, then creates a clearly labeled
synthetic historical call series for the hedge backtest.

## Financial Background

An option gives its owner the right, but not the obligation, to trade an underlying
asset at a specified strike. Delta estimates how much the option price changes for a
one-dollar change in the underlying price.

Delta hedging offsets an option position's directional exposure by holding shares of
the underlying. For a short call, the strategy generally buys shares. As delta
changes, the hedge is rebalanced, producing turnover and transaction costs.

The Dupire model describes a local volatility surface that varies by strike and
maturity. This project builds a smooth implied-volatility surface, calculates
Black-Scholes prices on a grid, numerically applies the Dupire formula, and then uses
the resulting local volatility in a Black-Scholes delta. This is an educational
approximation, not a production calibration.

The neural network predicts option delta directly. With historical option chains,
its target is a stabilized realized delta:

```text
realized_delta = next_option_price_change / next_stock_price_change
```

It does **not** predict stock returns. It learns a hedge ratio intended to offset
next-period option-price changes.

## Portfolio and Comparison

The default portfolio is short one approximately ATM call with a 100-share contract
multiplier. Both strategies hedge the same option observations:

- **Dupire Delta Hedge:** uses the local-volatility delta proxy.
- **Neural-Network Delta Hedge:** uses the NN-predicted delta.

Cash is adjusted whenever shares are traded. Transaction costs are charged in basis
points of traded stock notional.

The project compares final value, total P&L, daily P&L volatility, Sharpe ratio,
mean absolute hedging error, RMSE hedging error, maximum drawdown, transaction costs,
and turnover.

## Data Priority and Yahoo Rate-Limit Protection

The loader uses this priority:

1. `data/historical_options.csv`, when supplied
2. Local stock and current-option cache files
3. A small, slowly downloaded yfinance current-option snapshot
4. Fully synthetic fallbacks if Yahoo is unavailable

To reduce `YFRateLimitError` problems, the project:

- Reuses caches for 24 hours
- Downloads only four expirations by default
- Waits four seconds between option-chain requests
- Uses only one slow retry after failures
- Falls back to older caches or synthetic data instead of repeatedly requesting Yahoo

Do not delete caches or repeatedly rerun requests merely to obtain newer quotes.
Settings can be changed conservatively in `config.py`.

## Historical CSV Format

Place a file at `data/historical_options.csv` with these columns:

```text
date, expiration, option_type, strike, bid, ask, lastPrice, volume,
openInterest, impliedVolatility, underlying_price
```

Multiple observations of the same strike and expiration across dates are required to
construct genuine realized-delta targets and a meaningful historical backtest.

## Installation and Run

Python 3.10 or newer is recommended.

```bash
cd Project_2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

For a guided walkthrough with the main tables and plots displayed inline, open and
run `dupire_vs_nn_hedging.ipynb` from top to bottom.

## Outputs

- `data/surface_options_used.csv`
- `data/backtest_options_used.csv`
- `data/nn_training_dataset.csv`
- `data/out_of_sample_portfolio.csv`
- `results/nn_hedge_model.pt`
- `results/strategy_comparison.csv`
- `results/dupire_backtest.csv`
- `results/nn_backtest.csv`
- `results/plots/`

## Interpretation and Caveats

- Judge the NN using out-of-sample hedging error, not training fit.
- yfinance does not provide a full, clean historical options database.
- True backtesting requires historical option chains with repeated contract quotes.
- Synthetic results demonstrate the pipeline; they are not evidence about live markets.
- The Dupire implementation uses interpolation and numerical derivatives and is
  intentionally approximate.
- Bid-ask spreads, stale quotes, transaction costs, and liquidity strongly affect
  real hedging.
- Better hedging performance does not imply a profitable trading strategy.
