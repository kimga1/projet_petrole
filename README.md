# Brent-WTI Statistical Arbitrage and WTI Term Structure

Python research project on the relative-value relationship between Brent and WTI crude-oil futures. The project combines cointegration diagnostics, dynamic hedge-ratio estimation, a walk-forward mean-reversion backtest, volatility targeting, regime analysis, and WTI futures-curve indicators.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://brent-wti-stat-arb.streamlit.app/)

**Live dashboard:** [brent-wti-stat-arb.streamlit.app](https://brent-wti-stat-arb.streamlit.app/)

The interface is available in English and French. This is a research and educational project, not an investment recommendation or a production trading system.

## Research objective

Brent and WTI respond to common global oil-market factors but can diverge because of regional inventories, transport constraints, crude quality, export policy, and market structure. The project studies whether unusually large deviations in a hedged Brent-WTI spread subsequently mean-revert.

The core relationship is:

```text
Brent_t = alpha_t + beta_t * WTI_t + residual_t
```

The strategy estimates the hedge ratio `beta`, standardises the spread with a rolling z-score, and takes a relative-value position when the deviation exceeds a defined threshold.

## Dashboard guide

### Sidebar controls

| Control | Purpose |
|---|---|
| Price source | Switches between front-month futures and FRED spot prices. |
| Hedge ratio | Selects Kalman, rolling OLS, or static OLS estimation. |
| Z-score lookback | Sets the rolling window used to estimate the spread mean and standard deviation. |
| Entry threshold | Opens a position when the absolute z-score exceeds this value. |
| Exit threshold | Closes a position after the spread returns toward its rolling mean. |
| Time stop | Closes positions that remain open beyond the selected number of sessions. |
| Costs per leg | Applies bid-ask and commission assumptions to both legs. |
| Volatility targeting | Scales exposure to a selected annualised volatility target. |
| Advanced settings | Controls the Kalman state-noise parameter, rolling-OLS window, and optional z-score stop. |

### 1. Overview

The overview summarises the selected configuration and its main outputs:

- cumulative constant-notional P&L, annualised return, Sharpe ratio, and maximum drawdown;
- full-sample cointegration statistics, estimated half-life, Hurst exponent, and current z-score;
- dynamically hedged spread with rolling mean and entry bands;
- performance by predefined oil-market regime;
- current Brent and WTI legs, hedge ratio, spread level, and signal state;
- optional Brent and WTI price history with major market events.

### 2. Cointegration and regimes

This section examines whether the statistical relationship is stable through time:

- full-sample Engle-Granger and residual ADF tests;
- rolling 252-session cointegration p-value;
- dynamic hedge ratio compared with a 252-session rolling OLS estimate;
- rolling mean-reversion half-life;
- CUSUM break diagnostics on the hedge ratio;
- a descriptive tradability flag based on rolling cointegration and half-life conditions.

CUSUM break dates are retrospective diagnostics and are not used as causal trading signals.

### 3. WTI curve

The curve section uses EIA WTI contracts `C1` to `C4` to describe futures term structure:

- historical curve for a user-selected date;
- contango, backwardation, and flat-regime frequencies;
- `C4 - C1` slope and approximate annualised roll yield;
- relationship between curve regime and the Brent-WTI price spread;
- 21-session WTI forward returns by curve regime, with a significance diagnostic.

The EIA discontinued the `RCLC` series after 5 April 2024, so this section ends on that date. It covers the WTI curve only; no equivalent free Brent curve is included.

### 4. Backtest

The backtest section reports:

- Sharpe ratio, annualised return and volatility, maximum drawdown, trade count, and hit rate;
- cumulative P&L for the raw and volatility-managed strategies;
- a buy-and-hold Brent reference series;
- constant-notional drawdown and rolling one-year Sharpe ratio;
- comparison of online Kalman estimation, quarterly walk-forward blocks, and a single pre-2018 split;
- performance across six economic and oil-market regimes;
- two documented failure modes: structural relationship breaks and abnormal prices.

### 5. Trades and sensitivity

This section focuses on robustness rather than a single headline result:

- Sharpe sensitivity across entry thresholds and z-score lookbacks;
- sensitivity to transaction-cost assumptions;
- trade count, hit rate, average gain and loss, profit factor, and holding period;
- individual entry and exit dates, direction, z-scores, hedge ratios, position sizes, and net P&L.

## Baseline specification

The stored reference results use the following configuration:

| Parameter | Baseline value |
|---|---:|
| Price source | Front-month futures |
| Hedge estimator | Online Kalman filter |
| Z-score lookback | 40 sessions |
| Entry / exit | `|z| > 2.0` / `|z| < 0.5` |
| Maximum holding period | 90 sessions |
| Transaction cost | 2.5 bps per leg |
| Volatility target | 10% annualised |
| Maximum leverage | 3.0x |
| Entry hedge-ratio range | 0.5 to 1.5 |
| Warm-up period | 252 sessions |

The hedge-ratio and positive-price entry checks exclude positions that no longer represent an economically meaningful Brent-WTI spread, including the negative-WTI episode of April 2020.

## Reference results

Reference period: **30 July 2007 to 16 September 2026**, covering **4,761 trading sessions**. The first 252 sessions are used as a warm-up period.

| Metric | Raw strategy | Volatility-managed |
|---|---:|---:|
| Annualised return | 4.1% | 4.5% |
| Annualised volatility | 9.1% | 9.9% |
| Sharpe ratio | 0.45 | 0.45 |
| Maximum drawdown | -20.3% | -20.1% |
| Sortino ratio | 0.64 | 0.66 |
| Trades | 73 | 73 |
| Hit rate | 59% | 55% |
| Profit factor | 2.74 | 2.15 |

These figures describe one baseline specification and are not presented as stable expected returns. The dashboard exposes parameter sensitivity, transaction-cost sensitivity, execution-lag results, and regime-level breakdowns. Across the stored entry/lookback grid, the volatility-managed Sharpe ratio ranges from approximately **0.21 to 0.59**.

Selected statistical diagnostics for the full-sample static relationship:

| Diagnostic | Value |
|---|---:|
| Static cointegrating beta | 1.057 |
| Engle-Granger p-value | 0.013 |
| Residual ADF p-value | 0.003 |
| Estimated half-life | 18.9 sessions |
| Hurst exponent | 0.29 |

## Methodology

1. Load and align Brent and WTI price series.
2. Estimate a static, rolling, or dynamic hedge ratio.
3. Construct the hedged spread and rolling z-score.
4. Generate stateful entry, exit, and time-stop signals.
5. Shift execution to prevent the strategy from acting on unavailable information.
6. Apply costs to entries, exits, and volatility-target rebalancing.
7. Calculate constant-notional P&L and risk statistics.
8. Evaluate stability across time periods, parameter grids, costs, and execution assumptions.

The online Kalman filter and rolling statistics use information available up to each observation. End-to-end tests verify that running the pipeline on a historical prefix produces the same overlapping returns as running it on the complete dataset.

## Data

| Dataset | Source | Use |
|---|---|---|
| Brent and WTI front-month futures | Yahoo Finance | Main tradable-price analysis and backtest |
| Brent and WTI spot prices | FRED | Alternative source and robustness checks |
| WTI contracts C1-C4 | U.S. EIA | Curve slope, contango/backwardation, and roll-yield analysis |

Cached parquet files are included so that the dashboard and reference results do not require a network request at startup. An EIA API key is only needed to refresh the WTI curve data.

## Project structure

```text
projet_petrole/
|-- dashboard/              Streamlit application, interface modules, and CSS
|-- src/                    Data, statistics, signals, backtest, risk, and pipeline modules
|-- scripts/                Data refresh and reference-result generation
|-- notebooks/              Exploration, cointegration, backtest, and curve analysis
|-- data/                   Cached input datasets
|-- results/                Metrics, equity, trades, regimes, and sensitivity outputs
|-- tests/                  Mathematical, schema, interface, and no-look-ahead tests
|-- report/                 Short research summary
|-- requirements.txt        Runtime dependencies
`-- requirements-dev.txt    Test dependencies
```

## Run locally

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run dashboard/app.py
```

The local application is available at `http://localhost:8501`.

To refresh the datasets and regenerate the stored results:

```bash
python scripts/fetch_data.py
python scripts/run_backtest.py
```

To run the validation suite:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Main limitations

- Front-month continuous futures are not roll-adjusted. Contract changes can introduce price jumps that would not map directly to a live rolled position.
- The static full-sample cointegration test and the dynamically hedged trading spread are related but not identical statistical objects.
- Cointegration is not stable in every subperiod; the 2014-2016 structural break is a documented weak regime.
- Kalman results are sensitive to the observation-noise assumption.
- The baseline uses a close-based execution convention. Additional execution delay reduces performance.
- The time stop, entry guard, and removal of a z-score stop were selected with knowledge of the historical sample.
- The WTI curve dataset ends in April 2024 and does not include a Brent futures curve.
- Yahoo Finance data are appropriate for research and demonstration, not institutional execution or valuation.

## Disclaimer

This repository is provided for educational and research purposes. It does not constitute investment advice, a solicitation, or evidence of future trading performance.
