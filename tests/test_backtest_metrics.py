"""Tests du moteur de backtest (coûts, comptabilité) et des métriques."""
import numpy as np
import pandas as pd

from src import backtest as bt
from src import metrics as mt
from src.config import StrategyConfig


def _synthetic_prices(n=1500, seed=11):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 1, n)) + 80
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.9 * noise[t - 1] + rng.normal(0, 0.7)
    y = 1.0 * x + noise
    idx = pd.date_range("2012-01-01", periods=n, freq="B")
    return pd.DataFrame({"wti": x, "brent": y}, index=idx)


def test_costs_reduce_returns():
    """Des coûts plus élevés ne peuvent que réduire le P&L net."""
    prices = _synthetic_prices()
    zero = StrategyConfig(hedge_method="kalman", half_spread_bps=0.0, commission_bps=0.0)
    high = StrategyConfig(hedge_method="kalman", half_spread_bps=10.0, commission_bps=0.0)
    r0 = bt.run_backtest(prices, zero).returns.sum()
    r1 = bt.run_backtest(prices, high).returns.sum()
    assert r1 <= r0


def test_no_position_no_pnl():
    """Un seuil d'entrée inatteignable -> aucun trade -> P&L nul."""
    prices = _synthetic_prices()
    cfg = StrategyConfig(hedge_method="kalman", z_in=99.0)
    res = bt.run_backtest(prices, cfg)
    assert res.trades.empty
    assert np.isclose(res.returns.abs().sum(), 0.0)


def test_equity_finite():
    """La courbe d'équité reste finie (pas de NaN/inf)."""
    prices = _synthetic_prices()
    res = bt.run_backtest(prices, StrategyConfig(hedge_method="kalman"))
    assert np.isfinite(res.equity).all()


def test_trade_pnl_matches_daily():
    """La somme des P&L nets des trades ≈ somme des rendements nets quotidiens."""
    prices = _synthetic_prices()
    res = bt.run_backtest(prices, StrategyConfig(hedge_method="kalman"))
    if not res.trades.empty:
        assert abs(res.trades["net"].sum() - res.returns.sum()) < 1e-6


def test_sharpe_sign_and_scale():
    r = pd.Series(np.full(504, 0.001), index=pd.date_range("2015-01-01", periods=504, freq="B"))
    assert mt.sharpe(r) > 0
    # rendement constant -> vol nulle -> Sharpe non défini
    assert np.isnan(mt.sharpe(pd.Series([0.001] * 3)))  # <2 points std ok mais std=0


def test_max_drawdown_bounds():
    r = pd.Series([0.1, -0.5, 0.2, -0.3], index=pd.date_range("2015-01-01", periods=4, freq="B"))
    dd = mt.max_drawdown(r, compound=False)
    assert dd <= 0.0


def test_summary_keys():
    prices = _synthetic_prices()
    res = bt.run_backtest(prices, StrategyConfig(hedge_method="kalman"))
    s = mt.summary(res.returns, res.trades)
    for k in ("sharpe", "ann_return", "max_drawdown", "n_trades", "hit_rate"):
        assert k in s


def test_max_drawdown_in_notional_units():
    """Additif : perte depuis le pic en fraction du notionnel, pas du pic d'équité.
    Équité 1 -> 1,5 -> 1,3 : on a perdu 0,2 notionnel (et non 13,3 %)."""
    r = pd.Series([0.5, -0.2], index=pd.date_range("2015-01-01", periods=2, freq="B"))
    assert np.isclose(mt.max_drawdown(r, compound=False), -0.2)
    # Composé : % du pic. 1 -> 1,5 -> 1,2 = −20 %.
    assert np.isclose(mt.max_drawdown(r, compound=True), 1.5 * 0.8 / 1.5 - 1.0)


def test_sortino_uses_all_days():
    """Écart à la baisse = √ moyenne(min(r,0)²) sur tous les jours."""
    r = pd.Series([0.02, -0.01, 0.0, -0.03, 0.01])
    downside = np.sqrt((0.01 ** 2 + 0.03 ** 2) / 5)
    assert np.isclose(mt.sortino(r, periods=1), r.mean() / downside)


def test_annual_return_compound_is_cagr():
    """Composé : CAGR. +10 % puis −10 % -> 0,99 sur 2 périodes."""
    r = pd.Series([0.10, -0.10])
    assert np.isclose(mt.annual_return(r, periods=2, compound=True), 0.99 - 1.0)
    assert np.isclose(mt.annual_return(r, periods=2, compound=False), 0.0)
