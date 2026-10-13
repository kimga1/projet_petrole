"""Test central : ABSENCE DE LOOK-AHEAD.

Propriété : un signal/hedge causal calculé sur un préfixe des données doit être
IDENTIQUE (sur les dates communes) à celui calculé sur la série complète.
Si de l'information future fuyait, allonger la série changerait le passé.
"""
import numpy as np
import pandas as pd

from src import backtest as bt
from src import hedge_ratio as hr
from src import signal as sig
from src.config import StrategyConfig


def _synthetic_prices(n=1500, seed=7):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 1, n)) + 80
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.92 * noise[t - 1] + rng.normal(0, 0.6)
    y = 1.1 * x + 3.0 + noise
    idx = pd.date_range("2012-01-01", periods=n, freq="B")
    return pd.DataFrame({"wti": x, "brent": y}, index=idx)


def test_kalman_hedge_is_causal():
    prices = _synthetic_prices()
    cfg = StrategyConfig(hedge_method="kalman")
    full = hr.compute_hedge(prices, cfg)
    prefix = hr.compute_hedge(prices.iloc[:1000], cfg)
    common = prefix.index
    # Le beta filtré sur le préfixe == beta sur la série complète (mêmes dates).
    assert np.allclose(full.loc[common, "beta"], prefix["beta"], atol=1e-9, equal_nan=True)


def test_rolling_ols_is_causal():
    prices = _synthetic_prices()
    cfg = StrategyConfig(hedge_method="rolling_ols", ols_window=60)
    full = hr.compute_hedge(prices, cfg)
    prefix = hr.compute_hedge(prices.iloc[:1000], cfg)
    common = prefix.index
    assert np.allclose(full.loc[common, "beta"].dropna(),
                       prefix["beta"].loc[common].dropna(), atol=1e-9)


def test_signal_positions_are_causal():
    """Les positions cibles sur un préfixe == positions sur la série complète."""
    prices = _synthetic_prices()
    cfg = StrategyConfig(hedge_method="kalman")
    hedge_full = hr.compute_hedge(prices, cfg)
    sig_full = sig.build_signal(prices, hedge_full, cfg)

    pfx = prices.iloc[:1000]
    hedge_pfx = hr.compute_hedge(pfx, cfg)
    sig_pfx = sig.build_signal(pfx, hedge_pfx, cfg)

    common = sig_pfx.index
    pd.testing.assert_series_equal(
        sig_full.loc[common, "position"], sig_pfx["position"], check_names=False)


def test_entry_respects_stop_band():
    """On n'entre jamais au-delà du stop (|z|>z_stop) : anti-whipsaw."""
    idx = pd.date_range("2015-01-01", periods=300, freq="B")
    cfg = StrategyConfig(z_in=2.0, z_stop=4.0)
    z_hi = pd.Series(np.full(300, 5.0), index=idx)     # au-dessus du stop
    assert (sig.generate_signals(z_hi, cfg) == 0).all()
    z_band = pd.Series(np.full(300, 2.5), index=idx)   # dans la bande -> short
    assert (sig.generate_signals(z_band, cfg) == -1).any()


def test_execution_lag_shifts_positions():
    """Avec execution_lag>=1, la position en vigueur à t vient d'une décision < t."""
    prices = _synthetic_prices()
    cfg = StrategyConfig(hedge_method="kalman", execution_lag=1, warmup=100)
    res = bt.run_backtest(prices, cfg)
    # Un rendement non nul le jour t implique une position décidée AVANT t :
    # on vérifie qu'il n'y a pas de P&L le tout premier jour de warmup.
    assert res.daily["net_ret"].iloc[:cfg.warmup].abs().sum() == 0.0


def test_full_pipeline_is_causal():
    """Bout en bout (hedge, signal, garde-fou, moteur, vol-target) : les
    rendements calculés sur un préfixe == ceux de la série complète."""
    from src import pipeline as pl
    prices = _synthetic_prices()
    cfg = StrategyConfig(hedge_method="kalman", warmup=100)
    full = pl.run_strategy(prices, cfg, vol_target=0.10)
    pfx = pl.run_strategy(prices.iloc[:1000], cfg, vol_target=0.10)
    common = pfx["returns_rm"].index
    assert np.allclose(full["returns_raw"].loc[common], pfx["returns_raw"], atol=1e-12)
    assert np.allclose(full["returns_rm"].loc[common], pfx["returns_rm"], atol=1e-12)


def test_entry_guard_blocks_absurd_hedge():
    """Pas d'entrée si un prix est <= 0 ou si β sort de [beta_min, beta_max]."""
    idx = pd.date_range("2015-01-01", periods=5, freq="B")
    cfg = StrategyConfig(z_in=2.0)
    z = pd.Series(3.0, index=idx)                          # signal de short partout
    y = pd.Series(50.0, index=idx)
    x = pd.Series([40.0, -37.0, 40.0, 40.0, 40.0], index=idx)
    beta = pd.Series([-0.02, 1.0, 1.8, 0.4, 1.0], index=idx)
    ok = sig.entry_allowed(y, x, beta, cfg)
    assert ok.tolist() == [False, False, False, False, True]
    pos = sig.generate_signals(z, cfg, can_enter=ok)
    assert pos.tolist() == [0.0, 0.0, 0.0, 0.0, -1.0]


def test_warmup_position_not_inherited():
    """Une position ouverte pendant l'amorçage n'est pas reprise à la fin de
    celui-ci : on attend un retour à flat puis un vrai signal."""
    prices = _synthetic_prices(n=300)
    idx = prices.index
    target = pd.Series(0.0, index=idx)
    target.iloc[40:120] = -1.0        # ouverte pendant le warmup (100), fermée après
    target.iloc[150:170] = 1.0        # vrai signal après le warmup
    signal_df = pd.DataFrame({"beta": 1.0, "resid": 0.0, "zscore": 0.0,
                              "position": target}, index=idx)
    cfg = StrategyConfig(warmup=100, execution_lag=1)
    res = bt._simulate(prices, signal_df, cfg)
    held = res.daily["position"]
    assert (held.iloc[:151] == 0).all()               # rien d'hérité
    assert (held.iloc[151:171] == 1.0).all()          # le vrai signal passe
    assert len(res.trades) == 1
