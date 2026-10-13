"""Sizing & contrôle du risque : vol-targeting (position ∝ 1/vol du spread).

Motivation (à défendre)
-----------------------
Le signal brut a un Sharpe ≈ 0 : ses pertes se concentrent dans les phases de
forte volatilité du spread. Le vol-targeting — taille ∝ 1/vol — réduit
l'exposition quand le spread devient agité et l'augmente quand il est calme.

Quelle vol mesurer ? (point clé)
--------------------------------
On mesure la vol du **spread couvert supposé toujours investi**, PAS celle des
rendements de la stratégie : ces derniers valent 0 quand on est flat, donc
juste après une période sans position la vol mesurée est ≈ 0 et le
multiplicateur saute au plafond (levier max) — exactement au mauvais moment.

Anti look-ahead : le multiplicateur à t utilise la vol réalisée **jusqu'à t-1**.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS


def spread_returns(y: pd.Series, x: pd.Series, beta: pd.Series) -> pd.Series:
    """Rendement quotidien d'un long spread (1 bbl y contre β bbl x) toujours investi.

    r_t = (Δy_t − β_{t-1}·Δx_t) / (|y_{t-1}| + |β_{t-1}|·|x_{t-1}|)
    Même convention que le moteur (notionnel brut = 1) ; β_{t-1} est connu la
    veille -> causal. Sert uniquement à MESURER la vol du spread.
    """
    b = beta.shift(1)
    notional = y.shift(1).abs() + b.abs() * x.shift(1).abs()
    ret = (y.diff() - b * x.diff()) / notional.where(notional > 1e-8)
    return ret.rename("spread_ret")


def realized_vol(returns: pd.Series, lookback: int = 21,
                 periods: int = TRADING_DAYS) -> pd.Series:
    """Volatilité réalisée annualisée glissante."""
    return returns.rolling(lookback).std(ddof=1) * np.sqrt(periods)


def vol_target_scalar(
    returns: pd.Series,
    target_vol: float = 0.10,
    lookback: int = 21,
    max_leverage: float = 3.0,
    periods: int = TRADING_DAYS,
) -> pd.Series:
    """Multiplicateur de position pour cibler ``target_vol`` (annualisée).

    scalar_t = clip(target / vol_réalisée_{t-1}, 0, max_leverage), où la vol est
    celle de ``returns`` (en pratique : ``spread_returns``, cf. docstring du module).
    """
    rv = realized_vol(returns, lookback, periods).shift(1)   # lag -> causal
    scalar = (target_vol / rv).clip(lower=0.0, upper=max_leverage)
    return scalar.fillna(0.0).rename("vol_scalar")


def vol_target_overlay(daily: pd.DataFrame, scalar: pd.Series,
                       cost_bps_per_leg: float) -> pd.DataFrame:
    """Applique le multiplicateur jour par jour au P&L du moteur, coûts compris.

    La position du jour t (détenue de la clôture t-1 à la clôture t) est
    multipliée par s_t, fixé à la clôture t-1. Donc :
      - P&L du jour          = s_t × P&L brut du moteur ;
      - entrée               = coût du moteur × s_t (on ouvre s_t unités) ;
      - sortie               = coût du moteur × s_{t-1} (on solde la taille de la veille) ;
      - rééquilibrage        = coût/jambe × |s_t − s_{t-1}| × notionnel en cours
                               (jours où la même position est conservée).

    Renvoie un DataFrame ``gross`` / ``cost`` / ``net`` (net = gross − cost).
    """
    held = daily["position"]
    s = scalar.reindex(daily.index).fillna(0.0)
    s_prev = s.shift(1).fillna(0.0)
    size = s.where(held != 0, s_prev)                  # taille en vigueur
    kept = (held != 0) & (held.shift(1) == held)       # même position que la veille
    rebal = (cost_bps_per_leg / 1e4) * (s - s_prev).abs() * daily["notional"] * kept

    gross = daily["gross_ret"] * size
    cost = daily["cost"] * size + rebal
    return pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost},
                        index=daily.index)

# NB : un vol-targeting GARCH(1,1) a été prototypé puis retiré (fit plein
# échantillon = léger look-ahead, gain marginal vs la vol réalisée causale
# ci-dessus). Piste conservée en « prolongements » du rapport : forecast
# GARCH refit de façon glissante.
