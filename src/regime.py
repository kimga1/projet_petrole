"""Détection de régime : cointégration glissante, ruptures, indicateur tradable.

Objectif (à défendre)
---------------------
Documenter *où et pourquoi* la stratégie casse. La cointégration n'est pas
permanente : elle se dégrade lors des chocs structurels (glut de Cushing
2011-14, WTI négatif 2020...). On suit :
- la p-value de cointégration glissante (la relation tient-elle ?),
- le β glissant (le hedge dérive-t-il ?),
- la demi-vie glissante (la mean-reversion reste-t-elle exploitable ?),
- la vol réalisée (choc de risque),
et on repère les ruptures par un CUSUM simple (sans dépendance externe).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import features as ft
from . import hedge_ratio as hr
from .config import MARKET_EVENTS
from .stats_tests import rolling_coint_pvalue, rolling_half_life


def rolling_beta(y: pd.Series, x: pd.Series, window: int = 252) -> pd.Series:
    """β glissant (rolling OLS) pour suivre la dérive du hedge."""
    return hr.rolling_ols(y, x, window=window)["beta"].rename("rolling_beta")


def cusum_breaks(series: pd.Series, threshold: float = 5.0,
                 drift: float = 0.0, min_gap: int = 63) -> list[pd.Timestamp]:
    """Détecteur de ruptures de moyenne par CUSUM bilatéral (implémentation maison).

    Standardise la série, accumule les écarts à la moyenne et signale une
    rupture quand la somme cumulée dépasse ``threshold`` écarts-types, puis
    réinitialise. ``min_gap`` (en observations) évite les ruptures en rafale
    pendant un même choc prolongé. Renvoie les dates de rupture.

    NB : DIAGNOSTIC non causal — la standardisation utilise moyenne/écart-type
    plein échantillon. Sert à annoter/visualiser les ruptures a posteriori, pas
    à décider en temps réel (jamais utilisé dans la chaîne de trading).
    """
    s = pd.Series(series).dropna()
    if len(s) < 30:
        return []
    z = (s - s.mean()) / (s.std(ddof=1) + 1e-12)
    g_pos = 0.0
    g_neg = 0.0
    breaks = []
    last_i = -min_gap
    for i, (t, val) in enumerate(z.items()):
        g_pos = max(0.0, g_pos + val - drift)
        g_neg = min(0.0, g_neg + val + drift)
        if g_pos > threshold or g_neg < -threshold:
            if i - last_i >= min_gap:
                breaks.append(t)
                last_i = i
            g_pos = 0.0
            g_neg = 0.0
    return breaks


def regime_indicators(
    prices: pd.DataFrame,
    y: str = "brent",
    x: str = "wti",
    coint_window: int = 252,
    step: int = 5,
    beta_window: int = 252,
    vol_window: int = 21,
) -> pd.DataFrame:
    """Tableau d'indicateurs de régime alignés (échantillonné pour le coût CPU)."""
    coint_p = rolling_coint_pvalue(prices[y], prices[x], window=coint_window, step=step)
    half = rolling_half_life((np.log(prices[y]) - np.log(prices[x])),
                             window=coint_window, step=step)
    beta = rolling_beta(prices[y], prices[x], window=beta_window)
    spread_ret = (np.log(prices[y]) - np.log(prices[x])).diff()
    vol = ft.rolling_vol(spread_ret, window=vol_window)

    idx = prices.index
    out = pd.DataFrame(index=idx)
    out["coint_pvalue"] = coint_p.reindex(idx).ffill()
    out["rolling_half_life"] = half.reindex(idx).ffill()
    out["rolling_beta"] = beta
    out["spread_vol"] = vol
    # Flag « tradable » : cointégré (p<0.10) ET demi-vie raisonnable (2-150 j).
    out["cointegrated"] = out["coint_pvalue"] < 0.10
    out["tradeable"] = (
        out["cointegrated"]
        & out["rolling_half_life"].between(2, 150)
    )
    return out


def event_lines() -> pd.DataFrame:
    """Événements de marché connus (pour annoter les graphiques)."""
    df = pd.DataFrame(MARKET_EVENTS)
    df["date"] = pd.to_datetime(df["date"])
    return df
