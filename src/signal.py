"""Signal de trading : z-score glissant du résidu + machine à états entrée/sortie.

Convention de position (« spread position »)
--------------------------------------------
+1 = LONG le spread  : on achète y (Brent), on vend β·x (WTI). On prend +1
     quand z << 0 (spread sous sa moyenne -> on parie sur la remontée).
-1 = SHORT le spread : on vend y, on achète β·x. Pris quand z >> 0.
 0 = flat.

Anti look-ahead : la position renvoyée à la date t est décidée avec
l'information ≤ t. Le décalage d'exécution est géré dans backtest.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StrategyConfig


# --------------------------------------------------------------------------- #
# z-score glissant                                                            #
# --------------------------------------------------------------------------- #
def rolling_zscore(resid: pd.Series, lookback: int) -> pd.Series:
    """z_t = (resid_t - moyenne glissante) / écart-type glissant (causal)."""
    mu = resid.rolling(lookback).mean()
    sd = resid.rolling(lookback).std(ddof=1)
    z = (resid - mu) / sd
    z.name = "zscore"
    return z


# --------------------------------------------------------------------------- #
# Machine à états entrée / sortie                                             #
# --------------------------------------------------------------------------- #
def generate_signals(zscore: pd.Series, cfg: StrategyConfig,
                     can_enter: pd.Series | None = None) -> pd.Series:
    """Positions cibles {-1,0,+1} à partir du z-score (stateful).

    Entrée  : flat & z < -z_in  -> +1 ; flat & z > +z_in -> -1
              (uniquement les jours où ``can_enter`` est vrai, si fourni).
    Sortie  : |z| < z_out (retour vers la moyenne).
    Stop    : |z| > z_stop (divergence extrême -> on coupe, régime cassé ?).
    Holding : sortie forcée après max_holding_days (optionnel).
    """
    z = zscore.to_numpy(dtype=float)
    n = len(z)
    pos = np.zeros(n, dtype=float)
    allowed = (np.ones(n, dtype=bool) if can_enter is None
               else can_enter.reindex(zscore.index).fillna(False).to_numpy(dtype=bool))

    z_in, z_out = cfg.z_in, cfg.z_out
    z_stop = cfg.z_stop
    max_hold = cfg.max_holding_days

    current = 0.0
    days_in = 0
    for t in range(n):
        zt = z[t]
        if np.isnan(zt):
            pos[t] = 0.0
            current, days_in = 0.0, 0
            continue

        if current == 0.0:
            # entrée uniquement DANS la bande z_in < |z| < z_stop : on n'entre
            # jamais au-delà du stop (sinon whipsaw stop/ré-entrée en boucle).
            in_band = ((z_stop is None) or (abs(zt) < z_stop)) and allowed[t]
            if zt < -z_in and in_band:
                current = 1.0
                days_in = 1
            elif zt > z_in and in_band:
                current = -1.0
                days_in = 1
        else:
            days_in += 1
            exit_now = False
            # stop divergence
            if z_stop is not None and abs(zt) > z_stop:
                exit_now = True
            # sortie sur retour à la moyenne
            elif abs(zt) < z_out:
                exit_now = True
            # sortie sur retournement de signe franc (croisement de la moyenne)
            elif current == 1.0 and zt > z_out:
                exit_now = True
            elif current == -1.0 and zt < -z_out:
                exit_now = True
            # holding max
            elif max_hold is not None and days_in >= max_hold:
                exit_now = True
            if exit_now:
                current = 0.0
                days_in = 0
        pos[t] = current

    return pd.Series(pos, index=zscore.index, name="position")


def build_signal(prices: pd.DataFrame, hedge_df: pd.DataFrame,
                 cfg: StrategyConfig) -> pd.DataFrame:
    """Assemble le spread tradable, son z-score et les positions cibles.

    Le spread tradable est le **niveau** du spread cointégrant à β dynamique :
    ``spread_t = y_t - β_t·x_t`` (persistant, mean-reverting). On ne trade PAS
    l'innovation du Kalman (≈ bruit blanc) : le z-score glissant retranche la
    moyenne mobile, ce qui absorbe la constante α et la dérive lente de régime.
    """
    y = prices[cfg.price_col_y].reindex(hedge_df.index)
    x = prices[cfg.price_col_x].reindex(hedge_df.index)
    beta = hedge_df["beta"]

    if cfg.spread_kind == "price":
        spread = (y - x)
    elif cfg.spread_kind == "log":
        spread = (np.log(y) - np.log(x))
    else:  # "resid" : spread cointégrant à β dynamique
        spread = (y - beta * x)
    spread = spread.rename("resid")

    z = rolling_zscore(spread, cfg.z_lookback)
    can_enter = entry_allowed(y, x, beta, cfg)
    positions = generate_signals(z, cfg, can_enter=can_enter)

    out = pd.DataFrame(index=hedge_df.index)
    out["beta"] = beta
    out["resid"] = spread
    out["zscore"] = z
    out["can_enter"] = can_enter
    out["position"] = positions
    return out


def entry_allowed(y: pd.Series, x: pd.Series, beta: pd.Series,
                  cfg: StrategyConfig) -> pd.Series:
    """Garde-fou d'entrée (causal : information du jour seulement).

    Faux si un prix est <= 0 ou si β sort de [beta_min, beta_max] : le spread
    y − β·x n'a alors plus de sens économique (ex. 20/04/2020, WTI à −37 $,
    β Kalman < 0 -> la « couverture » devient une vente sèche de Brent).
    """
    ok = (y > 0) & (x > 0) & beta.notna()
    if cfg.beta_min is not None:
        ok &= beta >= cfg.beta_min
    if cfg.beta_max is not None:
        ok &= beta <= cfg.beta_max
    return ok.rename("can_enter")
