"""Hedge ratio (β) entre les deux jambes : statique, rolling OLS, Kalman.

Pourquoi un hedge ratio dynamique (à défendre)
----------------------------------------------
La relation Brent = α + β·WTI n'est pas constante : β dérive avec les régimes
(qualité, logistique Cushing, arbitrage maritime, export ban...). Un β figé
sur tout l'échantillon utiliserait de l'information future et sur/sous-couvre
selon l'époque.
- **Rolling OLS** : β estimé sur une fenêtre glissante (simple, transparent).
- **Kalman** : β modélisé en marche aléatoire, mis à jour bayésiennement à
  chaque observation. Plus réactif, sans fenêtre arbitraire, et fournit
  naturellement un résidu (innovation) normalisé -> z-score « intégré ».

Anti look-ahead : à chaque date t, β_t n'utilise que l'information ≤ t
(la fenêtre se termine en t / le filtre est causal). Le décalage d'exécution
est géré dans backtest.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import StrategyConfig


# --------------------------------------------------------------------------- #
# OLS statique (in-sample / split simple)                                      #
# --------------------------------------------------------------------------- #
def static_ols(y: pd.Series, x: pd.Series) -> pd.DataFrame:
    """OLS plein échantillon : renvoie alpha, beta (constants) et le résidu."""
    df = pd.concat([y, x], axis=1, keys=["y", "x"]).dropna()
    X = sm.add_constant(df["x"])
    fit = sm.OLS(df["y"], X).fit()
    alpha = float(fit.params["const"])
    beta = float(fit.params["x"])
    out = pd.DataFrame(index=df.index)
    out["alpha"] = alpha
    out["beta"] = beta
    out["resid"] = df["y"] - alpha - beta * df["x"]
    return out


# --------------------------------------------------------------------------- #
# Rolling OLS (fenêtre glissante causale)                                      #
# --------------------------------------------------------------------------- #
def rolling_ols(y: pd.Series, x: pd.Series, window: int = 60) -> pd.DataFrame:
    """β glissant via cov/var sur fenêtre (causale : se termine en t)."""
    df = pd.concat([y, x], axis=1, keys=["y", "x"]).dropna()
    yy, xx = df["y"], df["x"]

    roll_cov = xx.rolling(window).cov(yy)
    roll_var = xx.rolling(window).var(ddof=1)
    beta = roll_cov / roll_var
    alpha = yy.rolling(window).mean() - beta * xx.rolling(window).mean()
    resid = yy - alpha - beta * xx

    out = pd.DataFrame(index=df.index)
    out["alpha"] = alpha
    out["beta"] = beta
    out["resid"] = resid
    return out


# --------------------------------------------------------------------------- #
# Filtre de Kalman (régression linéaire dynamique)                            #
# --------------------------------------------------------------------------- #
def kalman_hedge(
    y: pd.Series,
    x: pd.Series,
    delta: float = 1e-4,
    obs_cov: float = 1e-3,
) -> pd.DataFrame:
    """Kalman maison : état θ_t = [α_t, β_t] en marche aléatoire.

    Observation : y_t = [1, x_t]·θ_t + v_t,  v_t ~ N(0, obs_cov).
    Transition  : θ_t = θ_{t-1} + w_t,       w_t ~ N(0, Q), Q = δ/(1-δ)·I.

    Renvoie alpha_t, beta_t (état filtré), resid_t (innovation = erreur de
    prévision, naturellement stationnaire si cointégration) et resid_std_t
    (√ variance d'innovation) pour un z-score « intégré ».

    Causalité : l'innovation e_t = y_t - H_t·θ_{t|t-1} n'utilise que le passé
    pour l'état -> pas de look-ahead.
    """
    df = pd.concat([y, x], axis=1, keys=["y", "x"]).dropna()
    yy = df["y"].to_numpy(dtype=float)
    xx = df["x"].to_numpy(dtype=float)
    n = len(df)

    Q = (delta / (1.0 - delta)) * np.eye(2)   # covariance du bruit d'état
    R = float(obs_cov)                         # variance du bruit d'observation

    theta = np.zeros(2)                        # [alpha, beta] initial
    P = np.eye(2) * 1.0                        # incertitude initiale

    alpha_arr = np.full(n, np.nan)
    beta_arr = np.full(n, np.nan)
    resid_arr = np.full(n, np.nan)
    resid_std_arr = np.full(n, np.nan)

    for t in range(n):
        H = np.array([1.0, xx[t]])             # vecteur d'observation

        # --- Prédiction ---
        # theta_pred = theta (marche aléatoire) ; P_pred = P + Q
        P_pred = P + Q

        # --- Innovation (erreur de prévision = résidu tradable) ---
        y_hat = H @ theta
        e = yy[t] - y_hat
        S = float(H @ P_pred @ H.T + R)         # variance d'innovation

        # --- Gain et mise à jour ---
        K = (P_pred @ H) / S
        theta = theta + K * e
        P = P_pred - np.outer(K, H) @ P_pred

        alpha_arr[t] = theta[0]
        beta_arr[t] = theta[1]
        resid_arr[t] = e
        resid_std_arr[t] = np.sqrt(S)

    out = pd.DataFrame(index=df.index)
    out["alpha"] = alpha_arr
    out["beta"] = beta_arr
    out["resid"] = resid_arr                    # innovation (spread à trader)
    out["resid_std"] = resid_std_arr            # écart-type d'innovation
    return out


# --------------------------------------------------------------------------- #
# Dispatcher                                                                   #
# --------------------------------------------------------------------------- #
def compute_hedge(
    prices: pd.DataFrame, cfg: StrategyConfig
) -> pd.DataFrame:
    """Calcule le hedge selon cfg.hedge_method et renvoie alpha/beta/resid.

    Le résidu 'resid' est le spread cointégrant que le signal exploitera.
    Pour spread_kind='price'/'log', le hedge sert de β mais le résidu peut
    être remplacé par le spread brut (géré dans signals.py).
    """
    y = prices[cfg.price_col_y]
    x = prices[cfg.price_col_x]

    if cfg.hedge_method == "ols":
        return static_ols(y, x)
    if cfg.hedge_method == "rolling_ols":
        return rolling_ols(y, x, window=cfg.ols_window)
    if cfg.hedge_method == "kalman":
        return kalman_hedge(y, x, delta=cfg.kalman_delta, obs_cov=cfg.kalman_obs_cov)
    raise ValueError(f"hedge_method inconnu: {cfg.hedge_method!r}")
