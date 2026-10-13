"""Tests statistiques : stationnarité (ADF), cointégration (Engle-Granger),
demi-vie de mean-reversion (Ornstein-Uhlenbeck / AR(1)), exposant de Hurst.

Pourquoi ces tests (à défendre)
-------------------------------
- **ADF sur chaque leg** : on attend des prix I(1) (non stationnaires). Trader
  un spread n'a de sens que si les niveaux sont non stationnaires mais qu'une
  combinaison linéaire l'est.
- **Cointégration (Engle-Granger)** : si Brent et WTI sont cointégrés, le
  résidu (spread cointégrant) est stationnaire -> mean-reversion tradable.
  On teste la stationnarité du résidu, pas seulement une corrélation.
- **Demi-vie** : calibre l'horizon de détention et les seuils. Un résidu
  stationnaire mais à demi-vie de 300 jours n'est pas tradable en pratique.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


# --------------------------------------------------------------------------- #
# ADF                                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class ADFResult:
    stat: float
    pvalue: float
    used_lag: int
    n_obs: int
    crit_values: dict
    is_stationary: bool  # au seuil 5 %

    def __repr__(self) -> str:
        verdict = "stationnaire" if self.is_stationary else "NON stationnaire"
        return (f"ADF(stat={self.stat:.3f}, p={self.pvalue:.4f}, "
                f"lag={self.used_lag}) -> {verdict} @5%")


def adf_test(series: pd.Series, regression: str = "c", alpha: float = 0.05) -> ADFResult:
    """Test Augmented Dickey-Fuller. H0 : racine unitaire (non stationnaire)."""
    s = pd.Series(series).dropna()
    try:
        # statsmodels >= 0.15 : result_object=False silence le FutureWarning
        stat, pvalue, used_lag, n_obs, crit, _ = adfuller(
            s.values, regression=regression, autolag="AIC", result_object=False
        )
    except TypeError:
        # statsmodels 0.14 : ce kwarg n'existe pas encore
        stat, pvalue, used_lag, n_obs, crit, _ = adfuller(
            s.values, regression=regression, autolag="AIC"
        )
    return ADFResult(
        stat=float(stat),
        pvalue=float(pvalue),
        used_lag=int(used_lag),
        n_obs=int(n_obs),
        crit_values={k: float(v) for k, v in crit.items()},
        is_stationary=bool(pvalue < alpha),
    )


# --------------------------------------------------------------------------- #
# Cointégration Engle-Granger                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class CointResult:
    alpha: float                 # constante de la régression y = alpha + beta*x
    beta: float                  # hedge ratio statique
    resid: pd.Series             # résidu (spread cointégrant)
    adf_resid: ADFResult         # ADF sur le résidu
    coint_pvalue: float          # p-value du test de cointégration (statsmodels)
    is_cointegrated: bool        # au seuil 5 %

    def __repr__(self) -> str:
        v = "cointégrés" if self.is_cointegrated else "NON cointégrés"
        return (f"Engle-Granger(beta={self.beta:.3f}, "
                f"coint_p={self.coint_pvalue:.4f}) -> {v} @5%")


def engle_granger(
    y: pd.Series, x: pd.Series, alpha_level: float = 0.05, add_const: bool = True
) -> CointResult:
    """Cointégration Engle-Granger en 2 étapes.

    1. OLS  y = alpha + beta*x  -> résidu.
    2. ADF sur le résidu (+ test coint de statsmodels avec valeurs critiques
       adaptées à un résidu estimé).
    """
    df = pd.concat([y, x], axis=1, keys=["y", "x"]).dropna()
    yy, xx = df["y"], df["x"]

    X = sm.add_constant(xx) if add_const else xx.to_frame()
    model = sm.OLS(yy, X).fit()
    beta = float(model.params.get("x", model.params.iloc[-1]))
    a = float(model.params.get("const", 0.0)) if add_const else 0.0
    resid = pd.Series(model.resid, index=df.index, name="coint_resid")

    # NB : cet ADF sur un résidu ESTIMÉ utilise les valeurs critiques standard de
    # Dickey-Fuller, trop libérales ici -> indicatif seulement. La DÉCISION de
    # cointégration s'appuie sur coint() ci-dessous, qui utilise les valeurs
    # critiques d'Engle-Granger (MacKinnon) adaptées à un résidu estimé.
    adf_resid = adf_test(resid, regression="c", alpha=alpha_level)
    coint_t, coint_p, _ = coint(yy, xx, trend="c", autolag="AIC")

    return CointResult(
        alpha=a,
        beta=beta,
        resid=resid,
        adf_resid=adf_resid,
        coint_pvalue=float(coint_p),
        is_cointegrated=bool(coint_p < alpha_level),
    )


# --------------------------------------------------------------------------- #
# Demi-vie de mean-reversion (Ornstein-Uhlenbeck / AR(1))                      #
# --------------------------------------------------------------------------- #
def half_life(spread: pd.Series) -> float:
    """Demi-vie de retour à la moyenne (en périodes/jours).

    Modèle AR(1) : s_t − μ = ρ (s_{t-1} − μ) + ε_t. On régresse Δs sur s_{t-1} :
    Δs_t = c + φ s_{t-1} + ε_t, avec φ = ρ − 1. L'écart à la moyenne est
    multiplié par ρ chaque jour -> demi-vie exacte = −ln(2)/ln(1+φ)
    (≈ −ln(2)/φ quand φ est petit, version continue Ornstein-Uhlenbeck).
    """
    s = pd.Series(spread).dropna()
    s_lag = s.shift(1)
    ds = (s - s_lag)
    df = pd.concat([ds, s_lag], axis=1, keys=["ds", "s_lag"]).dropna()
    if len(df) < 10:
        return float("nan")
    X = sm.add_constant(df["s_lag"])
    phi = float(sm.OLS(df["ds"], X).fit().params["s_lag"])
    if phi >= 0:
        return float("inf")  # pas de mean-reversion détectée
    if phi <= -1:
        return float("nan")  # ρ <= 0 : oscillation, demi-vie non définie
    return float(-np.log(2) / np.log1p(phi))


# --------------------------------------------------------------------------- #
# Exposant de Hurst (diagnostic mean-reversion vs trend)                       #
# --------------------------------------------------------------------------- #
def hurst_exponent(series: pd.Series, max_lag: int = 100) -> float:
    """Hurst via la variance des différences décalées.

    H < 0.5 : mean-reverting ; H = 0.5 : marche aléatoire ; H > 0.5 : trendant.
    """
    s = pd.Series(series).dropna().values
    max_lag = min(max_lag, len(s) // 2)
    if max_lag < 5:
        return float("nan")
    lags = range(2, max_lag)
    tau = [np.std(s[lag:] - s[:-lag], ddof=0) for lag in lags]
    tau = np.asarray(tau)
    mask = tau > 0
    if mask.sum() < 3:
        return float("nan")
    log_lags = np.log(np.fromiter(lags, dtype=float))[mask]
    poly = np.polyfit(log_lags, np.log(tau[mask]), 1)
    return float(poly[0])


# --------------------------------------------------------------------------- #
# Versions glissantes (pour la détection de régime)                           #
# --------------------------------------------------------------------------- #
def rolling_coint_pvalue(y: pd.Series, x: pd.Series, window: int = 252,
                         step: int = 5) -> pd.Series:
    """p-value de cointégration sur fenêtre glissante (échantillonnée tous
    ``step`` jours pour le coût de calcul). Montre où la cointégration casse."""
    df = pd.concat([y, x], axis=1, keys=["y", "x"]).dropna()
    idx, vals = [], []
    for end in range(window, len(df) + 1, step):
        w = df.iloc[end - window:end]
        try:
            _, p, _ = coint(w["y"], w["x"], trend="c", autolag=None, maxlag=1)
        except Exception:
            p = np.nan
        idx.append(df.index[end - 1])
        vals.append(p)
    return pd.Series(vals, index=pd.DatetimeIndex(idx), name="rolling_coint_p")


def rolling_half_life(spread: pd.Series, window: int = 252, step: int = 5) -> pd.Series:
    """Demi-vie sur fenêtre glissante."""
    s = pd.Series(spread).dropna()
    idx, vals = [], []
    for end in range(window, len(s) + 1, step):
        vals.append(half_life(s.iloc[end - window:end]))
        idx.append(s.index[end - 1])
    return pd.Series(vals, index=pd.DatetimeIndex(idx), name="rolling_half_life")
