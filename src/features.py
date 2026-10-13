"""Features : rendements, volatilité, statistiques descriptives, spreads.

Conventions
-----------
- Rendements **log** par défaut (additifs dans le temps, symétriques).
- Aucun forward-fill sur les rendements (= look-ahead) : on calcule les
  rendements sur les prix déjà alignés (jours cotés communs), le premier
  point est NaN puis retiré.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS


# --------------------------------------------------------------------------- #
# Rendements                                                                  #
# --------------------------------------------------------------------------- #
def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Rendements log : r_t = ln(P_t) - ln(P_{t-1}).

    Les prix <= 0 (ex. WTI négatif d'avril 2020) sont masqués en NaN avant le
    log (pas de -inf silencieux) ; ces jours tombent naturellement.
    """
    return np.log(prices.where(prices > 0)).diff()


# --------------------------------------------------------------------------- #
# Volatilité                                                                   #
# --------------------------------------------------------------------------- #
def rolling_vol(
    returns: pd.Series,
    window: int = 21,
    annualize: bool = True,
    periods: int = TRADING_DAYS,
) -> pd.Series:
    """Volatilité réalisée glissante (annualisée ×√periods par défaut)."""
    vol = returns.rolling(window).std(ddof=1)
    if annualize:
        vol = vol * np.sqrt(periods)
    return vol


# --------------------------------------------------------------------------- #
# Statistiques descriptives                                                    #
# --------------------------------------------------------------------------- #
def descriptive_stats(returns: pd.DataFrame, periods: int = TRADING_DAYS) -> pd.DataFrame:
    """Tableau de stats descriptives par colonne de rendements."""
    r = returns.dropna(how="all")
    stats = pd.DataFrame(index=r.columns)
    stats["n_obs"] = r.count()
    stats["mean_daily"] = r.mean()
    stats["std_daily"] = r.std(ddof=1)
    stats["ann_return"] = r.mean() * periods
    stats["ann_vol"] = r.std(ddof=1) * np.sqrt(periods)
    stats["skew"] = r.skew()
    stats["kurtosis"] = r.kurtosis()          # excès de kurtosis (Fisher)
    stats["min"] = r.min()
    stats["max"] = r.max()
    with np.errstate(invalid="ignore", divide="ignore"):
        stats["ann_sharpe"] = stats["ann_return"] / stats["ann_vol"]
    return stats


# --------------------------------------------------------------------------- #
# Spreads                                                                      #
# --------------------------------------------------------------------------- #
def price_spread(prices: pd.DataFrame, y: str = "brent", x: str = "wti") -> pd.Series:
    """Spread de prix simple : y - x (ex. Brent - WTI), en $/bbl."""
    s = prices[y] - prices[x]
    s.name = f"spread_{y}_{x}"
    return s


def log_spread(prices: pd.DataFrame, y: str = "brent", x: str = "wti") -> pd.Series:
    """Log-spread : ln(y) - ln(x) (sans dimension, ~ écart relatif).

    Prix <= 0 masqués en NaN (robuste au WTI négatif d'avril 2020).
    """
    s = np.log(prices[y].where(prices[y] > 0)) - np.log(prices[x].where(prices[x] > 0))
    s.name = f"logspread_{y}_{x}"
    return s


def rolling_mean_band(
    spread: pd.Series, window: int, n_std: float = 2.0
) -> pd.DataFrame:
    """Moyenne mobile + bandes ±n_std (pour la visualisation du spread)."""
    mu = spread.rolling(window).mean()
    sd = spread.rolling(window).std(ddof=1)
    return pd.DataFrame({
        "spread": spread,
        "mean": mu,
        "upper": mu + n_std * sd,
        "lower": mu - n_std * sd,
    })
