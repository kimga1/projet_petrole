"""Métriques de performance : Sharpe, drawdown, hit rate, etc.

Toutes les fonctions prennent une série de rendements NETS (après coûts) et
sont robustes aux périodes hors position (rendement = 0).

Deux conventions (paramètre ``compound``)
-----------------------------------------
- ``compound=False`` (défaut) : spread market-neutral à **notionnel constant**.
  P&L additif, équité = 1 + cumsum ; rendement annualisé = moyenne × 252 ;
  drawdown = perte depuis le pic **en fraction du notionnel** (même unité que
  la vol et le rendement — sinon les gains accumulés gonflent le dénominateur
  et sous-estiment les drawdowns tardifs).
- ``compound=True`` : position longue directionnelle (buy & hold).
  Équité = cumprod(1 + r) ; rendement annualisé = CAGR ; drawdown en % du pic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS


def equity_curve(returns: pd.Series, start: float = 1.0,
                 compound: bool = True) -> pd.Series:
    """Courbe d'équité.

    compound=True  : composée (cumprod) — adaptée à un long directionnel (B&H).
    compound=False : additive (1+cumsum) — convention correcte pour un spread
                     market-neutral à notionnel constant (robuste aux
                     rendements journaliers < -100 %).
    """
    r = returns.fillna(0.0)
    if compound:
        return start * (1.0 + r).cumprod()
    return start * (1.0 + r.cumsum())


def annual_return(returns: pd.Series, periods: int = TRADING_DAYS,
                  compound: bool = False) -> float:
    """Moyenne × periods (additif) ou CAGR (composé)."""
    r = returns.fillna(0.0)
    if not compound:
        return float(r.mean() * periods)
    if len(r) == 0:
        return float("nan")
    growth = float((1.0 + r).prod())
    if growth <= 0:
        return -1.0
    return float(growth ** (periods / len(r)) - 1.0)


def annual_vol(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods))


def sharpe(returns: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    """Sharpe annualisé (rf annualisé optionnel)."""
    r = returns.dropna()
    if r.std(ddof=1) == 0 or len(r) < 2:
        return float("nan")
    excess = r - rf / periods
    return float(np.sqrt(periods) * excess.mean() / r.std(ddof=1))


def sortino(returns: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    """Sortino annualisé.

    Écart à la baisse = √ moyenne(min(r − rf, 0)²) calculée sur TOUS les jours
    (les jours positifs comptent pour 0) — et non l'écart-type des seuls jours
    négatifs, qui retrancherait leur moyenne et changerait le dénominateur.
    """
    r = returns.dropna()
    excess = r - rf / periods
    downside = np.sqrt((np.minimum(excess, 0.0) ** 2).mean())
    if downside == 0 or np.isnan(downside):
        return float("nan")
    return float(np.sqrt(periods) * excess.mean() / downside)


def drawdown_series(returns: pd.Series, compound: bool = False) -> pd.Series:
    """Drawdown depuis le pic : fraction du notionnel (additif) ou % du pic (composé)."""
    eq = equity_curve(returns, compound=compound)
    if compound:
        return eq / eq.cummax() - 1.0
    return eq - eq.cummax()


def max_drawdown(returns: pd.Series, compound: bool = False) -> float:
    """Drawdown maximal (valeur négative), cf. ``drawdown_series``."""
    return float(drawdown_series(returns, compound=compound).min())


def calmar(returns: pd.Series, periods: int = TRADING_DAYS,
           compound: bool = False) -> float:
    """Rendement annualisé / |max drawdown|, dans la même convention."""
    mdd = max_drawdown(returns, compound=compound)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return float(annual_return(returns, periods, compound=compound) / abs(mdd))


def rolling_sharpe(returns: pd.Series, window: int = 252,
                   periods: int = TRADING_DAYS) -> pd.Series:
    mu = returns.rolling(window).mean()
    sd = returns.rolling(window).std(ddof=1)
    return (np.sqrt(periods) * mu / sd).rename("rolling_sharpe")


# --------------------------------------------------------------------------- #
# Métriques basées trades                                                      #
# --------------------------------------------------------------------------- #
def trade_stats(trades: pd.DataFrame) -> dict:
    """Hit rate, gain/perte moyens, profit factor à partir du journal de trades.

    Durées : ``avg_holding_days`` en jours calendaires, ``avg_holding_bars`` en
    séances (même unité que la demi-vie et le time-stop).
    """
    if trades is None or trades.empty or "net" not in trades.columns:
        return {"n_trades": 0, "hit_rate": float("nan"),
                "avg_win": float("nan"), "avg_loss": float("nan"),
                "win_loss_ratio": float("nan"), "profit_factor": float("nan"),
                "avg_holding_days": float("nan"), "avg_holding_bars": float("nan")}
    net = trades["net"]
    wins = net[net > 0]
    losses = net[net < 0]
    gross_win = wins.sum()
    gross_loss = abs(losses.sum())
    return {
        "n_trades": int(len(trades)),
        "hit_rate": float((net > 0).mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "win_loss_ratio": float(wins.mean() / abs(losses.mean()))
        if len(wins) and len(losses) and losses.mean() != 0 else float("nan"),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "avg_holding_days": float(trades["holding_days"].mean())
        if "holding_days" in trades else float("nan"),
        "avg_holding_bars": float(trades["holding_bars"].mean())
        if "holding_bars" in trades else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Résumé complet                                                               #
# --------------------------------------------------------------------------- #
def summary(returns: pd.Series, trades: pd.DataFrame | None = None,
            periods: int = TRADING_DAYS, compound: bool = False,
            positions: pd.Series | None = None) -> dict:
    """Dictionnaire de toutes les métriques clés (pour tableau/dashboard).

    compound : convention de la série (cf. docstring du module).
    positions : si fourni, le temps investi est mesuré sur la position
        (sinon : jours à rendement non nul).
    """
    r = returns.fillna(0.0)
    in_market = (positions.fillna(0.0) != 0) if positions is not None else (r != 0)
    out = {
        "ann_return": annual_return(r, periods, compound=compound),
        "ann_vol": annual_vol(r, periods),
        "sharpe": sharpe(r, periods=periods),
        "sortino": sortino(r, periods=periods),
        "max_drawdown": max_drawdown(r, compound=compound),
        "calmar": calmar(r, periods, compound=compound),
        # P&L cumulé / notionnel (additif) ou performance totale (composé)
        "total_return": (float((1.0 + r).prod() - 1.0) if compound else float(r.sum()))
        if len(r) else float("nan"),
        "pct_time_in_market": float(in_market.mean()),
    }
    if trades is not None:
        out.update(trade_stats(trades))
    return out
