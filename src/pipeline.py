"""Orchestration de bout en bout : prix -> hedge -> signal -> backtest -> risque.

Source unique de vérité partagée par les scripts et le dashboard, pour que
les résultats affichés soient exactement ceux calculés (pas de divergence).
"""
from __future__ import annotations

import pandas as pd

from . import backtest as bt
from . import hedge_ratio as hr
from . import metrics as mt
from . import signal as sig
from . import sizing as sz
from .config import StrategyConfig


def run_strategy(
    prices: pd.DataFrame,
    cfg: StrategyConfig | None = None,
    vol_target: float | None = None,
    vol_lookback: int = 21,
    max_leverage: float = 3.0,
) -> dict:
    """Exécute la stratégie complète et renvoie tout le nécessaire au reporting.

    Parameters
    ----------
    vol_target : si non-None, applique l'overlay de vol-targeting (contrôle de
        risque) sur les rendements bruts et calcule aussi les métriques associées.

    Returns
    -------
    dict avec : hedge, signal, result (BacktestResult), trades, returns_raw,
        equity_raw, metrics_raw, et si vol_target : spread_ret, vol_scalar,
        returns_rm, equity_rm, trades_rm (journal re-valorisé), metrics_rm ;
        plus bh_* (buy & hold Brent, composé).
    """
    cfg = cfg or StrategyConfig()
    hedge_df = hr.compute_hedge(prices, cfg)
    signal_df = sig.build_signal(prices, hedge_df, cfg)
    res = bt._simulate(prices, signal_df, cfg, label=cfg.hedge_method)
    daily = res.daily

    r_raw = res.returns
    out = {
        "cfg": cfg,
        "prices": prices,
        "hedge": hedge_df,
        "signal": signal_df,
        "result": res,
        "trades": res.trades,
        "returns_raw": r_raw,
        "equity_raw": 1.0 + r_raw.cumsum(),
        "metrics_raw": mt.summary(r_raw, res.trades, positions=daily["position"]),
    }

    if vol_target is not None:
        # Vol mesurée sur le spread supposé toujours investi (et non sur r_raw,
        # nul quand on est flat -> levier max à chaque entrée). Cf. sizing.py.
        spread_ret = sz.spread_returns(daily["y"], daily["x"], daily["beta"])
        scalar = sz.vol_target_scalar(spread_ret, target_vol=vol_target,
                                      lookback=vol_lookback, max_leverage=max_leverage)
        pnl_rm = sz.vol_target_overlay(daily, scalar, cfg.cost_bps_per_leg)
        r_rm = pnl_rm["net"].rename("vt_ret")
        trades_rm = bt.trades_with_pnl(res.trades, pnl_rm, size=scalar)
        out["spread_ret"] = spread_ret
        out["vol_scalar"] = scalar
        out["returns_rm"] = r_rm
        out["equity_rm"] = 1.0 + r_rm.cumsum()
        out["trades_rm"] = trades_rm
        out["metrics_rm"] = mt.summary(r_rm, trades_rm, positions=daily["position"])

    # Buy & hold Brent (référence) : long directionnel -> composé (courbe ET métriques).
    bh = prices[cfg.price_col_y].pct_change().fillna(0.0)
    out["bh_returns"] = bh
    out["bh_equity"] = (1.0 + bh).cumprod()
    out["bh_metrics"] = mt.summary(bh, compound=True)
    return out


def regime_breakdown(returns: pd.Series, segments: list[dict]) -> pd.DataFrame:
    """Sharpe et P&L par segment de régime (pour la ventilation de perf)."""
    rows = []
    for seg in segments:
        sub = returns.loc[seg["start"]:seg["end"]].dropna()
        if len(sub) < 20:
            continue
        rows.append({
            "regime": seg["name"],
            "start": seg["start"], "end": seg["end"],
            "sharpe": mt.sharpe(sub),
            "ann_return": mt.annual_return(sub),
            "pnl": float(sub.sum()),
            "max_dd": mt.max_drawdown(sub, compound=False),
            "n_days": int(len(sub)),
        })
    return pd.DataFrame(rows)
