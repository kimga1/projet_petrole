"""Exécute l'analyse complète et fige les résultats dans results/.

Produit les artefacts reproductibles consommés par le rapport, les notebooks
et (par défaut) le dashboard :
  - stats.json            : ADF, cointégration, demi-vie, Hurst
  - metrics.json          : métriques brut / risk-managed / buy&hold
  - equity.parquet        : courbes d'équité + drawdowns
  - trades.csv            : journal de trades (brut) ; trades_rm.csv : après vol-target
  - regime_breakdown.csv  : Sharpe & P&L par régime (risk-managed + brut)
  - pnl_by_year.csv       : P&L net par année
  - sensitivity_*.csv     : Sharpe sur grilles (z_in × lookback, coûts,
                            décalage d'exécution, bruit d'observation du Kalman)
  - wf_vs_split.json      : walk-forward vs split simple, à période égale
  - curve_*.{parquet,json}: structure de courbe WTI (si EIA dispo)

Usage : python scripts/run_backtest.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src import curve as cv  # noqa: E402
from src import data_loader as dl  # noqa: E402
from src import features as ft  # noqa: E402
from src import metrics as mt  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import stats_tests as st  # noqa: E402
from src.config import REGIME_SEGMENTS, StrategyConfig  # noqa: E402

RESULTS = config.RESULTS_DIR
VOL_TARGET = 0.10


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj.date())
    return obj


def main() -> None:
    print(">> Chargement des prix (futures)...")
    prices = dl.get_prices("futures")
    cfg = StrategyConfig(hedge_method="kalman")

    # --- 1) Tests statistiques ---
    print(">> Tests statistiques (ADF, cointégration, demi-vie)...")
    cr = st.engle_granger(prices["brent"], prices["wti"])
    logspread = ft.log_spread(prices)
    adf = {}
    for c in ("wti", "brent"):
        lvl = st.adf_test(prices[c])
        adf[f"adf_{c}"] = {
            "stat": lvl.stat, "pvalue": lvl.pvalue,
            # avec tendance : p-value nettement plus élevée -> racine unitaire non rejetée
            "pvalue_trend": st.adf_test(prices[c], regression="ct").pvalue,
            "pvalue_log": st.adf_test(np.log(prices[c].where(prices[c] > 0)).dropna()).pvalue,
        }
    # Cointégration par sous-période : la relation n'est pas stable dans le temps.
    coint_sub = {}
    for a, b in [("2007", "2013"), ("2014", "2019"), ("2020", "2026")]:
        q = prices.loc[a:b]
        coint_sub[f"{a}-{b}"] = st.engle_granger(q["brent"], q["wti"]).coint_pvalue
    # Demi-vie du spread réellement tradé (y − β_t·x, β Kalman) : son NIVEAU dérive
    # (α non retranché), d'où le centrage par moyenne glissante dans le z-score.
    traded = pl.run_strategy(prices, cfg)["signal"]["resid"].iloc[cfg.warmup:]
    stats = {
        **adf,
        "adf_ret_wti": {"pvalue": st.adf_test(ft.log_returns(prices["wti"]).dropna()).pvalue},
        "adf_ret_brent": {"pvalue": st.adf_test(ft.log_returns(prices["brent"]).dropna()).pvalue},
        "coint_beta": cr.beta, "coint_alpha": cr.alpha,
        "coint_pvalue": cr.coint_pvalue, "is_cointegrated": cr.is_cointegrated,
        "coint_pvalue_by_period": coint_sub,
        "adf_resid_pvalue": cr.adf_resid.pvalue,
        "half_life_resid": st.half_life(cr.resid),
        "half_life_logspread": st.half_life(logspread),
        "half_life_traded_spread": st.half_life(traded),
        "hurst_resid": st.hurst_exponent(cr.resid),
        "sample_start": str(prices.index.min().date()),
        "sample_end": str(prices.index.max().date()),
        "n_obs": int(len(prices)),
    }
    (RESULTS / "stats.json").write_text(json.dumps(_json_safe(stats), indent=2))

    # --- 2) Backtest complet (brut + risk-managed) ---
    print(">> Backtest (Kalman online + overlay vol-target)...")
    out = pl.run_strategy(prices, cfg, vol_target=VOL_TARGET)
    metrics = {
        "raw": out["metrics_raw"],
        "risk_managed": out["metrics_rm"],
        "buy_hold_brent": out["bh_metrics"],
        "vol_target": VOL_TARGET,
        "config": cfg.to_dict(),
    }
    (RESULTS / "metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2))

    equity = pd.DataFrame({
        "equity_raw": out["equity_raw"],
        "equity_rm": out["equity_rm"],
        "bh_equity": out["bh_equity"],
        "dd_raw": mt.drawdown_series(out["returns_raw"], compound=False),
        "dd_rm": mt.drawdown_series(out["returns_rm"], compound=False),
        "zscore": out["signal"]["zscore"],
        "position": out["signal"]["position"],
        "beta": out["signal"]["beta"],
        "spread": out["signal"]["resid"],
    })
    equity.to_parquet(RESULTS / "equity.parquet")
    out["trades"].to_csv(RESULTS / "trades.csv", index=False)
    out["trades_rm"].to_csv(RESULTS / "trades_rm.csv", index=False)

    # --- 3) Ventilation par régime + P&L annuel ---
    print(">> Ventilation par régime & P&L annuel...")
    rb = pl.regime_breakdown(out["returns_rm"], REGIME_SEGMENTS)
    rb_raw = pl.regime_breakdown(out["returns_raw"], REGIME_SEGMENTS)
    rb["sharpe_raw"] = rb_raw["sharpe"].to_numpy()
    rb["pnl_raw"] = rb_raw["pnl"].to_numpy()
    rb.to_csv(RESULTS / "regime_breakdown.csv", index=False)
    yr = out["returns_rm"].groupby(out["returns_rm"].index.year).sum()
    yr_raw = out["returns_raw"].groupby(out["returns_raw"].index.year).sum()
    pd.DataFrame({"pnl_risk_managed": yr, "pnl_raw": yr_raw}).to_csv(RESULTS / "pnl_by_year.csv")

    # --- 4) Sensibilité (Sharpe, risk-managed) ---
    print(">> Étude de sensibilité (z_in × z_lookback)...")
    z_ins = [1.5, 2.0, 2.5, 3.0]
    lbs = [20, 30, 40, 60, 90]
    grid = pd.DataFrame(index=z_ins, columns=lbs, dtype=float)
    for zi in z_ins:
        for lb in lbs:
            r = pl.run_strategy(prices, replace(cfg, z_in=zi, z_lookback=lb),
                                vol_target=VOL_TARGET)["returns_rm"]
            grid.loc[zi, lb] = mt.sharpe(r)
    grid.to_csv(RESULTS / "sensitivity_zin_lookback.csv")

    # sensibilité aux coûts
    cost_rows = []
    for c in [0.0, 1.0, 2.5, 5.0, 10.0]:
        r = pl.run_strategy(prices, replace(cfg, half_spread_bps=c, commission_bps=0.0),
                            vol_target=VOL_TARGET)["returns_rm"]
        cost_rows.append({"cost_bps_per_leg": c, "sharpe": mt.sharpe(r), "pnl": float(r.sum())})
    pd.DataFrame(cost_rows).to_csv(RESULTS / "sensitivity_costs.csv", index=False)

    # sensibilité au décalage d'exécution (1 = trade-on-close, 2 = clôture suivante)
    lag_rows = []
    for lag in [1, 2, 3]:
        o = pl.run_strategy(prices, replace(cfg, execution_lag=lag), vol_target=VOL_TARGET)
        lag_rows.append({"execution_lag": lag,
                         "sharpe_raw": mt.sharpe(o["returns_raw"]),
                         "sharpe_rm": mt.sharpe(o["returns_rm"]),
                         "ann_return_rm": o["metrics_rm"]["ann_return"],
                         "n_trades": o["metrics_rm"]["n_trades"]})
    pd.DataFrame(lag_rows).to_csv(RESULTS / "sensitivity_execution.csv", index=False)

    # sensibilité au bruit d'observation du Kalman (transparence : NON optimisé)
    kal_rows = []
    for obs in [1e-3, 1e-2, 1e-1, 1.0, 25.0]:
        o = pl.run_strategy(prices, replace(cfg, kalman_obs_cov=obs), vol_target=VOL_TARGET)
        b = o["hedge"]["beta"].iloc[cfg.warmup:]
        kal_rows.append({"kalman_obs_cov": obs,
                         "beta_min": float(b.min()), "beta_max": float(b.max()),
                         "sharpe_raw": mt.sharpe(o["returns_raw"]),
                         "sharpe_rm": mt.sharpe(o["returns_rm"]),
                         "n_trades": o["metrics_rm"]["n_trades"]})
    pd.DataFrame(kal_rows).to_csv(RESULTS / "sensitivity_kalman.csv", index=False)

    # --- 5) Comparaison walk-forward vs split simple ---
    print(">> Comparaison walk-forward / split simple...")
    from src import backtest as bt
    split = pd.Timestamp("2018-01-01")
    wf = bt.backtest_walk_forward_blocks(prices, cfg)
    sp = bt.backtest_simple_split(prices, cfg, split=split)

    def _is_oos(r):
        return mt.sharpe(r.loc[r.index < split]), mt.sharpe(r.loc[r.index >= split])

    on_is, on_oos = _is_oos(out["returns_raw"])
    wf_is, wf_oos = _is_oos(wf.returns)
    # Même fenêtre IS (< 2018) et OOS (>= 2018) pour les trois méthodes : la
    # comparaison ne mélange plus l'effet « méthode » et l'effet « période ».
    comp = {
        "split_date": split,
        "online_kalman_sharpe": mt.sharpe(out["returns_raw"]),
        "online_kalman_vt_sharpe": mt.sharpe(out["returns_rm"]),
        "wf_blocks_sharpe": mt.sharpe(wf.returns),
        "simple_split_is_sharpe": mt.sharpe(sp["is_returns"]),
        "simple_split_oos_sharpe": mt.sharpe(sp["oos_returns"]),
        "online_kalman_is_sharpe": on_is,
        "online_kalman_oos_sharpe": on_oos,
        "wf_blocks_is_sharpe": wf_is,
        "wf_blocks_oos_sharpe": wf_oos,
    }
    (RESULTS / "wf_vs_split.json").write_text(json.dumps(_json_safe(comp), indent=2))

    # --- 6) Structure de courbe WTI (si EIA dispo) ---
    curve = dl.get_curve()
    if curve is not None:
        print(">> Structure de courbe WTI...")
        cf = cv.curve_features(curve)
        cf.to_parquet(RESULTS / "curve_features.parquet")
        link = cv.link_curve_to_spread(cf, ft.price_spread(prices))
        fwd = cv.curve_vs_forward_returns(cf, prices["wti"], horizon=21)
        curve_json = {
            "regime_share": (cv.regime_share(cf) * 100).round(2).to_dict(),
            "n_days_unknown_regime": int(cf["regime"].isna().sum()),
            "roll_yield_ann_mean": float(cf["roll_yield_ann"].mean()),
            "roll_yield_ann_mean_2004": float(cf["roll_yield_ann"].loc["2004":].mean()),
            "corr_slope_spread": link["corr_slope_spread"],
            "spread_by_regime": link["spread_by_regime"].round(3).to_dict(),
            "link_start": link["start"], "link_end": link["end"],
            "fwd_ret_by_regime": (fwd["fwd_ret_by_regime"] * 100).round(3).to_dict(),
            "fwd_n_by_regime": fwd["n_by_regime"].to_dict(),
            "fwd_diff_back_minus_contango_pct": fwd["diff_back_minus_contango"] * 100,
            "fwd_t_stat_nonoverlap": fwd["t_stat"],
            "curve_start": str(cf.index.min().date()),
            "curve_end": str(cf.index.max().date()),
        }
        (RESULTS / "curve_summary.json").write_text(json.dumps(_json_safe(curve_json), indent=2))

    # --- Résumé console ---
    print("\n========== RÉSUMÉ ==========")
    print(f"Cointégration : beta={cr.beta:.3f} p={cr.coint_pvalue:.4f} "
          f"demi-vie={stats['half_life_resid']:.1f}j Hurst={stats['hurst_resid']:.2f}")
    m = out["metrics_rm"]
    print(f"Brut          : Sharpe={out['metrics_raw']['sharpe']:.2f}")
    print(f"Risk-managed  : Sharpe={m['sharpe']:.2f} ann={m['ann_return']*100:.1f}% "
          f"vol={m['ann_vol']*100:.1f}% maxDD={m['max_drawdown']*100:.1f}% (notionnel) "
          f"trades={m['n_trades']} hit={m['hit_rate']*100:.0f}%")
    print(f"Artefacts écrits dans {RESULTS}")


if __name__ == "__main__":
    main()
