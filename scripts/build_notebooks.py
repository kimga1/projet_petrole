"""Génère les 4 notebooks d'analyse (exploration + résultats commentés).

Les notebooks sont construits programmatiquement (nbformat) pour rester
reproductibles, puis exécutés via nbconvert (voir README). Chaque notebook
importe la bibliothèque src/ et illustre une étape de la méthodologie.

Usage : python scripts/build_notebooks.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NB_DIR.mkdir(exist_ok=True)

PREAMBLE = """import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "..")
%matplotlib inline
import numpy as np, pandas as pd, matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 4)
plt.rcParams["axes.grid"] = True
from src import (data_loader as dl, features as ft, stats_tests as st,
                 hedge_ratio as hr, signal as sig, backtest as bt, metrics as mt,
                 curve as cv, regime as rg, pipeline as pl)
from src.config import StrategyConfig, MARKET_EVENTS, REGIME_SEGMENTS
def events(ax):
    for e in MARKET_EVENTS:
        ax.axvline(pd.Timestamp(e["date"]), color="gray", ls=":", lw=1, alpha=.6)
"""


def build(cells, name, title):
    nb = new_notebook()
    nb.cells = [new_markdown_cell(title)] + cells
    nb.metadata["kernelspec"] = {"name": "projet_petrole",
                                 "display_name": "projet_petrole", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nbf.write(nb, NB_DIR / name)
    print("écrit", name)


# =========================================================================== #
# 01 — Exploration
# =========================================================================== #
build([
    new_markdown_cell("**Objectif** : charger et aligner Brent/WTI, examiner rendements, "
                      "volatilité et spreads. On travaille sur les **futures front-month** "
                      "(base tradable) ; le spot FRED sert de robustesse."),
    new_code_cell(PREAMBLE),
    new_code_cell("prices = dl.get_prices('futures')\n"
                  "fred = dl.get_prices('fred')\n"
                  "print('futures', prices.shape, prices.index.min().date(), '->', prices.index.max().date())\n"
                  "ax = prices.plot(title='Prix futures front-month Brent (BZ=F) & WTI (CL=F)')\n"
                  "events(ax); plt.ylabel('USD/bbl'); plt.show()"),
    new_markdown_cell("### Rendements & volatilité\n"
                      "Rendements **log** (additifs). Volatilité réalisée 21 j annualisée (×√252)."),
    new_code_cell("rets = ft.log_returns(prices).dropna()\n"
                  "display(ft.descriptive_stats(rets).round(4))\n"
                  "vol = pd.DataFrame({c: ft.rolling_vol(rets[c], 21) for c in rets})\n"
                  "ax = (vol*100).plot(title='Volatilité réalisée 21j annualisée (%)'); events(ax); plt.show()"),
    new_markdown_cell("### Spread Brent − WTI\n"
                      "Spread de prix et log-spread. On visualise la moyenne mobile et les bandes ±2σ."),
    new_code_cell("spread = ft.price_spread(prices)\n"
                  "band = ft.rolling_mean_band(spread, 40, n_std=2.0)\n"
                  "ax = band[['spread','mean','upper','lower']].plot(title='Spread Brent-WTI ($/bbl) + bandes ±2σ (40j)')\n"
                  "events(ax); plt.show()\n"
                  "print('spread moyen %.2f, min %.2f, max %.2f' % (spread.mean(), spread.min(), spread.max()))"),
    new_markdown_cell("**Lecture** : le spread oscille autour d'un niveau mais avec des épisodes de "
                      "dérive persistante (2011-2015 discount WTI, 2020 choc COVID) — d'où la nécessité "
                      "de tester la cointégration (notebook 02) avant de trader la mean-reversion."),
], "01_exploration.ipynb", "# 01 · Exploration des données Brent–WTI")


# =========================================================================== #
# 02 — Cointégration
# =========================================================================== #
build([
    new_markdown_cell("**Objectif** : valider statistiquement la mean-reversion — les niveaux sont-ils "
                      "I(1) ? le spread cointégrant est-il stationnaire ? à quelle vitesse revient-il ?"),
    new_code_cell(PREAMBLE + "\nprices = dl.get_prices('futures')"),
    new_markdown_cell("### ADF — prix ≈ I(1) (à la limite), rendements I(0)\n"
                      "Sans tendance, le Brent rejette de justesse la racine unitaire (p ≈ 0,046) ; "
                      "avec tendance (`ct`), aucun des deux ne la rejette."),
    new_code_cell("for c in ['wti','brent']:\n"
                  "    print(c, 'niveau          :', st.adf_test(prices[c]))\n"
                  "    print(c, 'niveau+tendance :', st.adf_test(prices[c], regression='ct'))\n"
                  "    print(c, 'rendement       :', st.adf_test(ft.log_returns(prices[c]).dropna()))"),
    new_markdown_cell("### Cointégration Engle-Granger\n"
                      "OLS `Brent = α + β·WTI`, puis ADF sur le résidu + test de cointégration."),
    new_code_cell("cr = st.engle_granger(prices['brent'], prices['wti'])\n"
                  "print(cr); print('ADF résidu:', cr.adf_resid)\n"
                  "ax = cr.resid.plot(title='Résidu cointégrant (spread stationnaire)'); ax.axhline(0, color='k', lw=.8)\n"
                  "events(ax); plt.show()"),
    new_markdown_cell("### Vitesse de retour à la moyenne\n"
                      "Demi-vie (régression Δspread ~ spread₋₁ ; l'écart est multiplié par ρ = 1+φ "
                      "chaque jour, d'où demi-vie = −ln2/ln(1+φ)) et exposant de Hurst."),
    new_code_cell("hl = st.half_life(cr.resid); hu = st.hurst_exponent(cr.resid)\n"
                  "print(f'demi-vie = {hl:.1f} jours | Hurst = {hu:.2f} (<0.5 -> mean-reverting)')\n"
                  "cr.resid.hist(bins=60); plt.title('Distribution du résidu cointégrant'); plt.show()\n"
                  "print('-> lookback z-score fixé à ~2x la demi-vie ~ 40 jours')"),
    new_markdown_cell("### Cointégration glissante — où la relation tient-elle ?"),
    new_code_cell("pv = st.rolling_coint_pvalue(prices['brent'], prices['wti'], window=252, step=5)\n"
                  "ax = pv.plot(title='p-value de cointégration glissante (252j)'); ax.axhline(0.05, color='r', ls='--')\n"
                  "events(ax); plt.ylim(0, .5); plt.show()"),
    new_code_cell("for a, b in [('2007','2013'), ('2014','2019'), ('2020','2026')]:\n"
                  "    q = prices.loc[a:b]\n"
                  "    print(a, '-', b, ': p =', round(st.engle_granger(q['brent'], q['wti']).coint_pvalue, 3))"),
    new_markdown_cell("**Lecture** : cointégration sur l'ensemble (p = 0,013) mais **pas stable** : non "
                      "significative sur 2007-2013 (glut de Cushing), nette ensuite — motivant le hedge "
                      "dynamique et le suivi de régime."),
], "02_cointegration.ipynb", "# 02 · Cointégration & mean-reversion")


# =========================================================================== #
# 03 — Backtest
# =========================================================================== #
build([
    new_markdown_cell("**Objectif** : hedge dynamique (Kalman), signal z-score, backtest sans look-ahead, "
                      "contrôle de risque, et comparaison walk-forward / split simple."),
    new_code_cell(PREAMBLE + "\nprices = dl.get_prices('futures')\ncfg = StrategyConfig(hedge_method='kalman')"),
    new_markdown_cell("### Hedge ratio dynamique (Kalman) & signal"),
    new_code_cell("hedge = hr.compute_hedge(prices, cfg)\n"
                  "s = sig.build_signal(prices, hedge, cfg)\n"
                  "fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)\n"
                  "ax[0].plot(hedge.index, hedge['beta']); ax[0].set_title('β Kalman (hedge dynamique)'); events(ax[0])\n"
                  "ax[1].plot(s.index, s['zscore']); ax[1].axhline(2, color='r', ls='--'); ax[1].axhline(-2, color='r', ls='--')\n"
                  "ax[1].axhline(0, color='k', lw=.6); ax[1].set_title('z-score du spread'); events(ax[1]); plt.show()"),
    new_markdown_cell("### Backtest : brut vs risk-managed (vol-target 10 %) vs Buy & Hold"),
    new_code_cell("out = pl.run_strategy(prices, cfg, vol_target=0.10)\n"
                  "keys = ['sharpe','ann_return','ann_vol','max_drawdown']\n"
                  "print('BRUT        :', {k: round(out['metrics_raw'][k],3) for k in keys})\n"
                  "print('RISK-MANAGED:', {k: round(out['metrics_rm'][k],3) for k in keys})\n"
                  "print('B&H (CAGR)  :', {k: round(out['bh_metrics'][k],3) for k in keys})\n"
                  "fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={'height_ratios':[3,1]})\n"
                  "ax[0].plot(out['equity_rm'], label='Risk-managed', color='green')\n"
                  "ax[0].plot(out['equity_raw'], label='Brut', color='teal', ls=':')\n"
                  "ax[0].plot(out['bh_equity'], label='B&H Brent', color='gray', ls='--')\n"
                  "ax[0].legend(); ax[0].set_title('Courbe d équité'); events(ax[0])\n"
                  "dd = mt.drawdown_series(out['returns_rm'], compound=False)\n"
                  "ax[1].fill_between(dd.index, dd*100, 0, color='crimson', alpha=.5); ax[1].set_title('Drawdown (% du notionnel)'); plt.show()"),
    new_markdown_cell("### Garde-fou d'entrée : le trade du WTI négatif\n"
                      "Le 20/04/2020 le Kalman produit β < 0 : sans garde-fou, le trade ouvert le lendemain "
                      "est une vente sèche de Brent."),
    new_code_cell("print(hedge['beta'].loc['2020-04-15':'2020-04-24'].round(3))\n"
                  "s_ng = s.copy(); s_ng['position'] = sig.generate_signals(s['zscore'], cfg)  # sans garde-fou\n"
                  "r_ng = bt._simulate(prices, s_ng, cfg)\n"
                  "print('Sharpe brut sans garde-fou :', round(mt.sharpe(r_ng.returns),2), '| avec :', round(mt.sharpe(out['returns_raw']),2))\n"
                  "print(r_ng.trades.nsmallest(1, 'net')[['entry_date','side','beta','net']])"),
    new_markdown_cell("### Walk-forward continu vs split simple — sur les MÊMES périodes\n"
                      "Comparer l'in-sample d'une méthode (2008-2017) à l'out-of-sample (2018+) mélange "
                      "l'effet « méthode » et l'effet « période » : on compare donc les trois méthodes "
                      "sur les deux mêmes fenêtres."),
    new_code_cell("wf = bt.backtest_walk_forward_blocks(prices, cfg)\n"
                  "sp = bt.backtest_simple_split(prices, cfg, split='2018-01-01')\n"
                  "cut = pd.Timestamp('2018-01-01')\n"
                  "rows = {'Kalman en ligne': out['returns_raw'], 'blocs trimestriels': wf.returns,\n"
                  "        'split unique (β figé)': sp['result'].returns}\n"
                  "display(pd.DataFrame({k: {'< 2018': mt.sharpe(r[r.index < cut]), '>= 2018': mt.sharpe(r[r.index >= cut])}\n"
                  "                      for k, r in rows.items()}).T.round(2))"),
    new_markdown_cell("### Sensibilité (Sharpe risk-managed) — pas de sur-optimisation"),
    new_code_cell("from dataclasses import replace\n"
                  "z_ins=[1.5,2.0,2.5,3.0]; lbs=[20,30,40,60,90]\n"
                  "grid=pd.DataFrame(index=z_ins, columns=lbs, dtype=float)\n"
                  "for zi in z_ins:\n"
                  "    for lb in lbs:\n"
                  "        grid.loc[zi,lb]=mt.sharpe(pl.run_strategy(prices, replace(cfg,z_in=zi,z_lookback=lb), vol_target=0.10)['returns_rm'])\n"
                  "im=plt.imshow(grid.values.astype(float), cmap='viridis', aspect='auto')\n"
                  "plt.xticks(range(len(lbs)), lbs); plt.yticks(range(len(z_ins)), z_ins)\n"
                  "plt.xlabel('lookback'); plt.ylabel('z_in'); plt.colorbar(im, label='Sharpe'); plt.title('Sensibilité Sharpe'); plt.show()\n"
                  "display(grid.round(2))"),
    new_markdown_cell("### Performance par régime"),
    new_code_cell("rb = pl.regime_breakdown(out['returns_rm'], REGIME_SEGMENTS)\n"
                  "display(rb[['regime','sharpe','pnl','max_dd','n_days']].round(3))"),
    new_markdown_cell("**Lecture** : à période égale, seules les méthodes à β constant (split, blocs) perdent "
                      "leur performance après 2018 ; le Kalman en ligne la conserve (≈ 0,45 / 0,45). Une fois "
                      "le trade aberrant du WTI négatif écarté, le signal brut a un Sharpe ≈ 0,45 ; le "
                      "vol-target fixe le risque (vol ≈ 10 %) **sans changer le Sharpe**. Le vrai mode de "
                      "casse reste 2014-2016."),
], "03_backtest.ipynb", "# 03 · Hedge dynamique, signal & backtest")


# =========================================================================== #
# 04 — Courbe & régime
# =========================================================================== #
build([
    new_markdown_cell("**Objectif** : structure de courbe WTI (contango/backwardation, roll yield), lien "
                      "avec le spread, et détection de régime (cointégration glissante, ruptures)."),
    new_code_cell(PREAMBLE + "\nprices = dl.get_prices('futures')\ncurve = dl.get_curve()\n"
                  "print('courbe EIA :', None if curve is None else (curve.shape, str(curve.index.max().date())))"),
    new_markdown_cell("### Contango / backwardation & roll yield  (EIA RCLC1..4, ≤ 2024-04-05)"),
    new_code_cell("cf = cv.curve_features(curve)\n"
                  "ax = cf['slope'].plot(title='Pente de courbe C4-C1 (>0 contango, <0 backwardation)')\n"
                  "ax.axhline(0, color='k', lw=.8); events(ax); plt.show()\n"
                  "print('répartition régimes (%):'); print((cv.regime_share(cf)*100).round(1))\n"
                  "print('roll yield annualisé moyen : %.1f%%' % (cf['roll_yield_ann'].mean()*100))"),
    new_markdown_cell("### Lien courbe ↔ spread & rendements forward"),
    new_code_cell("link = cv.link_curve_to_spread(cf, ft.price_spread(prices))\n"
                  "print('corr pente/spread :', round(link['corr_slope_spread'],3))\n"
                  "print('spread moyen par régime de courbe:'); print(link['spread_by_regime'].round(2))\n"
                  "fwd = cv.curve_vs_forward_returns(cf, prices['wti'], horizon=21)\n"
                  "print('rendement WTI forward 21j par régime (%):'); print((fwd['fwd_ret_by_regime']*100).round(2))\n"
                  "print('t (backwardation − contango, sans recouvrement) : %.2f' % fwd['t_stat'])"),
    new_markdown_cell("### Détection de régime : cointégration glissante, β glissant, ruptures"),
    new_code_cell("ri = rg.regime_indicators(prices)\n"
                  "fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)\n"
                  "ax[0].plot(ri.index, ri['coint_pvalue']); ax[0].axhline(0.05, color='r', ls='--')\n"
                  "ax[0].set_title('p-value cointégration glissante'); ax[0].set_ylim(0,.5); events(ax[0])\n"
                  "ax[1].plot(ri.index, ri['rolling_beta']); ax[1].set_title('β glissant (rolling OLS 252j)'); events(ax[1])\n"
                  "for d in rg.cusum_breaks(ri['rolling_beta'].dropna(), threshold=8):\n"
                  "    ax[1].axvline(d, color='orange', lw=1)\n"
                  "plt.show()\n"
                  "print('ruptures CUSUM sur β :', [str(d.date()) for d in rg.cusum_breaks(ri['rolling_beta'].dropna(), threshold=8)])"),
    new_markdown_cell("**Lecture** : sur 1983-2024 le WTI est un peu plus souvent en backwardation (52 %), "
                      "mais 2004-2024 a été dominé par le contango (carry négatif). Le spread est plus large "
                      "en contango ; la courbe ne prédit pas de rendements forward significativement "
                      "différents (|t| < 2). La cointégration glissante et les "
                      "ruptures de β matérialisent **où la relation se déforme** — là où la stratégie doit être "
                      "prudente.\n\n*Limites : cutoff EIA 2024-04-05, pas de courbe Brent gratuite.*"),
], "04_curve_regime.ipynb", "# 04 · Structure de courbe & régime")

print("\nTous les notebooks générés dans", NB_DIR)
