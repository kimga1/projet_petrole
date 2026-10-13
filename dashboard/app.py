"""Dashboard Streamlit — Brent-WTI Statistical Arbitrage & Term Structure.

Ce fichier ORCHESTRE : état (langue, couleur secondaire, disco), chargements
cachés, sidebar de paramètres et les cinq onglets. Le reste est délégué :
- ui.py        jetons de design, template Plotly, gabarits HTML, widgets ;
- style.css    feuille de style du thème TERMINAL (sombre, monospace) ;
- i18n.py      traductions FR -> EN (bouton EN | FR, ?lang=fr) ;
- theme.py     préréglages de couleur secondaire (bouton COULEUR, disco) ;
- src/         tous les calculs (source unique de vérité : src.pipeline).

Design : style TERMINAL (inspiration Bloomberg) — gris profonds, couleur
secondaire au choix (ambre par défaut), données en blanc, vert/rouge marchés,
monospace intégrale. Uniquement des éléments fonctionnels.

Lancement local :  streamlit run dashboard/app.py
Déploiement     :  Streamlit Community Cloud (fichier principal = dashboard/app.py)
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src import curve as cv  # noqa: E402
from src import data_loader as dl  # noqa: E402
from src import features as ft  # noqa: E402
from src import metrics as mt  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import regime as rg  # noqa: E402
from src import stats_tests as st_  # noqa: E402
from src.config import REGIME_SEGMENTS, StrategyConfig  # noqa: E402

from i18n import make_tr  # noqa: E402
from theme import ACCENT_PRESETS, DEFAULT_ACCENT, hue_deg, resolve_accent  # noqa: E402
from ui import (  # noqa: E402
    BORDER,
    CARD_INSET,
    DIVIDER,
    GREEN,
    LEGEND_TR,
    NEUTRAL,
    RED,
    TXT,
    TXT_BODY,
    TXT_FAINT,
    TXT_MUTED,
    accent_style,
    add_event_lines,
    card,
    inject_sidebar_reopener,
    kpi_card,
    load_base_css,
    param_slider,
    pills_or_radio,
    progress_row,
    setup_plotly_template,
    side_badge,
)

st.set_page_config(page_title="Brent-WTI Stat-Arb", layout="wide",
                   initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- #
# Couleur secondaire (préréglages theme.py) + mode disco.                     #
# État lu ICI, avant le template Plotly et le CSS. ?accent=<nom>&disco=1      #
# dans l'URL préconfigure (lien partageable). ACCENT / ACCENT_LT sont les     #
# hex du préréglage (Plotly ne résout pas les variables CSS) ; l'interface    #
# passe par var(--accent), que le mode disco anime côté client (le détail     #
# est documenté dans ui.accent_style).                                        #
# --------------------------------------------------------------------------- #
try:
    _qp = {k: str(v) for k, v in st.query_params.items()}
except Exception:                          # vieux Streamlit sans query_params
    _qp = {}
if "accent_pick" not in st.session_state:
    _wanted = _qp.get("accent", "").strip().lower()
    st.session_state.accent_pick = next(
        (n for n in ACCENT_PRESETS if n.lower() == _wanted), DEFAULT_ACCENT)
if "disco" not in st.session_state:
    st.session_state.disco = _qp.get("disco", "").lower() in ("1", "true", "on")
ACCENT, ACCENT_LT, ACCENT_RGB = resolve_accent(st.session_state.get("accent_pick"))
DISCO = bool(st.session_state.get("disco"))

setup_plotly_template(ACCENT)
st.markdown(f"<style>{load_base_css()}</style>", unsafe_allow_html=True)
st.markdown(accent_style(ACCENT, ACCENT_LT, ACCENT_RGB, hue_deg(ACCENT), DISCO),
            unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Chargement (caché)                                                          #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_prices(source: str) -> pd.DataFrame:
    return dl.get_prices(source)


@st.cache_data(show_spinner=False)
def load_curve() -> pd.DataFrame | None:
    return dl.get_curve()


@st.cache_data(show_spinner=False)
def compute_regime(source: str) -> pd.DataFrame:
    return rg.regime_indicators(load_prices(source))


@st.cache_data(show_spinner=False)
def pair_stats(source: str) -> dict:
    prices = load_prices(source)
    cr = st_.engle_granger(prices["brent"], prices["wti"])
    return {
        "beta": cr.beta, "coint_p": cr.coint_pvalue,
        "adf_resid_p": cr.adf_resid.pvalue,
        "is_coint": cr.is_cointegrated,
        "half_life": st_.half_life(cr.resid),
        "hurst": st_.hurst_exponent(cr.resid),
    }


@st.cache_data(show_spinner=False)
def run_strategy_cached(source, hedge_method, z_in, z_out, z_lookback, z_stop,
                        half_spread_bps, commission_bps, vol_target,
                        kalman_delta, ols_window, max_holding_days) -> dict:
    prices = load_prices(source)
    cfg = StrategyConfig(
        hedge_method=hedge_method, z_in=z_in, z_out=z_out, z_lookback=z_lookback,
        z_stop=z_stop, half_spread_bps=half_spread_bps, commission_bps=commission_bps,
        kalman_delta=kalman_delta, ols_window=ols_window,
        max_holding_days=(max_holding_days if max_holding_days and max_holding_days > 0 else None),
    )
    vt = vol_target if vol_target and vol_target > 0 else None
    return pl.run_strategy(prices, cfg, vol_target=vt)


@st.cache_data(show_spinner=False)
def sensitivity_grid(source, vol_target) -> pd.DataFrame:
    prices = load_prices(source)
    z_ins = [1.5, 2.0, 2.5, 3.0]
    lbs = [20, 30, 40, 60, 90]
    grid = pd.DataFrame(index=z_ins, columns=lbs, dtype=float)
    base = StrategyConfig(hedge_method="kalman")
    for zi in z_ins:
        for lb in lbs:
            r = pl.run_strategy(prices, replace(base, z_in=zi, z_lookback=lb),
                                vol_target=vol_target or None)
            key = "returns_rm" if vol_target else "returns_raw"
            grid.loc[zi, lb] = mt.sharpe(r[key])
    return grid.astype(float)


@st.cache_data(show_spinner=False)
def load_results_json(name: str) -> dict | None:
    p = config.RESULTS_DIR / name
    if p.exists():
        return json.loads(p.read_text())
    return None


# --------------------------------------------------------------------------- #
# Langue : anglais par défaut, sélecteur EN | FR en haut de page.             #
# ?lang=fr au chargement force le français (lien partageable). Le français    #
# du code est la référence ; i18n.py traduit à l'affichage.                   #
# --------------------------------------------------------------------------- #
if "lang_pick" not in st.session_state:
    try:
        _q = str(st.query_params.get("lang", "") or "").lower()
    except Exception:                     # vieux Streamlit sans query_params
        _q = ""
    st.session_state.lang_pick = "FR" if _q.startswith("fr") else "EN"
lang = "fr" if st.session_state.get("lang_pick") == "FR" else "en"
L = make_tr(lang)
_top_cols = st.columns([6.6, 1.4, 2])
with _top_cols[2]:
    if hasattr(st, "pills"):
        st.pills("Language", ["EN", "FR"], key="lang_pick",
                 label_visibility="collapsed")
    else:
        st.radio("Language", ["EN", "FR"], key="lang_pick", horizontal=True,
                 label_visibility="collapsed")
with _top_cols[1]:
    # Couleur secondaire + mode disco (l'état a été lu en tête de script).
    pop = (st.popover(L("Couleur")) if hasattr(st, "popover")
           else st.sidebar.expander(L("Couleur")))
    with pop:
        st.markdown(f'<div class="bw-micro">{L("Couleur secondaire")}</div>',
                    unsafe_allow_html=True)
        if hasattr(st, "pills"):
            st.pills(L("Couleur secondaire"), list(ACCENT_PRESETS),
                     key="accent_pick", format_func=L,
                     label_visibility="collapsed")
        else:
            st.radio(L("Couleur secondaire"), list(ACCENT_PRESETS),
                     key="accent_pick", format_func=L, horizontal=True,
                     label_visibility="collapsed")
        if hasattr(st, "toggle"):
            st.toggle("Disco", key="disco")
        else:
            st.checkbox("Disco", key="disco")

# --------------------------------------------------------------------------- #
# Sidebar : wordmark + carte PARAMÈTRES façon mockup                          #
# --------------------------------------------------------------------------- #
st.sidebar.markdown(
    f'<div class="bb-banner" style="font-size:11px;padding:3px 8px">'
    f'Brent–WTI Stat-Arb</div>'
    f'<div class="bw-ctx" style="margin:5px 0 8px">'
    f'{L("Terminal démo · moteur Kalman pairs")}</div>',
    unsafe_allow_html=True)

SOURCES = {"Futures front-month": "futures", "FRED spot": "fred"}

with st.sidebar.container(border=True, key="panel-params"):
    st.markdown(f'<div class="bw-micro" style="margin-bottom:4px">{L("Paramètres")}</div>',
                unsafe_allow_html=True)

    st.markdown(f'<p style="color:{TXT_MUTED};font-size:11.5px;margin:0 0 2px">'
                f'{L("Source de prix")}</p>', unsafe_allow_html=True)
    src_label = pills_or_radio("Source", list(SOURCES), "Futures front-month",
                               key="src", format_func=L)
    source = SOURCES[src_label]

    st.markdown(f'<p style="color:{TXT_MUTED};font-size:11.5px;margin:4px 0 2px">'
                f'Hedge ratio</p>', unsafe_allow_html=True)
    _hopts = ["kalman", "rolling_ols", "ols"]
    _hfmt = {"kalman": "Kalman", "rolling_ols": "rOLS", "ols": "OLS"}
    if hasattr(st, "segmented_control"):
        hedge_method = st.segmented_control(
            "Hedge ratio", _hopts, default="kalman",
            format_func=lambda m: _hfmt[m], label_visibility="collapsed") or "kalman"
    elif hasattr(st, "pills"):
        hedge_method = st.pills("Hedge ratio", _hopts, default="kalman",
                                format_func=lambda m: _hfmt[m],
                                label_visibility="collapsed") or "kalman"
    else:
        hedge_method = st.selectbox("Hedge ratio", _hopts)

    z_lookback = param_slider("z lookback", "p_lb", 10, 120, 40, 5,
                              lambda v: f"{v} {L('j')}")
    z_in = param_slider(L("entrée |z|"), "p_zin", 1.0, 4.0, 2.0, 0.1,
                        lambda v: f"{v:.1f}")
    z_out = param_slider(L("sortie |z|"), "p_zout", 0.0, 2.0, 0.5, 0.1,
                         lambda v: f"{v:.1f}")
    max_hold = param_slider("time-stop", "p_hold", 0, 250, 90, 10,
                            lambda v: f"{v} {L('j')}" if v else "off")
    cost_leg = param_slider(L("coûts / jambe"), "p_cost", 0.0, 15.0, 2.5, 0.5,
                            lambda v: f"{v:.1f} bps")

    st.markdown(f'<div style="border-top:1px solid {DIVIDER};margin:10px 0 4px"></div>',
                unsafe_allow_html=True)
    use_vt = (st.toggle("Vol-targeting", value=True)
              if hasattr(st, "toggle") else st.checkbox("Vol-targeting", value=True))
    vt_default = 0.10
    vol_target = float(st.session_state.get("p_vt", vt_default)) if use_vt else 0.0
    if use_vt:
        # st.caption (et non un <div> nu) : hauteur correctement comptée par le
        # conteneur markdown -> pas de débordement sous la bordure du panneau.
        st.caption(L("cible {v} % ann.").format(v=f"{vol_target*100:.0f}"))

adv = st.sidebar.expander(L("Avancé"))
with adv:
    if use_vt:
        st.slider(L("Vol cible annualisée"), 0.04, 0.25, vt_default, 0.01, key="p_vt")
    use_stop = st.checkbox(L("Stop sur z (déconseillé — cf. README)"), value=False,
                           key="p_use_stop")
    z_stop = (st.slider(L("Seuil stop |z|"), 3.0, 8.0, 5.0, 0.5, key="p_zstop")
              if use_stop else None)
    kalman_delta = st.select_slider(L("Kalman δ (réactivité β)"),
                                    options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3], value=1e-4,
                                    key="p_kdelta")
    ols_window = st.slider(L("Fenêtre rolling OLS (j)"), 20, 250, 60, 10, key="p_olsw")

_footer = L("Data : yfinance · FRED · EIA<br>Démo. Pas un conseil d'investissement.")
st.sidebar.markdown(
    f'<div style="color:{TXT_FAINT};font-size:10px;margin-top:12px">{_footer}</div>',
    unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Bandeau titre terminal (seul en haut : aucun widget adjacent -> aucun        #
# chevauchement possible ; le sélecteur de source vit dans la sidebar)         #
# --------------------------------------------------------------------------- #
st.markdown(
    f'<div class="bb-banner"><span>Brent–WTI Crude Stat-Arb</span>'
    f'<span>{L("Relative Value · Cointégration · 2007-2026")}</span></div>',
    unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Calcul                                                                       #
# --------------------------------------------------------------------------- #
prices = load_prices(source)
out = run_strategy_cached(source, hedge_method, z_in, z_out, z_lookback, z_stop,
                          cost_leg, 0.0, vol_target, kalman_delta,
                          ols_window, max_hold)
sig = out["signal"]
ret_key = "returns_rm" if "returns_rm" in out else "returns_raw"
eq_key = "equity_rm" if "equity_rm" in out else "equity_raw"
metrics = out.get("metrics_rm", out["metrics_raw"])
ps = pair_stats(source)

st.markdown(f'<div class="bw-ctx">{prices.index.min().date()} → '
            f'{prices.index.max().date()} · {len(prices):,} {L("séances")} · '
            f'hedge {hedge_method} · '
            f'vol-target {"ON (%.0f %%)" % (vol_target*100) if use_vt else "OFF"}</div>',
            unsafe_allow_html=True)

tabs = st.tabs([L("1 VUE D'ENSEMBLE"), L("2 COINTÉGRATION"), L("3 COURBE WTI"),
                "4 BACKTEST", "5 TRADES"])

# =========================================================================== #
# 1) VUE D'ENSEMBLE                                                            #
# =========================================================================== #
with tabs[0]:
    last_z = float(sig["zscore"].dropna().iloc[-1])
    last_pos = int(np.sign(float(sig["position"].dropna().iloc[-1])))
    pos_txt = {1: "LONG", -1: "SHORT", 0: "FLAT"}[last_pos]
    pos_color = {1: GREEN, -1: RED, 0: NEUTRAL}[last_pos]

    pnl_cum = float(out[eq_key].dropna().iloc[-1] - 1.0)
    ann, shp, mdd = metrics["ann_return"], metrics["sharpe"], metrics["max_drawdown"]

    hero_c, vit_c = st.columns([1.45, 2])
    with hero_c:
        st.markdown(
            f'<div class="bw-hero">'
            f'<div class="bw-micro">{L("P&L cumulé · notionnel constant")}</div>'
            f'<div class="val">{pnl_cum*100:+.1f} %</div>'
            f'<div style="margin-top:7px">'
            f'<span class="bw-badge {"up" if ann >= 0 else "down"}">{ann*100:+.1f} {L("% / an")}</span>'
            f'<span class="bw-badge {"up" if shp >= 0 else "down"}">Sharpe {shp:+.2f}</span>'
            f'<span class="bw-badge down">DD {mdd*100:.1f} %</span></div>'
            f'<div class="sub">{prices.index.min().date()} → {prices.index.max().date()} · '
            f'{len(prices):,} {L("séances")} · {L("notionnel constant")}</div></div>',
            unsafe_allow_html=True)
    with vit_c:
        vit = [
            (L("β cointégrant"), f"{ps['beta']:.3f}", L("hedge dynamique"),
             TXT_FAINT, ""),
            ("p-value EG", f"{ps['coint_p']:.3f}",
             L("cointégrés ✓") if ps["is_coint"] else L("non cointégrés"),
             GREEN if ps["is_coint"] else RED, ""),
            (L("Demi-vie"), f"{ps['half_life']:.0f} {L('j')}",
             L("horizon de détention"), TXT_FAINT, ""),
            ("Hurst", f"{ps['hurst']:.2f}",
             "mean-reverting" if ps["hurst"] < 0.5 else L("trendant"),
             GREEN if ps["hurst"] < 0.5 else RED, ""),
            (L("Signal du jour"), f"z {last_z:+.1f}", pos_txt, pos_color,
             "border-color:var(--accent);"),
        ]
        cells = "".join(
            f'<div class="bw-card" style="flex:1 1 104px;min-width:104px;'
            f'padding:10px 12px;{extra}">'
            f'<div class="mono" style="color:{TXT};font-size:13.5px;font-weight:600;'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{v}</div>'
            f'<div style="color:{sc};font-size:10px;font-weight:600;margin:3px 0 6px;'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{s}</div>'
            f'<div style="border-top:1px solid {DIVIDER};padding-top:5px" class="bw-micro">{lbl}</div>'
            f'</div>'
            for lbl, v, s, sc, extra in vit)
        st.markdown(
            f'<div class="bw-card" style="padding:11px 12px">'
            f'<div class="bw-micro" style="margin-bottom:7px">Spread vitals</div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:8px">{cells}</div></div>',
            unsafe_allow_html=True)

    with card(L("Spread cointégrant & bandes ±{v}σ").format(v=f"{z_in:.1f}")):
        band = ft.rolling_mean_band(sig["resid"], z_lookback, n_std=z_in)
        PERIODS = {"1A": 252, "3A": 756, "5A": 1260, "10A": 2520, "Max": None}
        sel = pills_or_radio(L("Période"), list(PERIODS), "5A", key="period",
                             format_func=L)
        nd = PERIODS[sel]
        band_v = band if nd is None else band.iloc[-nd:]

        figs = go.Figure()
        figs.add_trace(go.Scatter(x=band_v.index, y=band_v["upper"], line=dict(width=0),
                                  showlegend=False, hoverinfo="skip"))
        figs.add_trace(go.Scatter(x=band_v.index, y=band_v["lower"], fill="tonexty",
                                  fillcolor=f"rgba({ACCENT_RGB},0.12)", line=dict(width=0),
                                  name=f"±{z_in:.1f}σ"))
        figs.add_trace(go.Scatter(x=band_v.index, y=band_v["mean"], name=L("Moyenne mobile"),
                                  line=dict(color=NEUTRAL, dash="dash", width=1)))
        figs.add_trace(go.Scatter(x=band_v.index, y=band_v["spread"], name="Spread",
                                  line=dict(color=ACCENT, width=2)))
        if sel == "Max":
            add_event_lines(figs)
        figs.update_layout(height=300, margin=dict(l=0, r=0, t=8, b=0), legend=LEGEND_TR)
        st.plotly_chart(figs, width='stretch')

    tbl_c, legs_c = st.columns([2, 1])
    with tbl_c:
        rb = pl.regime_breakdown(out[ret_key], REGIME_SEGMENTS)
        rows = "".join(
            f'<tr><td>{L(r.regime)}</td>'
            f'<td class="num" style="color:{GREEN if r.sharpe >= 0 else RED};font-weight:700">'
            f'{r.sharpe:+.2f}</td>'
            f'<td class="num">{r.ann_return*100:+.1f} %</td>'
            f'<td class="num" style="color:{RED}">{r.max_dd*100:.1f} %</td></tr>'
            for r in rb.itertuples())
        st.markdown(
            f'<div class="bw-card"><div class="bw-ct">{L("Performance par régime")}</div>'
            f'<table class="bw-table">'
            f'<tr><th>{L("Régime")}</th><th>Sharpe</th><th>{L("Rend. ann.")}</th>'
            f'<th>Max DD</th></tr>'
            f'{rows}</table></div>', unsafe_allow_html=True)
    with legs_c:
        chg_b = prices["brent"].iloc[-1] / prices["brent"].iloc[-2] - 1
        chg_w = prices["wti"].iloc[-1] / prices["wti"].iloc[-2] - 1
        spread_now = float(sig["resid"].dropna().iloc[-1])
        legs = [
            ("BZ", "Brent front-month", f"{prices['brent'].iloc[-1]:.2f} $",
             f"{chg_b*100:+.2f} %", GREEN if chg_b >= 0 else RED),
            ("CL", f"WTI × {out['signal']['beta'].dropna().iloc[-1]:.3f}",
             f"{prices['wti'].iloc[-1]:.2f} $",
             f"{chg_w*100:+.2f} %", GREEN if chg_w >= 0 else RED),
            ("Δ", "Spread (y − βx)", f"{spread_now:.2f}",
             f"z {last_z:+.2f}", pos_color),
        ]
        rows = "".join(
            f'<div style="display:flex;align-items:center;gap:9px;padding:7px 0;'
            f'border-bottom:1px solid {DIVIDER}">'
            f'<span style="width:30px;height:30px;border-radius:9px;background:{CARD_INSET};'
            f'border:1px solid {BORDER};display:inline-flex;align-items:center;'
            f'justify-content:center;color:var(--accent-lt);font-size:10px;font-weight:700">{t}</span>'
            f'<span style="flex:1"><span style="color:{TXT};font-size:12px;font-weight:600">{n}</span></span>'
            f'<span style="text-align:right"><span class="mono" style="color:{TXT};font-size:12px">{p}</span><br>'
            f'<span class="mono" style="color:{c};font-size:10px">{ch}</span></span></div>'
            for t, n, p, ch, c in legs)
        next_action = L("Pas d'entrée tant que |z| &lt; {zi}. Sortie |z| &lt; {zo} "
                        "· time-stop {mh} séances.").format(
            zi=f"{z_in:.1f}", zo=f"{z_out:.1f}", mh=max_hold)
        st.markdown(
            f'<div class="bw-card"><div class="bw-ct">{L("Jambes du jour")}</div>{rows}'
            f'<div class="bw-nested" style="margin-top:9px">'
            f'<div class="bw-micro">{L("Prochaine action")}</div>'
            f'<div style="color:{TXT_BODY};font-size:11px;margin-top:4px;line-height:1.5">'
            f'{next_action}</div></div></div>',
            unsafe_allow_html=True)

    with st.expander(L("Historique des prix Brent & WTI")):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices.index, y=prices["brent"], name="Brent",
                                 line=dict(color=ACCENT, width=1.6)))
        fig.add_trace(go.Scatter(x=prices.index, y=prices["wti"], name="WTI",
                                 line=dict(color=NEUTRAL, width=1.4)))
        add_event_lines(fig)
        fig.update_layout(height=240, margin=dict(l=0, r=0, t=8, b=0), legend=LEGEND_TR)
        st.plotly_chart(fig, width='stretch')

# =========================================================================== #
# 2) COINTÉGRATION & RÉGIME                                                    #
# =========================================================================== #
with tabs[1]:
    k = st.columns(5)
    k[0].markdown(kpi_card(L("β cointégrant"), f"{ps['beta']:.3f}"), unsafe_allow_html=True)
    k[1].markdown(kpi_card("p-value EG", f"{ps['coint_p']:.3f}",
                           L("cointégrés ✓") if ps["is_coint"] else L("non cointégrés"),
                           GREEN if ps["is_coint"] else RED), unsafe_allow_html=True)
    k[2].markdown(kpi_card(L("ADF résidu"), f"{ps['adf_resid_p']:.3f}",
                           L("stationnaire ✓") if ps["adf_resid_p"] < 0.05
                           else L("non stationnaire"),
                           GREEN if ps["adf_resid_p"] < 0.05 else RED), unsafe_allow_html=True)
    k[3].markdown(kpi_card(L("Demi-vie"), f"{ps['half_life']:.0f} {L('j')}"),
                  unsafe_allow_html=True)
    k[4].markdown(kpi_card("Hurst", f"{ps['hurst']:.2f}",
                           "mean-reverting" if ps["hurst"] < 0.5 else L("trendant"),
                           GREEN if ps["hurst"] < 0.5 else RED), unsafe_allow_html=True)

    with st.spinner(L("Cointégration glissante...")):
        ri = compute_regime(source)
    with card(L("p-value de cointégration glissante · fenêtre 252 j")):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ri.index, y=ri["coint_pvalue"], line_color=ACCENT,
                                 name="p-value"))
        fig.add_hline(y=0.05, line_dash="dash", line_color=RED, annotation_text="5 %")
        fig.add_vrect(x0="2014-07-01", x1="2016-12-31", fillcolor="rgba(255,68,56,0.07)",
                      line_width=0, annotation_text=L("2014-16 · rupture (shale + export ban)"),
                      annotation_position="top left",
                      annotation_font=dict(size=9.5, color=RED))
        add_event_lines(fig)
        fig.update_layout(height=220, margin=dict(l=0, r=0, t=8, b=0), yaxis_range=[0, 0.5])
        st.plotly_chart(fig, width='stretch')

    cc = st.columns(2)
    with cc[0]:
        with card(L("β glissant (hedge dynamique)")):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sig.index, y=sig["beta"], line_color=ACCENT,
                                     name=f"β ({hedge_method})"))
            fig.add_trace(go.Scatter(x=ri.index, y=ri["rolling_beta"],
                                     line=dict(color=NEUTRAL, dash="dot", width=1),
                                     name=L("β rolling OLS 252 j")))
            add_event_lines(fig)
            fig.update_layout(height=210, margin=dict(l=0, r=0, t=8, b=0), legend=LEGEND_TR)
            st.plotly_chart(fig, width='stretch')
    with cc[1]:
        with card(L("Demi-vie glissante")):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ri.index, y=ri["rolling_half_life"], line_color=ACCENT,
                                     name=L("demi-vie (j)")))
            fig.add_hline(y=ps["half_life"], line_dash="dash", line_color=GREEN,
                          annotation_text=L("{v} j (plein échantillon)")
                          .format(v=f"{ps['half_life']:.0f}"))
            fig.update_layout(height=210, margin=dict(l=0, r=0, t=8, b=0),
                              yaxis_range=[0, 150])
            st.plotly_chart(fig, width='stretch')

    with card(L("Ruptures (CUSUM sur β) & flag « tradable »")):
        flag = ri["tradeable"].astype(float).fillna(0.0)
        fig = go.Figure(go.Heatmap(
            x=flag.index, y=[""], z=[flag.values], showscale=False, zmin=0, zmax=1,
            colorscale=[[0, "rgba(255,68,56,0.5)"], [1, "rgba(0,200,110,0.35)"]],
            hovertemplate="%{x|%Y-%m-%d} · %{z}<extra></extra>"))
        for d in rg.cusum_breaks(ri["rolling_beta"].dropna(), threshold=8):
            fig.add_vline(x=d, line_color=RED, line_width=1.5)
        for d, lbl in [("2015-12-18", "export ban levé"), ("2020-04-20", "WTI < 0"),
                       ("2022-02-24", "Ukraine")]:
            fig.add_annotation(x=pd.Timestamp(d), y=-0.9, yref="paper", showarrow=False,
                               text=L(lbl), font=dict(size=9, color=TXT_FAINT))
        fig.update_layout(height=100, margin=dict(l=0, r=0, t=4, b=24),
                          yaxis=dict(showticklabels=False))
        st.plotly_chart(fig, width='stretch')
        st.caption(L("Vert = cointégré & demi-vie exploitable · rouge = relation suspecte "
                     "· traits rouges = ruptures CUSUM (diagnostic a posteriori, non causal)."))

# =========================================================================== #
# 3) COURBE WTI                                                                #
# =========================================================================== #
with tabs[2]:
    curve = load_curve()
    if curve is None:
        st.warning(L("Courbe EIA indisponible (ajouter EIA_API_KEY dans .env / secrets Streamlit)."))
    else:
        cf = cv.curve_features(curve)
        eia_note = L("L'EIA a arrêté la publication des séries RCLC après le 2024-04-05 "
                     "— cet onglet est borné à cette date.")
        st.markdown(f'<div class="bw-warn">! &nbsp;{eia_note}</div>',
                    unsafe_allow_html=True)

        valid = curve.dropna()
        cc = st.columns([1.3, 1])
        with cc[0]:
            with card(None):
                sel = st.select_slider(L("Date de la courbe"), options=list(valid.index),
                                       value=valid.index[-1], key="curve_date",
                                       format_func=lambda d: str(pd.Timestamp(d).date()))
                row = curve.loc[sel]
                slope = row["c4"] - row["c1"]
                reg = "Backwardation" if slope < 0 else "Contango"
                reg_kind = "up" if slope < 0 else "blue"
                st.markdown(
                    f'<div class="bw-ct" style="display:inline">{L("Courbe au")} '
                    f'{pd.Timestamp(sel).date()}</div> &nbsp;'
                    f'<span class="bw-badge {reg_kind}">{reg} · C4−C1 = {slope:+.2f}</span>',
                    unsafe_allow_html=True)
                fig = go.Figure(go.Scatter(
                    x=["C1", "C2", "C3", "C4"],
                    y=[row["c1"], row["c2"], row["c3"], row["c4"]],
                    mode="lines+markers+text",
                    text=[f"{row[c]:.2f}" for c in ("c1", "c2", "c3", "c4")],
                    textposition="top center",
                    textfont=dict(family="IBM Plex Mono", size=10, color=TXT_BODY),
                    line=dict(color=ACCENT, width=2),
                    marker=dict(symbol="circle-open", size=8,
                                line=dict(width=2.5, color=ACCENT))))
                fig.update_layout(height=250, margin=dict(l=0, r=0, t=18, b=0))
                st.plotly_chart(fig, width='stretch')
        with cc[1]:
            share = (cv.regime_share(cf) * 100)
            bars = "".join([
                progress_row("Backwardation", float(share.get("backwardation", 0)), GREEN),
                progress_row("Contango", float(share.get("contango", 0)), "var(--accent)"),
                progress_row(L("Plat"), float(share.get("flat", 0)), NEUTRAL),
            ])
            roll_txt = L("Roll yield annualisé moyen ≈ {v} % sur l'historique complet ; "
                         "la sous-période 2004-2024 était nettement en contango (≈ −9 %) "
                         "— biais de fenêtre corrigé par la pagination EIA.").format(
                v=f'{cf["roll_yield_ann"].mean()*100:.1f}')
            st.markdown(
                f'<div class="bw-card"><div class="bw-ct">{L("Répartition des régimes")} · '
                f'{cf.index.min().year}–{cf.index.max().year}</div>{bars}'
                f'<div class="bw-nested" style="margin-top:9px">'
                f'<div style="color:{TXT_BODY};font-size:11px;line-height:1.5">'
                f'{roll_txt}</div></div></div>', unsafe_allow_html=True)

        with card(L("Pente C4 − C1")):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cf.index, y=cf["slope"], line_color=ACCENT,
                                     name=L("pente")))
            fig.add_hline(y=0, line_dash="dash", line_color="#4A4A4A")
            fig.update_layout(height=200, margin=dict(l=0, r=0, t=8, b=0))
            st.plotly_chart(fig, width='stretch')
            st.caption(L("Au-dessus de 0 = contango · en dessous = backwardation."))

        link = cv.link_curve_to_spread(cf, ft.price_spread(prices))
        fwd = cv.curve_vs_forward_returns(cf, prices["wti"], horizon=21)
        fb = fwd["fwd_ret_by_regime"]
        s_by = link["spread_by_regime"]
        k = st.columns(3)
        k[0].markdown(kpi_card(L("Corr. pente ↔ spread"), f"{link['corr_slope_spread']:+.3f}",
                               f"{L('niveaux, descriptif')} · "
                               f"{link['start'].year}–{link['end'].year}"),
                      unsafe_allow_html=True)
        k[1].markdown(kpi_card(L("Spread moyen par régime"),
                               f"{s_by.get('contango', float('nan')):.2f} / "
                               f"{s_by.get('backwardation', float('nan')):.2f}",
                               "contango / backwardation"), unsafe_allow_html=True)
        t_fwd = fwd.get("t_stat", float("nan"))
        k[2].markdown(kpi_card(L("Rendement WTI forward 21 j"),
                               f"{fb.get('backwardation', float('nan'))*100:+.2f} % / "
                               f"{fb.get('contango', float('nan'))*100:+.2f} %",
                               f"backw. / contango · t = {t_fwd:+.1f} "
                               f"({L('non significatif') if abs(t_fwd) < 2 else L('significatif')})"),
                      unsafe_allow_html=True)

# =========================================================================== #
# 4) BACKTEST                                                                  #
# =========================================================================== #
with tabs[3]:
    k = st.columns(6)
    k[0].markdown(kpi_card("Sharpe", f"{metrics['sharpe']:+.2f}", None,
                           GREEN if metrics["sharpe"] >= 0 else RED), unsafe_allow_html=True)
    k[1].markdown(kpi_card(L("Rend. annualisé"), f"{metrics['ann_return']*100:+.1f} %"),
                  unsafe_allow_html=True)
    k[2].markdown(kpi_card(L("Vol annualisée"), f"{metrics['ann_vol']*100:.1f} %"),
                  unsafe_allow_html=True)
    k[3].markdown(kpi_card("Max DD", f"{metrics['max_drawdown']*100:.1f} %"),
                  unsafe_allow_html=True)
    k[4].markdown(kpi_card("Trades", f"{metrics['n_trades']}"), unsafe_allow_html=True)
    k[5].markdown(kpi_card("Hit rate", f"{metrics['hit_rate']*100:.0f} %"),
                  unsafe_allow_html=True)

    with card(L("Équité · P&L cumulé, notionnel constant")):
        fig = go.Figure()
        if "equity_rm" in out:
            fig.add_trace(go.Scatter(x=out["equity_rm"].index, y=out["equity_rm"],
                                     name="Risk-managed",
                                     line=dict(color=GREEN, width=2.2),
                                     fill="tozeroy", fillcolor="rgba(0,200,110,0.05)"))
        fig.add_trace(go.Scatter(x=out["equity_raw"].index, y=out["equity_raw"],
                                 name=L("Brut"), line=dict(color=ACCENT, dash="dot", width=1.4)))
        fig.add_trace(go.Scatter(x=out["bh_equity"].index, y=out["bh_equity"],
                                 name="B&H Brent",
                                 line=dict(color=NEUTRAL, dash="dash", width=1.2)))
        # Labels d'événements DANS la zone de tracé (y=0.97, ancrés en haut) :
        # la légende vit au-dessus du graphe -> aucune superposition possible.
        # Hauteurs alternées : deux événements proches (COVID 2020 / Ukraine
        # 2022) ne peuvent pas se chevaucher, même à faible largeur.
        for i, (d, lbl) in enumerate([("2015-12-18", "export ban levé"),
                                      ("2020-04-20", "COVID"),
                                      ("2022-02-24", "Ukraine")]):
            fig.add_vline(x=pd.Timestamp(d), line_width=1, line_dash="dot",
                          line_color="#4A4A4A")
            fig.add_annotation(x=pd.Timestamp(d), y=0.99 - 0.08 * (i % 2),
                               yref="paper", yanchor="top", showarrow=False,
                               text=L(lbl), font=dict(size=9, color=TXT_FAINT))
        fig.update_layout(height=290, margin=dict(l=0, r=0, t=24, b=0), legend=LEGEND_TR)
        st.plotly_chart(fig, width='stretch')

    left, right = st.columns([2, 1.15])
    with left:
        with card("Drawdown"):
            dd = mt.drawdown_series(out[ret_key], compound=False)
            fig = go.Figure(go.Scatter(x=dd.index, y=dd * 100,
                                       name=L("Drawdown (% du notionnel)"),
                                       line=dict(color=RED, width=1.4),
                                       fill="tozeroy", fillcolor="rgba(255,68,56,0.25)"))
            fig.update_layout(height=170, margin=dict(l=0, r=0, t=6, b=0))
            st.plotly_chart(fig, width='stretch')
            st.caption(L("Pire drawdown : {v} % du notionnel (perte depuis le pic, "
                         "même unité que la vol et le rendement).")
                       .format(v=f"{metrics['max_drawdown']*100:.1f}"))

        with card(L("Sharpe glissant · 1 an")):
            rs = mt.rolling_sharpe(out[ret_key], 252)
            fig = go.Figure(go.Scatter(x=rs.index, y=rs, line_color=ACCENT,
                                       name=L("Sharpe glissant")))
            fig.add_hline(y=0, line_dash="dash", line_color="#4A4A4A")
            fig.update_layout(height=160, margin=dict(l=0, r=0, t=6, b=0))
            st.plotly_chart(fig, width='stretch')

        wf = load_results_json("wf_vs_split.json")
        if wf and "online_kalman_is_sharpe" in wf:
            rows = "".join(
                f'<tr><td>{name}</td>'
                f'<td class="num">{wf[k + "_is_sharpe"]:+.2f}</td>'
                f'<td class="num" style="color:{GREEN if wf[k + "_oos_sharpe"] >= 0.3 else RED};'
                f'font-weight:700">{wf[k + "_oos_sharpe"]:+.2f}</td></tr>'
                for name, k in [(L("Kalman en ligne (β chaque jour)"), "online_kalman"),
                                (L("Blocs (β ré-estimé / trimestre)"), "wf_blocks"),
                                (L("Split unique (β estimé avant 2018)"), "simple_split")])
            wf_note = L("Comparées sur les mêmes fenêtres, seules les méthodes à β figé "
                        "perdent leur performance hors échantillon ; le Kalman en ligne "
                        "la conserve.")
            st.markdown(
                f'<div class="bw-card"><div class="bw-ct">'
                f'{L("Walk-forward vs split unique · Sharpe brut, mêmes périodes")}</div>'
                f'<table class="bw-table"><tr><th>{L("Méthode")}</th><th>&lt; 2018</th>'
                f'<th>≥ 2018</th></tr>{rows}</table>'
                f'<div style="color:{TXT_MUTED};font-size:11px;margin-top:7px;line-height:1.5">'
                f'{wf_note}</div></div>',
                unsafe_allow_html=True)
    with right:
        with card(L("Sharpe par régime")):
            rb = pl.regime_breakdown(out[ret_key], REGIME_SEGMENTS)
            fig = go.Figure(go.Bar(
                x=rb["sharpe"], y=[L(r) for r in rb["regime"]], orientation="h",
                marker_color=[GREEN if s >= 0 else RED for s in rb["sharpe"]],
                text=[f"{s:+.2f}" for s in rb["sharpe"]], textposition="outside",
                textfont=dict(family="IBM Plex Mono", size=10)))
            fig.update_layout(height=250, margin=dict(l=0, r=26, t=6, b=0),
                              yaxis=dict(autorange="reversed", tickfont=dict(size=9.5)),
                              xaxis=dict(range=[min(-1, rb["sharpe"].min() - .4),
                                                max(2, rb["sharpe"].max() + .5)]))
            st.plotly_chart(fig, width='stretch')
            mode1 = L("<b>1. Structurel</b> (2014-16, export ban) : la relation se "
                      "déplace durablement — problème de signal/régime.")
            mode2 = L("<b>2. Prix aberrant</b> (20/04/2020, WTI à −37 $) : le β du "
                      "Kalman devient négatif — neutralisé par le garde-fou d'entrée "
                      "(prix &gt; 0, β ∈ [0,5 ; 1,5]).")
            st.markdown(
                f'<div class="bw-nested"><div class="bw-micro">{L("Deux modes de casse")}</div>'
                f'<div style="color:{TXT_BODY};font-size:11px;margin-top:4px;line-height:1.55">'
                f'{mode1}<br>{mode2}</div></div>', unsafe_allow_html=True)

# =========================================================================== #
# 5) TRADES & SENSIBILITÉ                                                      #
# =========================================================================== #
with tabs[4]:
    # Journal cohérent avec les KPI : re-valorisé par le vol-target s'il est actif.
    trades = out.get("trades_rm", out["trades"])
    ts = mt.trade_stats(trades)

    hm_c, cost_c = st.columns([1.5, 1])
    with hm_c:
        with card(L("Sensibilité du Sharpe · z d'entrée × lookback")):
            grid = sensitivity_grid(source, vol_target if use_vt else 0.0)
            x_lbl = [f"{c} {L('j')}" for c in grid.columns]
            y_lbl = [f"z {i}" for i in grid.index]
            fig = go.Figure(go.Heatmap(
                z=grid.values, x=x_lbl, y=y_lbl,
                colorscale=[[0, f"rgba({ACCENT_RGB},0.10)"], [1, f"rgba({ACCENT_RGB},0.85)"]],
                xgap=5, ygap=5,
                text=grid.round(2).values, texttemplate="%{text}",
                textfont=dict(family="IBM Plex Mono", size=11),
                colorbar=dict(title="Sharpe", thickness=7, outlinewidth=0,
                              tickfont=dict(size=9.5))))
            try:
                dv = grid.loc[2.0, 40]
                fig.add_annotation(x=f"40 {L('j')}", y="z 2.0", text=f"{dv:.2f} ●",
                                   showarrow=False,
                                   font=dict(family="IBM Plex Mono", size=11, color="#fff"),
                                   bordercolor="#fff", borderwidth=2, borderpad=4,
                                   bgcolor=CARD_INSET)
            except Exception:
                pass
            fig.update_layout(height=260, margin=dict(l=0, r=0, t=6, b=0),
                              xaxis_title="lookback", yaxis_title=L("z d'entrée"))
            # key -> classe st-key-heat_sens : cible du filtre disco (raster).
            st.plotly_chart(fig, width='stretch', key="heat_sens")
            st.caption(L("Surface lisse et positive sur une large plage — le défaut "
                         "(z 2.0 × 40 j, marqué ●) est un choix principé ≈ 2× la "
                         "demi-vie, pas le maximum de la grille."))
    with cost_c:
        with card(L("Sensibilité aux coûts")):
            costs_path = config.RESULTS_DIR / "sensitivity_costs.csv"
            if costs_path.exists():
                costs_df = pd.read_csv(costs_path)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=costs_df["cost_bps_per_leg"], y=costs_df["sharpe"],
                                         mode="lines+markers",
                                         line=dict(color=ACCENT, width=2),
                                         marker=dict(size=5), name="Sharpe"))
                near = costs_df.iloc[(costs_df["cost_bps_per_leg"] - 2.5).abs().argmin()]
                fig.add_trace(go.Scatter(x=[near["cost_bps_per_leg"]], y=[near["sharpe"]],
                                         mode="markers", showlegend=False,
                                         marker=dict(size=10, symbol="circle-open",
                                                     line=dict(width=2.5, color="#fff"))))
                fig.update_layout(height=200, margin=dict(l=0, r=0, t=6, b=0),
                                  xaxis_title=L("coûts (bps/jambe)"), yaxis_title="Sharpe")
                st.plotly_chart(fig, width='stretch')
        st.markdown(
            f'<div class="bw-card"><div class="bw-micro">{L("Stats de trades")}</div>'
            f'<div class="mono" style="color:{TXT_BODY};font-size:11px;margin-top:5px;line-height:1.7">'
            f'{ts["n_trades"]} trades · hit {ts["hit_rate"]*100:.0f} %<br>'
            f'{L("gain moy.")} <span style="color:{GREEN}">{ts["avg_win"]*100:+.1f} %</span> · '
            f'{L("perte moy.")} <span style="color:{RED}">{ts["avg_loss"]*100:+.1f} %</span><br>'
            f'profit factor {ts["profit_factor"]:.2f} · '
            f'{L("détention moy.")} {ts["avg_holding_bars"]:.0f} {L("séances")}</div></div>',
            unsafe_allow_html=True)

    if not trades.empty:
        recent = trades.tail(6).iloc[::-1]
        rows = "".join(
            f'<tr><td class="num">{pd.Timestamp(r.entry_date).date()}</td>'
            f'<td>{side_badge(int(r.side))}</td>'
            f'<td class="num">{r.entry_z:+.2f}</td>'
            f'<td class="num">{(("%+.2f" % r.exit_z) if pd.notna(r.exit_z) else "—")}</td>'
            f'<td class="num">{r.beta:.3f}</td>'
            f'<td class="num">{int(r.holding_bars)}</td>'
            f'<td class="num" style="text-align:right;font-weight:700;'
            f'color:{GREEN if r.net >= 0 else RED}">{r.net*100:+.2f} %</td></tr>'
            for r in recent.itertuples())
        journal_title = L("Journal de trades · {n} plus récents sur {total}").format(
            n=min(6, len(trades)), total=len(trades))
        st.markdown(
            f'<div class="bw-card"><div class="bw-ct">{journal_title}</div>'
            f'<table class="bw-table">'
            f'<tr><th>{L("Entrée")}</th><th>{L("Sens")}</th><th>{L("z entrée")}</th>'
            f'<th>{L("z sortie")}</th><th>β</th><th>{L("Séances")}</th>'
            f'<th style="text-align:right">{L("P&L net")}</th></tr>'
            f'{rows}</table></div>', unsafe_allow_html=True)
        with st.expander(L("Voir les {n} trades").format(n=len(trades))):
            show = trades.copy()
            for col in ("entry_date", "exit_date"):
                if col in show:
                    show[col] = pd.to_datetime(show[col]).dt.date
            num = [c for c in ("gross", "cost", "net", "entry_z", "exit_z", "beta", "size")
                   if c in show]
            show[num] = show[num].round(4)
            st.dataframe(show, width='stretch', height=300, hide_index=True)
    else:
        st.write(L("Aucun trade sur la période."))

# --------------------------------------------------------------------------- #
# Barre de statut terminal                                                     #
# --------------------------------------------------------------------------- #
_status = L("Data yfinance · FRED · EIA &nbsp;|&nbsp; trade-on-close · coûts {c} "
            "bps/jambe &nbsp;|&nbsp; walk-forward · zéro look-ahead &nbsp;|&nbsp; "
            "démo — pas un conseil d'investissement").format(c=f"{cost_leg:.1f}")
st.markdown(f'<div class="bb-status">{_status}</div>', unsafe_allow_html=True)

inject_sidebar_reopener(L("Rouvrir les paramètres"))
