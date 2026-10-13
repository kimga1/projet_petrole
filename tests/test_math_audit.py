"""AUDIT MATHÉMATIQUE — chaque calcul est recalculé par une voie INDÉPENDANTE.

Principe : on ne fait pas confiance au code, on le contre-vérifie.
- P&L : reconstruction trade par trade depuis les prix bruts (unités, β verrouillé,
  coûts sur turnover) et comparaison au journal du moteur.
- Métriques : Sharpe/vol/Sortino/drawdown/hit-rate recalculés à la main.
- Vol-target : rendement du spread toujours investi, scalaire (fenêtre décalée
  -> causalité) et overlay (taille, coûts d'entrée/sortie/rééquilibrage)
  recalculés en boucle.
- Kalman / demi-vie / Hurst : vérité terrain synthétique.
- Courbe : pente, roll yield, parts de régime recalculés.

Les tests sur données réelles lisent le cache parquet (skip si absent).
"""
import numpy as np
import pandas as pd
import pytest

from src import curve as cv
from src import data_loader as dl
from src import features as ft
from src import hedge_ratio as hr
from src import metrics as mt
from src import pipeline as pl
from src import stats_tests as st_
from src.config import REGIME_SEGMENTS, StrategyConfig

ATOL = 1e-9


# --------------------------------------------------------------------------- #
# Fixtures données réelles                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def prices():
    df = dl.load_cache("prices_futures")
    if df is None:
        pytest.skip("cache futures absent")
    return dl.align_prices(df)


@pytest.fixture(scope="module")
def run(prices):
    return pl.run_strategy(prices, StrategyConfig(), vol_target=0.10)


# --------------------------------------------------------------------------- #
# 1) P&L : reconstruction indépendante trade par trade                         #
# --------------------------------------------------------------------------- #
def test_trade_pnl_reconstruction(run):
    """Recalcule chaque trade depuis les prix bruts et compare au moteur."""
    cfg = run["cfg"]
    daily = run["result"].daily
    trades = run["trades"]
    assert not trades.empty

    idx = daily.index
    y = daily["y"].to_numpy()
    x = daily["x"].to_numpy()
    beta = daily["beta"].to_numpy()
    cost_leg = cfg.cost_bps_per_leg / 1e4
    pos = {d: i for i, d in enumerate(idx)}

    has_open_col = "open_at_end" in trades.columns

    for tr in trades.itertuples():
        t0 = pos[pd.Timestamp(tr.entry_date)]
        te = pos[pd.Timestamp(tr.exit_date)]
        sign = float(tr.side)
        is_open = has_open_col and tr.open_at_end is True

        b_lock = beta[t0 - 1]                       # β connu au signal (lag=1)
        assert np.isclose(b_lock, tr.beta, atol=1e-9), "β verrouillé ≠ journal"
        gross_entry = abs(y[t0 - 1]) + abs(b_lock) * abs(x[t0 - 1])
        scale = 1.0 / gross_entry
        bu = sign * scale                           # bbl Brent
        wu = -sign * b_lock * scale                 # bbl WTI
        # |entry_z| > z_in : toute entrée est un vrai franchissement de seuil
        # (aucune position héritée de l'amorçage).
        assert abs(tr.entry_z) > cfg.z_in
        # Garde-fou : β plausible et prix > 0 à la date du signal.
        assert cfg.beta_min <= b_lock <= cfg.beta_max
        assert y[t0 - 1] > 0 and x[t0 - 1] > 0

        # P&L brut : somme des variations pendant la détention. Trade fermé :
        # t0 .. te-1 (le jour de sortie, les unités sont déjà nulles) ;
        # trade encore ouvert : jusqu'à la fin de l'échantillon.
        end = len(idx) if is_open else te
        gross = sum(bu * (y[t] - y[t - 1]) + wu * (x[t] - x[t - 1])
                    for t in range(t0, end))
        assert np.isclose(gross, tr.gross, atol=1e-9), \
            f"P&L brut reconstruit {gross:.6g} ≠ moteur {tr.gross:.6g}"
        assert tr.holding_bars == end - t0
        # Notionnel détenu chaque jour, valorisé à la clôture de la veille.
        notional = daily["notional"].to_numpy()
        for t in range(t0, end):
            assert np.isclose(notional[t], abs(bu) * abs(y[t - 1]) + abs(wu) * abs(x[t - 1]),
                              atol=1e-12)

        # Coûts : entrée (prix t0-1) + sortie (prix te-1) sur le notionnel absolu.
        cost = cost_leg * (abs(bu) * abs(y[t0 - 1]) + abs(wu) * abs(x[t0 - 1]))
        if not is_open:
            cost += cost_leg * (abs(bu) * abs(y[te - 1]) + abs(wu) * abs(x[te - 1]))
        assert np.isclose(cost, tr.cost, atol=1e-9), \
            f"coût reconstruit {cost:.6g} ≠ moteur {tr.cost:.6g}"
        assert np.isclose(tr.net, tr.gross - tr.cost, atol=1e-12)


def test_daily_identities(run):
    """net = brut - coût ; équité additive = 1 + cumsum(net) ; somme trades = somme daily."""
    d = run["result"].daily
    assert np.allclose(d["net_ret"], d["gross_ret"] - d["cost"], atol=1e-12)
    assert np.allclose(d["equity"], 1.0 + d["net_ret"].cumsum(), atol=1e-12)
    assert np.isclose(run["trades"]["net"].sum(), d["net_ret"].sum(), atol=1e-8)


# --------------------------------------------------------------------------- #
# 2) Vol-targeting : causalité et identité                                     #
# --------------------------------------------------------------------------- #
def test_vol_target_manual(run):
    """Vol mesurée sur le spread TOUJOURS investi (pas sur r_raw, nul hors
    position), puis overlay jour par jour recalculé en boucle."""
    d = run["result"].daily
    cost_leg = run["cfg"].cost_bps_per_leg / 1e4
    y, x, b = d["y"].to_numpy(), d["x"].to_numpy(), d["beta"].to_numpy()
    n = len(d)

    # 1) rendement du long spread (notionnel 1, β de la veille)
    sr = np.full(n, np.nan)
    for t in range(1, n):
        g = abs(y[t - 1]) + abs(b[t - 1]) * abs(x[t - 1])
        sr[t] = (y[t] - y[t - 1] - b[t - 1] * (x[t] - x[t - 1])) / g
    sr = pd.Series(sr, index=d.index)
    assert np.allclose(sr, run["spread_ret"], atol=1e-12, equal_nan=True)

    # 2) multiplicateur : vol réalisée jusqu'à t-1 (causal)
    manual = (0.10 / (sr.rolling(21).std(ddof=1) * np.sqrt(252)).shift(1)) \
        .clip(lower=0.0, upper=3.0).fillna(0.0)
    assert np.allclose(manual, run["vol_scalar"], atol=1e-12)

    # 3) overlay : taille s_t tant qu'on détient la position, coût de sortie à la
    #    taille de la veille, rééquilibrage |s_t − s_{t-1}| × notionnel en cours.
    s, pos = manual.to_numpy(), d["position"].to_numpy()
    net, notional = d["net_ret"].to_numpy(), d["notional"].to_numpy()
    rm = np.zeros(n)
    for t in range(1, n):
        if pos[t] != 0:
            rm[t] = s[t] * net[t]
            if pos[t - 1] == pos[t]:
                rm[t] -= cost_leg * abs(s[t] - s[t - 1]) * notional[t]
        else:
            rm[t] = s[t - 1] * net[t]
    assert np.allclose(rm, run["returns_rm"], atol=1e-12)


def test_vol_target_hits_target(run):
    """Régression du bug « vol mesurée sur des zéros » : la vol réalisée en
    position doit rester proche de la cible (elle était ~2,2× trop haute)."""
    d = run["result"].daily
    vol_in_pos = run["returns_rm"][d["position"] != 0].std(ddof=1) * np.sqrt(252)
    assert 0.07 < vol_in_pos < 0.15
    at_cap = (run["vol_scalar"][d["position"] != 0] >= 3.0 - 1e-9).mean()
    assert at_cap < 0.10


def test_rm_trade_journal(run):
    """Journal risk-managed : P&L par trade = somme de returns_rm sur la fenêtre."""
    tr, rm = run["trades_rm"], run["returns_rm"]
    for e, x_, net in zip(tr["entry_date"], tr["exit_date"], tr["net"]):
        assert np.isclose(net, rm.loc[pd.Timestamp(e):pd.Timestamp(x_)].sum(), atol=1e-12)
    assert np.isclose(tr["net"].sum(), rm.sum(), atol=1e-9)   # aucun P&L hors trade
    assert np.allclose(tr["net"], tr["gross"] - tr["cost"], atol=1e-12)


# --------------------------------------------------------------------------- #
# 3) Métriques recalculées à la main                                           #
# --------------------------------------------------------------------------- #
def test_metrics_manual(run):
    r = run["returns_rm"].fillna(0.0)
    m = run["metrics_rm"]
    assert np.isclose(m["sharpe"], np.sqrt(252) * r.mean() / r.std(ddof=1), atol=1e-10)
    assert np.isclose(m["ann_return"], r.mean() * 252, atol=1e-12)
    assert np.isclose(m["ann_vol"], r.std(ddof=1) * np.sqrt(252), atol=1e-12)
    eq = 1.0 + r.cumsum()
    # drawdown en fraction du notionnel (même unité que vol et rendement)
    assert np.isclose(m["max_drawdown"], (eq - eq.cummax()).min(), atol=1e-12)
    assert np.isclose(m["calmar"], m["ann_return"] / abs(m["max_drawdown"]), atol=1e-12)
    assert np.isclose(m["total_return"], r.sum(), atol=1e-12)
    # Sortino : écart à la baisse sur TOUS les jours (jours positifs = 0)
    downside = np.sqrt((np.minimum(r, 0.0) ** 2).mean())
    assert np.isclose(m["sortino"], np.sqrt(252) * r.mean() / downside, atol=1e-10)


def test_trade_stats_manual(run):
    tr = run["trades"]
    m = mt.trade_stats(tr)
    net = tr["net"]
    wins, losses = net[net > 0], net[net < 0]
    assert m["n_trades"] == len(tr)
    assert np.isclose(m["hit_rate"], (net > 0).mean(), atol=1e-12)
    assert np.isclose(m["avg_win"], wins.mean(), atol=1e-12)
    assert np.isclose(m["avg_loss"], losses.mean(), atol=1e-12)
    assert np.isclose(m["profit_factor"], wins.sum() / abs(losses.sum()), atol=1e-12)


def test_rolling_sharpe_manual(run):
    r = run["returns_rm"]
    rs = mt.rolling_sharpe(r, 252)
    i = len(r) - 1
    w = r.iloc[i - 251:i + 1]
    assert np.isclose(rs.iloc[i], np.sqrt(252) * w.mean() / w.std(ddof=1), atol=1e-10)


def test_regime_breakdown_manual(run):
    rb = pl.regime_breakdown(run["returns_rm"], REGIME_SEGMENTS)
    seg = REGIME_SEGMENTS[4]                      # COVID & reprise
    sub = run["returns_rm"].loc[seg["start"]:seg["end"]].dropna()
    row = rb[rb["regime"] == seg["name"]].iloc[0]
    assert np.isclose(row["sharpe"], np.sqrt(252) * sub.mean() / sub.std(ddof=1), atol=1e-9)
    assert np.isclose(row["pnl"], sub.sum(), atol=1e-12)


def test_bh_equity_compound(run, prices):
    """Buy & hold : courbe ET métriques composées (CAGR, drawdown en % du pic)."""
    bh = prices["brent"].pct_change().fillna(0.0)
    eq = (1.0 + bh).cumprod()
    assert np.allclose(run["bh_equity"], eq, atol=1e-12)
    m = run["bh_metrics"]
    cagr = (prices["brent"].iloc[-1] / prices["brent"].iloc[0]) ** (252 / len(bh)) - 1.0
    assert np.isclose(m["ann_return"], cagr, rtol=1e-9)
    assert np.isclose(m["max_drawdown"], (eq / eq.cummax() - 1.0).min(), atol=1e-12)
    assert np.isclose(m["total_return"], eq.iloc[-1] - 1.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# 4) Statistiques du couple : recalcul direct                                  #
# --------------------------------------------------------------------------- #
def test_engle_granger_beta_manual(prices):
    y, x = prices["brent"], prices["wti"]
    res = st_.engle_granger(y, x)
    beta_manual = x.cov(y) / x.var(ddof=1)        # OLS simple avec constante
    alpha_manual = y.mean() - beta_manual * x.mean()
    assert np.isclose(res.beta, beta_manual, rtol=1e-9)
    assert np.isclose(res.alpha, alpha_manual, rtol=1e-6)


def test_half_life_manual(prices):
    res = st_.engle_granger(prices["brent"], prices["wti"])
    s = res.resid
    ds, lag = s.diff().dropna(), s.shift(1).dropna()
    df = pd.concat([ds, lag], axis=1, keys=["ds", "lag"]).dropna()
    phi = df["lag"].cov(df["ds"]) / df["lag"].var(ddof=1)
    rho = 1.0 + phi                               # coefficient AR(1)
    # l'écart à la moyenne est multiplié par ρ chaque jour : ρ^h = 1/2
    assert np.isclose(st_.half_life(s), np.log(0.5) / np.log(rho), rtol=1e-9)


# --------------------------------------------------------------------------- #
# 5) Vérités terrain synthétiques                                              #
# --------------------------------------------------------------------------- #
def test_kalman_recovers_true_beta():
    rng = np.random.default_rng(42)
    n = 1500
    x = pd.Series(50 + np.cumsum(rng.normal(0, 1, n)),
                  index=pd.date_range("2015-01-01", periods=n, freq="B"))
    y = 0.5 + 1.3 * x + rng.normal(0, 0.4, n)
    out = hr.kalman_hedge(y, x, delta=1e-4, obs_cov=1e-3)
    assert abs(out["beta"].iloc[-1] - 1.3) < 0.08
    inno = out["resid"].iloc[300:]
    assert abs(inno.mean()) < 0.1 * inno.std()    # innovation ~ centrée


def test_hurst_ground_truth():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2015-01-01", periods=4000, freq="B")
    iid = pd.Series(rng.normal(0, 1, 4000), index=idx)
    walk = pd.Series(np.cumsum(rng.normal(0, 1, 4000)), index=idx)
    assert st_.hurst_exponent(iid) < 0.15         # mean-reverting extrême
    assert 0.4 < st_.hurst_exponent(walk) < 0.6   # marche aléatoire


def test_rolling_vol_manual():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.01, 300),
                  index=pd.date_range("2020-01-01", periods=300, freq="B"))
    v = ft.rolling_vol(r, window=21)
    assert np.isclose(v.iloc[-1], r.iloc[-21:].std(ddof=1) * np.sqrt(252), atol=1e-15)


def test_descriptive_stats_manual():
    rng = np.random.default_rng(2)
    r = pd.DataFrame({"a": rng.normal(0.0005, 0.01, 1000)},
                     index=pd.date_range("2020-01-01", periods=1000, freq="B"))
    stats = ft.descriptive_stats(r)
    assert np.isclose(stats.loc["a", "ann_return"], r["a"].mean() * 252, atol=1e-15)
    assert np.isclose(stats.loc["a", "ann_vol"], r["a"].std(ddof=1) * np.sqrt(252),
                      atol=1e-15)


# --------------------------------------------------------------------------- #
# 6) Courbe WTI : définitions recalculées                                      #
# --------------------------------------------------------------------------- #
def test_curve_definitions():
    curve = dl.load_cache("eia_curve")
    if curve is None:
        pytest.skip("cache courbe absent")
    curve = dl._clean_index(curve)
    cf = cv.curve_features(curve)
    assert np.allclose(cf["slope"], curve["c4"] - curve["c1"], atol=1e-12,
                       equal_nan=True)
    ok = (curve["c2"] > 0)
    assert np.allclose(cf.loc[ok, "roll_yield_m"],
                       (curve.loc[ok, "c1"] - curve.loc[ok, "c2"]) / curve.loc[ok, "c2"],
                       atol=1e-12, equal_nan=True)
    assert np.isclose(cv.regime_share(cf).sum(), 1.0, atol=1e-12)
    # pente inconnue <=> régime inconnu (et non « flat »)
    assert (cf["regime"].isna() == cf["slope"].isna()).all()
    # contango <=> pente > 0 ; flat <=> pente exactement nulle
    sub = cf.dropna(subset=["slope"])
    assert ((sub["regime"] == "contango") == (sub["slope"] > 0)).all()
    assert ((sub["regime"] == "flat") == (sub["slope"] == 0)).all()

def test_forward_returns_tstat_manual(prices):
    """t de Welch sur un échantillon SANS recouvrement (une date tous les 21 j)."""
    curve = dl.load_cache("eia_curve")
    if curve is None:
        pytest.skip("cache courbe absent")
    cf = cv.curve_features(dl._clean_index(curve))
    res = cv.curve_vs_forward_returns(cf, prices["wti"], horizon=21)
    px = prices["wti"].where(prices["wti"] > 0)
    fwd = np.log(px.shift(-21) / px)
    df = pd.concat([cf["regime"], fwd.rename("f")], axis=1, sort=True).dropna().iloc[::21]
    b = df.loc[df["regime"] == "backwardation", "f"]
    c = df.loc[df["regime"] == "contango", "f"]
    t = (b.mean() - c.mean()) / np.sqrt(b.var(ddof=1) / len(b) + c.var(ddof=1) / len(c))
    assert np.isclose(res["t_stat"], t, rtol=1e-9)
