"""Moteur de backtest du spread — rigueur = point clé.

Règles (non négociables)
------------------------
- **Zéro info future** : la position en vigueur le jour t provient d'un signal
  décidé en t-1 (execution_lag), et le hedge β est verrouillé à l'entrée avec
  l'information disponible à ce moment-là.
  Convention : signal sur la clôture t-1, exécution à cette même clôture, P&L
  capté de t-1 à t (« trade-on-close »). Pour un remplissage plus prudent à la
  barre suivante, augmenter ``execution_lag`` (les références internes sont
  décalées de ``lag`` de façon cohérente).
- **Coûts paramétrables** : bid-ask (demi-spread) + commission par jambe,
  facturés sur le turnover à chaque changement de position (entrée/sortie).
- **Comptabilité auditable** : moteur écrit en boucle explicite plutôt que
  vectorisé, pour que la causalité et les coûts soient vérifiables ligne à ligne.

Modèle de P&L
-------------
À l'entrée on déploie 1 unité de capital de base sur le portefeuille cointégrant
(1 bbl y, β bbl x, β figé à l'entrée -> pas de rebalancement intra-trade, donc
turnover uniquement entrée/sortie). Le rendement quotidien est le P&L de ce
portefeuille rapporté au capital de base ; le sizing (vol-target) module
l'exposition et donc le rendement. Entre deux trades, rendement = 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import hedge_ratio as hr
from . import signal as sig
from .config import StrategyConfig


# --------------------------------------------------------------------------- #
# Résultat                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class BacktestResult:
    daily: pd.DataFrame      # colonnes : position, beta_used, gross_ret, cost, net_ret, equity, ...
    trades: pd.DataFrame     # journal des trades
    returns: pd.Series       # net_ret (pour Sharpe & co.)
    equity: pd.Series        # courbe d'équité (base 1.0)
    cfg: StrategyConfig
    label: str = "online"


# --------------------------------------------------------------------------- #
# Moteur bas niveau                                                            #
# --------------------------------------------------------------------------- #
def _simulate(
    prices: pd.DataFrame,
    signal_df: pd.DataFrame,
    cfg: StrategyConfig,
    size: pd.Series | None = None,
    label: str = "online",
) -> BacktestResult:
    """Simule le P&L jour par jour à partir des positions cibles."""
    y_col, x_col = cfg.price_col_y, cfg.price_col_x
    df = prices[[y_col, x_col]].join(signal_df[["beta", "resid", "zscore", "position"]], how="inner")
    df = df.dropna(subset=["position"])
    idx = df.index
    n = len(df)

    py = df[y_col].to_numpy(float)          # prix y (Brent)
    px = df[x_col].to_numpy(float)          # prix x (WTI)
    beta = df["beta"].to_numpy(float)
    zsc = df["zscore"].to_numpy(float)
    target = df["position"].to_numpy(float)

    lag = max(1, cfg.execution_lag)
    # position en vigueur le jour t = cible décidée en t-lag
    held = np.concatenate([np.zeros(lag), target[:-lag]]) if n > lag else np.zeros(n)
    beta_sig = np.concatenate([np.full(lag, np.nan), beta[:-lag]]) if n > lag else np.full(n, np.nan)
    z_sig = np.concatenate([np.full(lag, np.nan), zsc[:-lag]]) if n > lag else np.full(n, np.nan)

    if size is None:
        size_sig = np.ones(n)
    else:
        s_al = size.reindex(idx).to_numpy(float)
        size_sig = np.concatenate([np.ones(lag), s_al[:-lag]]) if n > lag else np.ones(n)
        size_sig = np.nan_to_num(size_sig, nan=1.0)

    # warmup : pas de trade avant cfg.warmup barres. Une position ouverte PENDANT
    # l'amorçage n'est pas héritée (elle serait entrée sans franchissement de
    # z_in observé) : on attend son retour à flat, puis un vrai signal.
    warm = min(cfg.warmup, n)
    if 0 < warm < n and held[warm] != 0.0 and held[warm - 1] == held[warm]:
        carried, i = held[warm], warm
        while i < n and held[i] == carried:
            held[i] = 0.0
            i += 1
    held[:warm] = 0.0

    cost_leg = cfg.cost_bps_per_leg / 1e4

    brent_u = 0.0          # bbl y détenus
    wti_u = 0.0            # bbl x détenus
    prev_bu = 0.0
    prev_wu = 0.0

    gross_ret = np.zeros(n)
    cost = np.zeros(n)
    net_ret = np.zeros(n)
    notional = np.zeros(n)     # notionnel brut détenu le jour t, valorisé à la clôture t-1
    beta_used = np.full(n, np.nan)

    trades = []
    entry = None           # dict de l'entrée en cours

    for t in range(1, n):
        s = held[t]

        # (Re)définition des unités quand la position change (entrée / sortie).
        if held[t] != held[t - 1]:
            # Garde : un retournement direct (±1 -> ∓1 sans jour flat) ne doit pas
            # écraser le trade en cours. La machine à états passe normalement par
            # flat, mais on clôture explicitement par sécurité.
            if s != 0.0 and entry is not None:
                entry["exit_date"] = idx[t]
                entry["exit_z"] = float(z_sig[t]) if np.isfinite(z_sig[t]) else np.nan
                entry["holding_days"] = int((idx[t] - entry["entry_date"]).days)
                entry["holding_bars"] = int(t - entry.pop("_t0"))
                entry["net"] = entry["gross"] - entry["cost"]
                trades.append(entry)
                entry = None
            if s != 0.0:
                # ENTRÉE : verrouille β et taille aux conditions du signal (t-lag).
                b_lock = beta_sig[t]
                m_lock = size_sig[t]
                if not np.isfinite(b_lock):
                    b_lock = beta[t - 1]
                m_lock = float(np.clip(m_lock, 0.0, cfg.max_leverage))
                # Prix d'exécution = clôture t-1 (bord où la position s'active) ;
                # notionnel en valeur absolue (robuste au WTI négatif).
                gross_entry = abs(py[t - 1]) + abs(b_lock) * abs(px[t - 1])
                scale = (m_lock / gross_entry) if gross_entry > 1e-8 else 0.0
                brent_u = np.sign(s) * scale
                wti_u = -np.sign(s) * b_lock * scale
                entry = {
                    "entry_date": idx[t], "side": int(np.sign(s)),
                    "beta": float(b_lock), "size": float(m_lock),
                    "entry_z": float(z_sig[t]) if np.isfinite(z_sig[t]) else np.nan,
                    "gross": 0.0, "cost": 0.0, "_t0": t,
                }
            else:
                brent_u = 0.0
                wti_u = 0.0

        beta_used[t] = -wti_u / brent_u if brent_u != 0 else np.nan
        notional[t] = abs(brent_u) * abs(py[t - 1]) + abs(wti_u) * abs(px[t - 1])

        # P&L du jour (variation de prix sur les unités détenues).
        pnl = brent_u * (py[t] - py[t - 1]) + wti_u * (px[t] - px[t - 1])
        # Coût du turnover facturé au prix d'EXÉCUTION (clôture t-1), cohérent
        # avec l'assiette du notionnel ; en valeur absolue.
        turn = abs(brent_u - prev_bu) * abs(py[t - 1]) + abs(wti_u - prev_wu) * abs(px[t - 1])
        c = cost_leg * turn

        gross_ret[t] = pnl
        cost[t] = c
        net_ret[t] = pnl - c

        if entry is not None:
            entry["gross"] += pnl
            entry["cost"] += c

        # Clôture de trade (retour à flat). exit_z = z au moment de la DÉCISION
        # de sortie (t-lag), symétrique de entry_z.
        if s == 0.0 and held[t - 1] != 0.0 and entry is not None:
            entry["exit_date"] = idx[t]
            entry["exit_z"] = float(z_sig[t]) if np.isfinite(z_sig[t]) else np.nan
            entry["holding_days"] = int((idx[t] - entry["entry_date"]).days)
            entry["holding_bars"] = int(t - entry.pop("_t0"))
            entry["net"] = entry["gross"] - entry["cost"]
            trades.append(entry)
            entry = None

        prev_bu, prev_wu = brent_u, wti_u

    # Trade encore ouvert en fin d'échantillon.
    if entry is not None:
        entry["exit_date"] = idx[-1]
        entry["exit_z"] = float(zsc[-1]) if np.isfinite(zsc[-1]) else np.nan
        entry["holding_days"] = int((idx[-1] - entry["entry_date"]).days)
        entry["holding_bars"] = int(n - entry.pop("_t0"))
        entry["net"] = entry["gross"] - entry["cost"]
        entry["open_at_end"] = True
        trades.append(entry)

    daily = pd.DataFrame({
        "y": py, "x": px, "beta": beta, "zscore": zsc,
        "position": held, "beta_used": beta_used, "notional": notional,
        "gross_ret": gross_ret, "cost": cost, "net_ret": net_ret,
    }, index=idx)
    # P&L ADDITIF sur notionnel constant (convention correcte pour un spread
    # market-neutral ; robuste aux rendements < -100 % du jour WTI négatif).
    daily["equity"] = 1.0 + daily["net_ret"].cumsum()
    daily["gross_equity"] = 1.0 + daily["gross_ret"].cumsum()

    trades_df = pd.DataFrame(trades)
    ret = daily["net_ret"].copy()
    ret.name = "net_ret"
    return BacktestResult(daily=daily, trades=trades_df, returns=ret,
                          equity=daily["equity"], cfg=cfg, label=label)


def trades_with_pnl(trades: pd.DataFrame, pnl: pd.DataFrame,
                    size: pd.Series | None = None) -> pd.DataFrame:
    """Journal de trades re-valorisé avec un autre P&L quotidien (ex. vol-target).

    ``pnl`` a les colonnes gross / cost / net ; chaque trade les somme sur
    [entry_date, exit_date], mêmes bornes que le moteur (jour d'entrée et jour
    de sortie inclus). ``size`` : multiplicateur en vigueur le jour d'entrée.
    """
    if trades is None or trades.empty:
        return trades
    out = trades.copy()
    spans = [(pd.Timestamp(e), pd.Timestamp(x))
             for e, x in zip(out["entry_date"], out["exit_date"])]
    for col in ("gross", "cost", "net"):
        out[col] = [float(pnl[col].loc[e:x].sum()) for e, x in spans]
    if size is not None:
        out["size"] = [float(size.loc[e]) for e, _ in spans]
    return out


# --------------------------------------------------------------------------- #
# Driver principal : backtest « online » (walk-forward continu)               #
# --------------------------------------------------------------------------- #
def run_backtest(
    prices: pd.DataFrame,
    cfg: StrategyConfig | None = None,
    size: pd.Series | None = None,
) -> BacktestResult:
    """Backtest continu : hedge glissant/Kalman + z-score glissant (causaux).

    C'est la version la plus rigoureuse : chaque paramètre à t n'utilise que
    l'information ≤ t (ré-estimation glissante à la journée = walk-forward continu).
    """
    cfg = cfg or StrategyConfig()
    hedge_df = hr.compute_hedge(prices, cfg)
    signal_df = sig.build_signal(prices, hedge_df, cfg)
    return _simulate(prices, signal_df, cfg, size=size, label="online")


# --------------------------------------------------------------------------- #
# Comparaison : split simple in-sample / OOS                                   #
# --------------------------------------------------------------------------- #
def backtest_simple_split(
    prices: pd.DataFrame,
    cfg: StrategyConfig | None = None,
    split: str | pd.Timestamp = "2018-01-01",
) -> dict:
    """β + constante estimés UNE fois en in-sample, appliqués en OOS (fixe).

    Sert de référence pour montrer l'intérêt de la ré-estimation glissante.
    """
    cfg = cfg or StrategyConfig()
    split = pd.Timestamp(split)
    y = prices[cfg.price_col_y]
    x = prices[cfg.price_col_x]

    train = prices.loc[prices.index < split]
    ols = hr.static_ols(train[cfg.price_col_y], train[cfg.price_col_x])
    alpha = float(ols["alpha"].iloc[0])
    beta = float(ols["beta"].iloc[0])

    resid = (y - alpha - beta * x).rename("resid")
    hedge_df = pd.DataFrame({"alpha": alpha, "beta": beta, "resid": resid}, index=prices.index)
    signal_df = sig.build_signal(prices, hedge_df, cfg)

    res = _simulate(prices, signal_df, cfg, label="simple_split")
    return {
        "result": res,
        "beta": beta, "alpha": alpha, "split": split,
        "is_returns": res.returns.loc[res.returns.index < split],
        "oos_returns": res.returns.loc[res.returns.index >= split],
    }


# --------------------------------------------------------------------------- #
# Walk-forward par blocs (ré-estimation périodique)                           #
# --------------------------------------------------------------------------- #
def backtest_walk_forward_blocks(
    prices: pd.DataFrame,
    cfg: StrategyConfig | None = None,
    train_days: int = 504,     # ~2 ans d'entraînement
    test_days: int = 63,       # ~1 trimestre OOS par bloc
) -> BacktestResult:
    """β ré-estimé (OLS statique) sur chaque fenêtre train, appliqué au bloc
    test suivant ; positions OOS concaténées puis simulées.

    Différence avec 'online' : β est constant par bloc (pas à la journée), ce
    qui illustre le compromis fréquence de ré-estimation / réactivité.
    """
    cfg = cfg or StrategyConfig()
    y = prices[cfg.price_col_y]
    x = prices[cfg.price_col_x]
    n = len(prices)

    beta_series = pd.Series(np.nan, index=prices.index, name="beta")
    resid_series = pd.Series(np.nan, index=prices.index, name="resid")

    start = train_days
    while start < n:
        tr = slice(start - train_days, start)
        te = slice(start, min(start + test_days, n))
        y_tr, x_tr = y.iloc[tr], x.iloc[tr]
        ols = hr.static_ols(y_tr, x_tr)
        a = float(ols["alpha"].iloc[0])
        b = float(ols["beta"].iloc[0])
        te_idx = prices.index[te]
        beta_series.loc[te_idx] = b
        resid_series.loc[te_idx] = (y.loc[te_idx] - a - b * x.loc[te_idx])
        start += test_days

    hedge_df = pd.DataFrame({"beta": beta_series, "resid": resid_series})
    signal_df = sig.build_signal(prices, hedge_df, cfg)
    return _simulate(prices, signal_df, cfg, label="walk_forward_blocks")
