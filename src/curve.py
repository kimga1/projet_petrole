"""Structure de courbe WTI : pente, contango/backwardation, roll yield.

Données : contrats WTI 1..4 (EIA RCLC1..4, $/bbl), <= 2024-04-05.

Définitions (à défendre)
------------------------
- **Pente** = C4 - C1. > 0 : contrats lointains plus chers -> **contango**.
  < 0 : contrats proches plus chers -> **backwardation**.
- **Roll yield** ≈ (C1 - C2)/C2 : rendement de portage d'un long roulé du
  contrat proche vers le suivant. > 0 en backwardation (on roule vers moins
  cher), < 0 en contango. Contrats ~mensuels -> annualisation ×12.
- Lien testé : régime de courbe (contango/backwardation) ↔ spread Brent-WTI et
  rendements forward du WTI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def curve_features(curve: pd.DataFrame, flat_band: float = 0.0) -> pd.DataFrame:
    """Calcule pente, pente %, roll yield et le régime à partir de c1..c4."""
    required = ["c1", "c2", "c3", "c4"]
    missing = [col for col in required if col not in curve.columns]
    if missing:
        raise KeyError(f"colonnes de courbe manquantes: {missing}")
    c = curve.copy()
    out = pd.DataFrame(index=c.index)
    out["c1"] = c["c1"]
    out["c4"] = c["c4"]
    out["slope"] = c["c4"] - c["c1"]
    # Divisions protégées : dénominateur <= 0 (ex. WTI négatif) -> NaN (évite
    # une inversion de signe silencieuse du roll yield / de la pente %).
    c1_pos = c["c1"].where(c["c1"] > 0)
    c2_pos = c["c2"].where(c["c2"] > 0)
    out["slope_pct"] = (c["c4"] - c["c1"]) / c1_pos
    out["roll_yield_m"] = (c["c1"] - c["c2"]) / c2_pos       # mensuel
    out["roll_yield_ann"] = out["roll_yield_m"] * 12.0        # annualisé (approx. linéaire)
    regime = np.where(
        out["slope"] > flat_band, "contango",
        np.where(out["slope"] < -flat_band, "backwardation", "flat"),
    )
    # Pente inconnue (ex. C4 absent avant 1985) -> régime inconnu (NaN), et non
    # « flat » : sinon ces jours gonflent artificiellement la part « plat ».
    out["regime"] = pd.Series(regime, index=out.index).where(out["slope"].notna())
    return out


def regime_timeline(curve_feat: pd.DataFrame) -> pd.Series:
    """Série catégorielle du régime de courbe (pour la timeline du dashboard)."""
    return curve_feat["regime"].astype("category")


def regime_share(curve_feat: pd.DataFrame) -> pd.Series:
    """Répartition du temps passé dans chaque régime (jours à pente connue)."""
    return curve_feat["regime"].dropna().value_counts(normalize=True)


def link_curve_to_spread(curve_feat: pd.DataFrame, spread: pd.Series) -> dict:
    """Lien régime/pente de courbe ↔ niveau du spread Brent-WTI (descriptif).

    Calculé sur les dates communes aux deux séries (``start``/``end``), pas sur
    tout l'historique de la courbe. NB : la corrélation porte sur les NIVEAUX de
    deux séries très persistantes -> descriptive seulement, pas une preuve de lien.
    """
    df = pd.concat([curve_feat[["slope", "regime"]], spread.rename("spread")],
                   axis=1, sort=True).dropna()
    if df.empty:
        return {"corr_slope_spread": np.nan, "spread_by_regime": pd.Series(dtype=float)}
    return {
        "corr_slope_spread": float(df["slope"].corr(df["spread"])),
        "spread_by_regime": df.groupby("regime")["spread"].mean(),
        "n_obs": int(len(df)),
        "start": df.index.min(),
        "end": df.index.max(),
    }


def curve_vs_forward_returns(
    curve_feat: pd.DataFrame, wti_price: pd.Series, horizon: int = 21
) -> dict:
    """Rendements forward du WTI (à `horizon` jours) conditionnés au régime.

    Teste l'intuition : la backwardation (offre tendue) précède-t-elle des
    rendements WTI différents du contango ?

    Significativité : des rendements à 21 j calculés chaque jour se recouvrent
    (20 jours sur 21 en commun) -> on ne garde qu'une date tous les ``horizon``
    jours pour le t de Welch (backwardation − contango), sans quoi l'erreur
    standard serait fortement sous-estimée.
    """
    px = wti_price.where(wti_price > 0)                  # log indéfini si prix <= 0
    fwd = np.log(px.shift(-horizon) / px).rename("fwd_ret")
    df = pd.concat([curve_feat["regime"], fwd], axis=1, sort=True).dropna()
    if df.empty:
        return {"fwd_ret_by_regime": pd.Series(dtype=float), "horizon": horizon}
    grp = df.groupby("regime")["fwd_ret"]

    sub = df.iloc[::horizon].groupby("regime")["fwd_ret"]   # sans recouvrement
    m, sd, n = sub.mean(), sub.std(ddof=1), sub.count()
    diff = t_stat = np.nan
    if {"backwardation", "contango"} <= set(n.index) and min(n["backwardation"], n["contango"]) > 1:
        diff = float(m["backwardation"] - m["contango"])
        se = np.sqrt(sd["backwardation"] ** 2 / n["backwardation"]
                     + sd["contango"] ** 2 / n["contango"])
        t_stat = float(diff / se) if se > 0 else np.nan
    return {
        "fwd_ret_by_regime": grp.mean(),
        "fwd_ret_std_by_regime": grp.std(),
        "n_by_regime": grp.count(),
        "horizon": horizon,
        "diff_back_minus_contango": diff,     # sur l'échantillon sans recouvrement
        "t_stat": t_stat,                     # |t| < 2 -> pas de différence significative
        "n_nonoverlap": n,
    }
