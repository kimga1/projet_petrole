"""Traductions FR -> EN du dashboard.

Le français (langue du code) est la référence : les clés du dictionnaire sont
les textes français exacts tels qu'écrits dans app.py, les valeurs leur
traduction anglaise. Une clé absente est rendue telle quelle (fallback sûr).

Les gabarits gardent leurs champs ``{nom}`` : ils doivent être identiques
dans la clé et la traduction (vérifié par tests/test_i18n.py).
"""
from __future__ import annotations

TRANSLATIONS: dict[str, str] = {
    # --- Sidebar -----------------------------------------------------------
    "Terminal démo · moteur Kalman pairs": "Demo terminal · Kalman pairs engine",
    "Paramètres": "Parameters",
    "Source de prix": "Price source",
    "Futures front-month": "Front-month futures",
    "entrée |z|": "entry |z|",
    "sortie |z|": "exit |z|",
    "coûts / jambe": "costs / leg",
    "j": "d",
    "cible {v} % ann.": "target {v}% ann.",
    "Avancé": "Advanced",
    "Vol cible annualisée": "Annualized target vol",
    "Stop sur z (déconseillé — cf. README)": "z-stop (not recommended — see README)",
    "Seuil stop |z|": "Stop threshold |z|",
    "Kalman δ (réactivité β)": "Kalman δ (β responsiveness)",
    "Fenêtre rolling OLS (j)": "Rolling OLS window (d)",
    "Data : yfinance · FRED · EIA<br>Démo. Pas un conseil d'investissement.":
        "Data: yfinance · FRED · EIA<br>Demo. Not investment advice.",
    "Rouvrir les paramètres": "Reopen parameters",
    # --- Couleur secondaire / disco ---------------------------------------
    "Couleur secondaire": "Secondary color",
    "Couleur": "Color",
    "Ambre": "Amber",
    "Menthe": "Mint",
    "Rose": "Pink",
    "Bleu": "Blue",
    # --- Bandeau, contexte, onglets ---------------------------------------
    "Relative Value · Cointégration · 2007-2026":
        "Relative Value · Cointegration · 2007-2026",
    "séances": "sessions",
    "1 VUE D'ENSEMBLE": "1 OVERVIEW",
    "2 COINTÉGRATION": "2 COINTEGRATION",
    "3 COURBE WTI": "3 WTI CURVE",
    # --- Onglet 1 · Vue d'ensemble ----------------------------------------
    "P&L cumulé · notionnel constant": "Cumulative P&L · constant notional",
    "% / an": "% / yr",
    "notionnel constant": "constant notional",
    "β cointégrant": "Cointegrating β",
    "hedge dynamique": "dynamic hedge",
    "cointégrés ✓": "cointegrated ✓",
    "non cointégrés": "not cointegrated",
    "Demi-vie": "Half-life",
    "horizon de détention": "holding horizon",
    "trendant": "trending",
    "Signal du jour": "Today's signal",
    "Spread cointégrant & bandes ±{v}σ": "Cointegrating spread & ±{v}σ bands",
    "Période": "Period",
    "1A": "1Y",
    "3A": "3Y",
    "5A": "5Y",
    "10A": "10Y",
    "Moyenne mobile": "Rolling mean",
    "Performance par régime": "Performance by regime",
    "Régime": "Regime",
    "Rend. ann.": "Ann. ret.",
    "Jambes du jour": "Today's legs",
    "Prochaine action": "Next action",
    "Pas d'entrée tant que |z| &lt; {zi}. Sortie |z| &lt; {zo} · time-stop {mh} séances.":
        "No entry while |z| &lt; {zi}. Exit when |z| &lt; {zo} · time-stop {mh} sessions.",
    "Historique des prix Brent & WTI": "Brent & WTI price history",
    # --- Régimes (noms de src/config.REGIME_SEGMENTS) ----------------------
    "Pré-glut (2007-2010)": "Pre-glut (2007-2010)",
    "Krach & export ban (2014-2016)": "Crash & export ban (2014-2016)",
    "Normalisation (2017-2019)": "Normalization (2017-2019)",
    "COVID & reprise (2020-2021)": "COVID & recovery (2020-2021)",
    "Guerre & inflation (2022+)": "War & inflation (2022+)",
    # --- Onglet 2 · Cointégration ------------------------------------------
    "ADF résidu": "Residual ADF",
    "stationnaire ✓": "stationary ✓",
    "non stationnaire": "non-stationary",
    "Cointégration glissante...": "Rolling cointegration...",
    "p-value de cointégration glissante · fenêtre 252 j":
        "Rolling cointegration p-value · 252-day window",
    "2014-16 · rupture (shale + export ban)":
        "2014-16 · break (shale + export ban)",
    "β glissant (hedge dynamique)": "Rolling β (dynamic hedge)",
    "β rolling OLS 252 j": "β rolling OLS 252 d",
    "Demi-vie glissante": "Rolling half-life",
    "demi-vie (j)": "half-life (d)",
    "{v} j (plein échantillon)": "{v} d (full sample)",
    "Ruptures (CUSUM sur β) & flag « tradable »":
        "Breaks (CUSUM on β) & “tradeable” flag",
    "export ban levé": "export ban lifted",
    "Vert = cointégré & demi-vie exploitable · rouge = relation suspecte · traits rouges = ruptures CUSUM (diagnostic a posteriori, non causal).":
        "Green = cointegrated & workable half-life · red = suspect relationship · red lines = CUSUM breaks (after-the-fact diagnostic, not causal).",
    # --- Onglet 3 · Courbe WTI ---------------------------------------------
    "Courbe EIA indisponible (ajouter EIA_API_KEY dans .env / secrets Streamlit).":
        "EIA curve unavailable (add EIA_API_KEY to .env / Streamlit secrets).",
    "L'EIA a arrêté la publication des séries RCLC après le 2024-04-05 — cet onglet est borné à cette date.":
        "The EIA stopped publishing the RCLC series after 2024-04-05 — this tab is capped at that date.",
    "Date de la courbe": "Curve date",
    "Courbe au": "Curve as of",
    "Répartition des régimes": "Regime mix",
    "Plat": "Flat",
    "Roll yield annualisé moyen ≈ {v} % sur l'historique complet ; la sous-période 2004-2024 était nettement en contango (≈ −9 %) — biais de fenêtre corrigé par la pagination EIA.":
        "Average annualized roll yield ≈ {v}% over the full history; the 2004-2024 sub-period was firmly in contango (≈ −9%) — a window bias fixed by paginating the EIA API.",
    "Pente C4 − C1": "Slope C4 − C1",
    "pente": "slope",
    "Au-dessus de 0 = contango · en dessous = backwardation.":
        "Above 0 = contango · below = backwardation.",
    "Corr. pente ↔ spread": "Slope ↔ spread corr.",
    "niveaux, descriptif": "levels, descriptive",
    "Spread moyen par régime": "Average spread by regime",
    "Rendement WTI forward 21 j": "WTI forward return · 21 d",
    "non significatif": "not significant",
    "significatif": "significant",
    # --- Onglet 4 · Backtest -----------------------------------------------
    "Rend. annualisé": "Ann. return",
    "Vol annualisée": "Ann. volatility",
    "Équité · P&L cumulé, notionnel constant":
        "Equity · cumulative P&L, constant notional",
    "Brut": "Raw",
    "Drawdown (% du notionnel)": "Drawdown (% of notional)",
    "Pire drawdown : {v} % du notionnel (perte depuis le pic, même unité que la vol et le rendement).":
        "Worst drawdown: {v}% of notional (loss from the peak, same unit as vol and return).",
    "Sharpe glissant · 1 an": "Rolling Sharpe · 1 yr",
    "Sharpe glissant": "Rolling Sharpe",
    "Walk-forward vs split unique · Sharpe brut, mêmes périodes":
        "Walk-forward vs single split · raw Sharpe, same windows",
    "Méthode": "Method",
    "Kalman en ligne (β chaque jour)": "Online Kalman (daily β)",
    "Blocs (β ré-estimé / trimestre)": "Blocks (β re-estimated quarterly)",
    "Split unique (β estimé avant 2018)": "Single split (β estimated pre-2018)",
    "Comparées sur les mêmes fenêtres, seules les méthodes à β figé perdent leur performance hors échantillon ; le Kalman en ligne la conserve.":
        "Compared on the same windows, only the fixed-β methods lose their out-of-sample performance; the online Kalman keeps it.",
    "Sharpe par régime": "Sharpe by regime",
    "Deux modes de casse": "Two failure modes",
    "<b>1. Structurel</b> (2014-16, export ban) : la relation se déplace durablement — problème de signal/régime.":
        "<b>1. Structural</b> (2014-16, export ban): the relationship shifts durably — a signal/regime problem.",
    "<b>2. Prix aberrant</b> (20/04/2020, WTI à −37 $) : le β du Kalman devient négatif — neutralisé par le garde-fou d'entrée (prix &gt; 0, β ∈ [0,5 ; 1,5]).":
        "<b>2. Absurd price</b> (2020-04-20, WTI at −$37): the Kalman β turns negative — neutralized by the entry guard (price &gt; 0, β ∈ [0.5, 1.5]).",
    # --- Onglet 5 · Trades --------------------------------------------------
    "Sensibilité du Sharpe · z d'entrée × lookback":
        "Sharpe sensitivity · entry z × lookback",
    "z d'entrée": "entry z",
    "Surface lisse et positive sur une large plage — le défaut (z 2.0 × 40 j, marqué ●) est un choix principé ≈ 2× la demi-vie, pas le maximum de la grille.":
        "Smooth, positive surface over a wide range — the default (z 2.0 × 40 d, marked ●) is a principled choice ≈ 2× the half-life, not the grid maximum.",
    "Sensibilité aux coûts": "Cost sensitivity",
    "coûts (bps/jambe)": "costs (bps/leg)",
    "Stats de trades": "Trade stats",
    "gain moy.": "avg win",
    "perte moy.": "avg loss",
    "détention moy.": "avg holding",
    "Journal de trades · {n} plus récents sur {total}":
        "Trade log · {n} most recent of {total}",
    "Entrée": "Entry",
    "Sens": "Side",
    "z entrée": "entry z",
    "z sortie": "exit z",
    "Séances": "Sessions",
    "P&L net": "Net P&L",
    "Voir les {n} trades": "Show all {n} trades",
    "Aucun trade sur la période.": "No trades in this period.",
    # --- Barre de statut ----------------------------------------------------
    "Data yfinance · FRED · EIA &nbsp;|&nbsp; trade-on-close · coûts {c} bps/jambe &nbsp;|&nbsp; walk-forward · zéro look-ahead &nbsp;|&nbsp; démo — pas un conseil d'investissement":
        "Data yfinance · FRED · EIA &nbsp;|&nbsp; trade-on-close · costs {c} bps/leg &nbsp;|&nbsp; walk-forward · zero look-ahead &nbsp;|&nbsp; demo — not investment advice",
}


def make_tr(lang: str):
    """Renvoie la fonction de traduction : identité en FR, dictionnaire en EN."""
    if lang == "en":
        return lambda s: TRANSLATIONS.get(s, s)
    return lambda s: s
