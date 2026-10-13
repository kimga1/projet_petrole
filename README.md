# Brent–WTI · Statistical Arbitrage & Term Structure

Valeur relative sur le crude : stratégie de **mean-reversion sur le spread Brent–WTI**,
fondée statistiquement (**cointégration**), backtestée proprement (**coûts, walk-forward,
zéro look-ahead**), enrichie de l'analyse de **structure de courbe WTI** (contango/backwardation)
et d'un **dashboard interactif** Streamlit (bilingue : anglais par défaut, bouton EN | FR ;
couleur secondaire au choix et **mode disco** 🪩 — parce qu'un terminal peut aussi être fun).

> Projet de démonstration quant (valeur relative / stat-arb). **Aucune recommandation
> d'investissement.**

---

## 1. Résultats clés

Backtest sur **futures front-month** Brent/WTI (2007-07-30 → 2026-09-16, 4 761 séances,
dont 1 an d'amorçage sans position), hedge **Kalman** (β dynamique), ré-estimation
glissante continue, exécution *trade-on-close*, coûts 2,5 bps/jambe, position investie
~62 % du temps.

| Métrique | Brut | **Risk-managed** (vol-target 10 %) | Buy & Hold Brent |
|---|---:|---:|---:|
| Sharpe annualisé | +0,45 | **+0,45** | +0,24 |
| Rendement annualisé | +4,1 % | **+4,5 %** | +1,8 % (CAGR) |
| Volatilité annualisée | 9,1 % | 9,9 % | 38,4 % |
| Max drawdown | −20,3 % du notionnel | **−20,1 % du notionnel** | −86,8 % |
| Sortino | 0,64 | 0,66 | 0,33 |
| # trades / hit rate | 73 / 59 % | 73 / 55 % | — |
| Profit factor | 2,74 | 2,15 | — |
| Durée moyenne d'un trade | 40 séances (59 j calendaires) | idem | — |

**Statistiques du spread** : β cointégrant = 1,057 · p-value de cointégration = **0,013**
(cointégrés sur l'ensemble, mais **pas sur 2007-2013** : p = 0,56) · ADF du résidu p = 0,003 ·
**demi-vie = 18,9 j** · Hurst = 0,29 (< 0,5 → mean-reverting).

**Le message central** (et honnête) : le signal de mean-reversion a un **Sharpe ≈ 0,45 net
de coûts, identique avant et après 2018** (0,45 / 0,45). Deux corrections ont changé la
lecture du projet :

1. **Un seul trade écrasait tout le reste.** Le 20/04/2020 (WTI à −37 $), le filtre de
   Kalman produit un β négatif ; le trade ouvert le lendemain avec β ≈ 0 n'était pas un trade
   de spread mais une vente sèche de Brent (−0,58 de P&L, soit plus que tout le gain de
   l'échantillon). Un **garde-fou d'entrée** (prix > 0 et β ∈ [0,5 ; 1,5]) l'élimine :
   Sharpe brut +0,08 → **+0,45**. Ce garde-fou est justifié économiquement (un spread à
   β < 0 n'est pas un spread), mais il a été motivé par un cas observé : l'ampleur de
   l'effet montre à quel point un résultat de backtest peut tenir à un seul événement.
2. **Le vol-targeting ne crée pas de performance.** Mesuré correctement (vol du spread,
   pas des rendements nuls hors position), il fixe le niveau de risque (vol **9,9 %** pour
   10 % visés) sans changer le Sharpe (0,45 → 0,45). L'ancien « sauvetage de 2020 »
   (−0,56 → +0,46) était un artefact : un seul jour à levier 2,9×.

La zone de casse qui reste est la **rupture structurelle 2014-2016** (shale + levée de
l'export ban US) : Sharpe −0,40.

---

## 2. Structure du repo

```
projet_petrole/
├── src/                      # bibliothèque (11 modules, testés)
│   ├── config.py             # chemins, tickers, StrategyConfig, clé EIA
│   ├── data_loader.py        # yfinance + FRED + EIA, alignement, cache parquet
│   ├── features.py           # rendements, vol, spreads, stats descriptives
│   ├── stats_tests.py        # ADF, Engle-Granger, demi-vie (OU/AR1), Hurst
│   ├── hedge_ratio.py        # OLS statique, rolling OLS, filtre de Kalman (maison)
│   ├── signal.py             # z-score glissant, machine à états, garde-fou d'entrée
│   ├── backtest.py           # moteur P&L, walk-forward, split simple (sans look-ahead)
│   ├── metrics.py            # Sharpe, Sortino, drawdown, hit rate, rolling Sharpe...
│   ├── curve.py              # pente, contango/backwardation, roll yield
│   ├── regime.py             # cointégration glissante, ruptures (CUSUM), flag tradable
│   ├── sizing.py             # vol-targeting (position ∝ 1/vol du spread)
│   └── pipeline.py           # orchestration bout-en-bout (source unique de vérité)
├── scripts/
│   ├── fetch_data.py         # télécharge & cache toutes les données
│   └── run_backtest.py       # fige tous les résultats dans results/
├── dashboard/                # dashboard Streamlit (5 onglets)
│   ├── app.py                # orchestration : état, sidebar, onglets
│   ├── ui.py                 # jetons de design, template Plotly, gabarits HTML
│   ├── style.css             # feuille de style du thème terminal
│   ├── i18n.py               # traductions FR -> EN (bouton EN | FR)
│   └── theme.py              # couleur secondaire : préréglages + disco
├── notebooks/                # exploration & résultats commentés (01→04)
├── tests/                    # 54 tests (no-look-ahead + audit mathématique + i18n/thème)
├── results/                  # artefacts reproductibles (métriques, courbes, sensibilités)
├── data/                     # cache parquet (versionné)
├── report/report.md          # note de synthèse 2 pages
├── requirements.txt          # + requirements-dev.txt (pytest)
└── .streamlit/               # thème + template secrets
```

---

## 3. Installation & lancement

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate    |  Linux/Mac : source .venv/bin/activate
pip install -r requirements.txt

# (optionnel) clé EIA pour l'onglet courbe : copier .env.example -> .env et renseigner
python scripts/fetch_data.py       # télécharge & cache Brent/WTI (+ courbe si clé EIA)
python scripts/run_backtest.py     # génère tous les résultats dans results/
pytest -q                          # 54 tests

streamlit run dashboard/app.py     # dashboard local -> http://localhost:8501
```

Interface en **anglais par défaut** ; bouton **EN | FR** en haut de page pour basculer
(`?lang=fr` dans l'URL ouvre directement en français). Le bouton **🎨** à côté change la
couleur secondaire (6 préréglages) et active le **mode disco** (la teinte tourne en continu, graphiques et
tableaux compris ; seul le vert/rouge des P&L reste stable). Liens partageables : `?accent=rose`,
`?disco=1`.

Le cœur (spread, cointégration, backtest) tourne **sans aucune clé** via yfinance + FRED.
Seul l'onglet *Courbe WTI* nécessite une clé EIA (gratuite).

---

## 4. Déploiement Streamlit Community Cloud (URL publique)

1. Pousser le repo sur GitHub.
2. Sur [share.streamlit.io](https://share.streamlit.io) → *New app* → fichier principal
   `dashboard/app.py`.
3. *App settings → Secrets* : coller `EIA_API_KEY = "..."` (cf. `.streamlit/secrets.toml.example`).
4. Les données parquet cachées étant versionnées, le dashboard démarre immédiatement
   (pas de dépendance réseau au chargement).

---

## 5. Méthodologie — le cœur (et le « pourquoi » de chaque choix)

Chaque décision est justifiée en une phrase — c'est ce qui se défend en entretien.

| Choix | Pourquoi |
|---|---|
| **Futures front-month (CL=F, BZ=F)** comme base | C'est **tradable** et *live* : le spread front-month est ce qu'un desk exécute réellement. Le spot FRED sert de robustesse / historique long, pas de support d'exécution (sur le spot FRED, même stratégie : Sharpe 0,32 risk-managed). |
| **Cointégration AVANT de trader** | On ne trade un spread que si une **combinaison linéaire des prix est stationnaire** (Engle-Granger, valeurs critiques de MacKinnon). Ici p = 0,013 sur l'ensemble. Deux nuances assumées : (1) l'ADF sur les niveaux est **à la limite** (Brent p = 0,046, WTI p = 0,06 ; avec tendance 0,16 / 0,21 → racine unitaire non rejetée) ; (2) la relation **n'est pas stable** : p = 0,56 sur 2007-2013, 0,003 sur 2014-2019, 0,001 sur 2020-2026. |
| **Log-returns** pour les stats | Additifs dans le temps, quasi symétriques. |
| **Spread = niveau `y − β·x`** (pas l'innovation du Kalman) | L'innovation du filtre est ≈ un bruit blanc (c'est le signe qu'il fonctionne) → non tradable. Le **niveau** du spread est persistant ; le z-score glissant retranche sa moyenne mobile, ce qui absorbe la constante α et la dérive lente de régime. |
| **Demi-vie → horizon & lookback** | Demi-vie **18,9 j**, formule exacte −ln2/ln(1+φ) de la régression Δspread ~ spread₋₁ (l'approximation −ln2/φ donnait 19,2 j). Elle est mesurée sur le résidu cointégrant **statique** et sert d'ordre de grandeur : le spread réellement tradé (β dynamique, sans α) dérive en niveau (demi-vie brute ≈ 340 j), d'où le centrage par moyenne glissante. **Lookback ≈ 2× demi-vie ≈ 40 j** : choix *principé*, pas maximisant (la grille monte à 0,59 pour 30 j ; on ne l'a pas retenu). |
| **Entrée \|z\|>2, sortie \|z\|<0,5 ; pas de stop sur z ; sortie temporelle 90 séances** | Un *stop au niveau z* entre en conflit avec l'entrée (whipsaw) et coupe les grosses réversions : Sharpe 0,45 → 0,40 avec un stop à \|z\| > 3. Le time-stop à 90 séances (≈ 4-5× la demi-vie) coupe les positions qui ne mean-revertent plus. ⚠️ C'est aussi le meilleur de sa grille (0,36-0,45 pour 60, 120 ou sans) : choisi en connaissant l'échantillon. |
| **Hedge ratio dynamique (Kalman)** | La relation Brent = α + β·WTI **dérive** avec les régimes (logistique Cushing, export ban, qualité). Un β figé sur tout l'échantillon utilise de l'information future. Le Kalman modélise β en marche aléatoire, mis à jour bayésiennement, causal, sans fenêtre arbitraire. Implémenté **à la main** (predict/update). ⚠️ Le bruit d'observation par défaut (1e-3 $²) est très petit devant la variance du résidu (~25 $²) : β est très réactif, et le résultat dépend de ce paramètre (Sharpe 0,45 à 0,83 selon sa valeur, cf. `results/sensitivity_kalman.csv`). Le défaut est la valeur la **plus basse** de la grille : on ne l'a pas optimisé. |
| **Garde-fou d'entrée** | Pas d'entrée si un prix est ≤ 0 ou si β sort de [0,5 ; 1,5] (Brent et WTI cotent à des niveaux proches → β ≈ 1). Sans lui, le WTI négatif d'avril 2020 donne β < 0 et la « couverture » devient une position sèche sur le Brent. |
| **Walk-forward (ré-estimation glissante)** | Kalman + z-score glissant = ré-estimation à la journée n'utilisant que le passé = **walk-forward continu**. Comparé **sur les mêmes fenêtres** (< 2018 / ≥ 2018) : Kalman en ligne **0,45 / 0,45** ; β ré-estimé par blocs trimestriels 0,69 / 0,14 ; β figé (split unique) 0,63 / 0,11. Ce qui décroche après 2018, ce sont les méthodes à **β constant**, pas le Kalman. |
| **Exécution *trade-on-close*, coûts sur le turnover** | Le signal est calculé sur la clôture t−1 et exécuté à **cette même clôture** (réaliste avec des ordres au settlement, TAS) ; β et taille verrouillés à l'entrée. Plus prudent, l'exécution à la **clôture suivante** donne Sharpe 0,42 brut / 0,36 risk-managed (`results/sensitivity_execution.csv`). Coûts (bid-ask + commission) facturés à chaque changement de position. |
| **Vol-targeting (position ∝ 1/vol)** | Vol mesurée sur le **spread supposé toujours investi**, jusqu'à la veille (causal). Mesurée sur les rendements de la stratégie (nuls quand on est flat), elle tombait à ~0 avant chaque entrée → levier maximal (3×) dans 52 % des trades et vol réalisée de 17,9 % au lieu de 10 %. Taille ajustée chaque jour, **rééquilibrage facturé**. Effet : vol 9,9 %, Sharpe inchangé (0,45). |
| **P&L additif, notionnel constant** | Convention correcte pour un spread market-neutral (on ne compose pas le rendement d'un spread), et robuste au **WTI négatif d'avril 2020**. Le **drawdown est donc exprimé en fraction du notionnel** (même unité que la vol et le rendement), pas en % d'un pic d'équité gonflé par les gains accumulés. Le buy & hold, lui, est composé : CAGR et drawdown en % du pic. |

---

## 6. Où et pourquoi la stratégie casse (à documenter, pas à cacher)

Ventilation du Sharpe par régime :

| Régime | Risk-managed | Brut | Lecture |
|---|---:|---:|---|
| Pré-glut 2007-2010 | +1,40 | +0,86 | relation stable (1er trade en déc. 2008, après l'amorçage) — alors même que la cointégration n'est pas significative sur 2007-2013 |
| WTI discount / Cushing glut 2011-2014 | +0,54 | +0,64 | mean-reversion autour du discount grandissant, captée par le β dynamique |
| **Krach & export ban 2014-2016** | **−0,40** | **−0,19** | **la vraie rupture structurelle** : krach shale puis **levée de l'export ban US (déc. 2015)** re-normalisent durablement le spread → mean-reversion contre-productive |
| Normalisation 2017-2019 | +0,29 | +0,33 | régime calme, edge modeste |
| COVID & reprise 2020-2021 | +0,39 | +0,65 | WTI négatif neutralisé par le garde-fou (P&L 2020 ≈ 0 : +0,02 brut, −0,02 risk-managed) |
| Guerre & inflation 2022+ | +0,46 | +0,43 | backwardation soutenue, spread plus mobile |

Deux modes de casse identifiés : (1) **rupture structurelle de la relation** (2014-2016) —
un problème de *signal*/régime, suivi par la cointégration glissante et le flag « tradable » ;
(2) **prix aberrant** (20/04/2020) qui rend le hedge ratio absurde — traité par le garde-fou
d'entrée, pas par le vol-targeting.

---

## 7. Structure de courbe WTI (EIA RCLC1..4, ≤ 2024-04-05)

- **52,0 % backwardation / 47,8 % contango / 0,3 % plat** sur **1983-2024** (historique EIA
  complet ; 445 jours sans contrat C4, surtout avant 1985, sont exclus au lieu d'être comptés
  « plats »). Roll yield annualisé moyen **≈ +0,4 %**. NB : la sous-période **2004-2024 était
  nettement en contango** (roll yield ≈ −8,6 %, carry négatif) — l'écart illustre pourquoi il
  fallait **paginer l'API EIA** (sinon on ne récupérait que les 5000 dernières lignes, soit
  2004-2024, avec une conclusion biaisée).
- **Lien courbe ↔ spread** (dates communes seulement : **2007-2024**) : le spread Brent-WTI
  est plus large en **contango** (6,18 $) qu'en **backwardation** (4,28 $) — cohérent avec
  l'idée qu'un excès d'offre (contango) pèse sur le WTI/Cushing. La corrélation pente/spread
  (+0,14) porte sur les niveaux de deux séries très persistantes : elle est **descriptive**,
  pas une preuve de lien.
- **Courbe ↔ rendements forward** : +0,22 % à 21 j après backwardation contre −0,06 % après
  contango, mais la différence **n'est pas significative** (t = −0,4 sur un échantillon sans
  recouvrement, une date tous les 21 jours ; le signe s'inverse même).

---

## 8. Limites (assumées)

- **Cutoff EIA 2024-04-05** : l'EIA a arrêté la publication des séries RCLC — l'onglet courbe
  est borné à cette date (labellisé partout).
- **Pas de courbe Brent gratuite** : l'analyse de term structure est faite sur le WTI (données
  EIA gratuites) ; une courbe Brent nécessiterait une source payante (ICE).
- **Séries front-month continues non ajustées** : au changement de contrat, CL=F et BZ=F
  sautent de l'écart entre échéances ; ce saut entre dans le P&L alors qu'une position réelle
  roulée ne le capte pas (et les deux contrats ne roulent pas à la même date). Même remarque
  pour le buy & hold. Une série ajustée des rolls demanderait des données par échéance.
- **Spot vs futures** : sur la période récente, le spread *spot FRED* diverge du spread
  *futures* (qualité/timing de publication FRED). On privilégie les **futures** comme base
  tradable ; FRED sert la robustesse et l'historique long.
- **yfinance** : données de clôture non institutionnelles (suffisantes pour la démonstration ;
  un desk utiliserait un flux propre).
- **Choix faits en connaissant l'échantillon** : retrait du stop sur z, time-stop à 90 séances,
  garde-fou d'entrée. Tout est causal (aucune donnée future dans les calculs), mais ces choix
  de conception ne sont pas un vrai test hors échantillon.

---

## 9. Reproductibilité

`python scripts/run_backtest.py` régénère l'intégralité de `results/` (statistiques, métriques,
courbes d'équité, journaux de trades brut et risk-managed, ventilation par régime, grilles de
sensibilité seuils / coûts / exécution / Kalman, comparaison walk-forward/split à période
égale, structure de courbe). `pytest -q` valide 54 tests, dont la **propriété d'absence de
look-ahead de bout en bout** (le pipeline complet calculé sur un préfixe donne exactement les
mêmes rendements que sur la série complète) et un **audit mathématique** qui recalcule par une
voie indépendante le P&L trade par trade, le vol-targeting, les métriques et les statistiques.
