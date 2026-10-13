# Note de recherche — Mean-reversion sur le spread Brent–WTI

*Valeur relative sur le crude · cointégration, walk-forward, contrôle de risque, structure de courbe*
*Données : futures front-month CL=F/BZ=F (2007-2026) · spot FRED · courbe WTI EIA. Note de démonstration, pas de recommandation d'investissement.*

---

## 1. Thèse et contexte

Le Brent et le WTI sont deux références du brut liées par l'arbitrage physique (qualité,
transport maritime, logistique de Cushing). Leur écart oscille autour d'un niveau d'équilibre
mais **cet équilibre se déplace par régimes** : discount WTI durant la saturation de Cushing
(2011-2014), re-normalisation après la levée de l'export ban US (déc. 2015), choc du WTI
négatif (avril 2020). La thèse testée : **le spread est cointégré et mean-reverting à l'échelle
de quelques semaines, exploitable tant que la relation tient et que le risque est maîtrisé.**

## 2. Données et construction

On travaille sur les **futures front-month** (base tradable, *live*) : Brent (BZ=F) et WTI (CL=F).
Calendriers alignés par intersection des séances cotées (pas de forward-fill sur les prix
servant aux rendements = pas de look-ahead). Le spot long FRED (DCOILWTICO/DCOILBRENTEU) sert
la robustesse et l'historique long ; la courbe WTI (EIA RCLC1..4) alimente l'analyse de term
structure (bornée au 2024-04-05, arrêt de publication EIA).

## 3. Validation statistique

- **Non-stationnarité des niveaux — à la limite** : ADF sur les prix WTI p = 0,06, Brent
  p = 0,046 (rejet de justesse à 5 %) ; avec tendance, p = 0,21 et 0,16 → racine unitaire non
  rejetée. Les rendements sont clairement stationnaires. On retient des prix ≈ I(1), en le
  signalant.
- **Cointégration (Engle-Granger)** : β = 1,06, **p-value = 0,013**, ADF du résidu p = 0,003
  sur l'ensemble. Mais la relation **n'est pas stable** : p = 0,56 sur 2007-2013 (non
  cointégrés), 0,003 sur 2014-2019, 0,001 sur 2020-2026.
- **Vitesse de retour** : **demi-vie ≈ 18,9 jours** (−ln2/ln(1+φ), régression Δspread ~
  spread₋₁ sur le résidu statique), Hurst = 0,29 → horizon de quelques semaines ;
  **lookback du z-score ≈ 2× la demi-vie (40 j)**.

## 4. Signal et backtest

Spread tradable = **niveau** du spread à β dynamique, `spread_t = y_t − β_t·x_t` (et non
l'innovation du Kalman, qui est ≈ bruit blanc), centré par sa moyenne glissante. Signal =
**z-score glissant** : entrée si |z| > 2, sortie si |z| < 0,5, **sortie temporelle à 90
séances** (≈ 4-5× la demi-vie). Un stop au niveau z a été retiré (whipsaw avec l'entrée ;
0,45 → 0,40 avec un stop à |z| > 3). Hedge ratio estimé par **filtre de Kalman** (β en marche
aléatoire, mise à jour bayésienne causale), avec un **garde-fou d'entrée** : pas de trade si un
prix est ≤ 0 ou si β sort de [0,5 ; 1,5].

Rigueur de backtest : exécution ***trade-on-close*** (signal et exécution à la même clôture ;
à la clôture suivante le Sharpe passe de 0,45 à 0,36), β verrouillé à l'entrée, **coûts** de
2,5 bps/jambe sur le turnover, P&L **additif** sur notionnel constant (convention
market-neutral, robuste au WTI négatif). Kalman + z-score glissant = **walk-forward continu**.
Un test vérifie que le pipeline complet, calculé sur un préfixe des données, donne exactement
les mêmes rendements : aucune information future n'est utilisée.

**Le walk-forward, comparé à période égale.** Sharpe brut avant / après 2018 :

| Méthode | < 2018 | ≥ 2018 |
|---|---:|---:|
| Kalman en ligne (β chaque jour) | +0,45 | **+0,45** |
| β ré-estimé par blocs trimestriels (2 ans) | +0,69 | +0,14 |
| β figé estimé avant 2018 (split unique) | +0,63 | +0,11 |

Ce qui décroche après 2018, ce sont les méthodes à **β constant** ; le Kalman conserve sa
performance.

## 5. Ce qui a changé le résultat : un trade, et une mesure de vol

**Un trade aberrant.** Le 20/04/2020, le WTI clôture à −37 $ et le Kalman produit un β
négatif. Sans garde-fou, le trade ouvert le lendemain (β ≈ 0) est une vente sèche de Brent,
pas un trade de spread : −0,58 de P&L, plus que tout le gain net de l'échantillon. Avec le
garde-fou, le Sharpe brut passe de **+0,08 à +0,45**. La règle est économiquement justifiée,
mais elle a été motivée par un cas observé : un backtest peut tenir à un seul événement.

**Le vol-targeting ne crée pas l'alpha.** La vol doit être mesurée sur le spread supposé
toujours investi : mesurée sur les rendements de la stratégie (nuls hors position), elle
tombait à ~0 avant chaque entrée et envoyait la taille au plafond (3×). Corrigé, le
vol-target fixe le risque (vol 9,9 % pour 10 % visés) **sans changer le Sharpe** :

| | Brut | Risk-managed | Buy & Hold Brent |
|---|---:|---:|---:|
| Sharpe | +0,45 | **+0,45** | +0,24 |
| Rendement annualisé | +4,1 % | +4,5 % | +1,8 % (CAGR) |
| Volatilité | 9,1 % | 9,9 % | 38,4 % |
| Max drawdown | −20 % du notionnel | −20 % du notionnel | −87 % |
| Trades / hit rate | 73 / 59 % | 73 / 55 % | — |

**Par régime** (Sharpe risk-managed) : pré-glut 2007-2010 **+1,40** ; Cushing 2011-2014
+0,54 ; **krach & export ban 2014-2016 −0,40** ; normalisation 2017-2019 +0,29 ; COVID &
reprise 2020-2021 +0,39 ; guerre & inflation 2022+ +0,46. Le seul vrai mode de casse est la
**rupture structurelle 2014-2016** (shale + fin de l'export ban re-normalisent le spread → la
mean-reversion parie contre une tendance durable), un problème de **signal/régime** suivi par
la cointégration glissante et un flag « tradable ».

**Robustesse** : coûts (0,51 à 0 bps → 0,29 à 10 bps/jambe) ; surface seuil × lookback
positive partout (0,21 à 0,59 ; le défaut 0,45 n'en est pas le maximum) ; spot FRED au lieu des
futures : 0,32. **Points de fragilité assumés** : le bruit d'observation du Kalman (Sharpe 0,45
à 0,83 selon sa valeur — le défaut est le plus bas) et le time-stop (meilleur de sa grille).

## 6. Structure de courbe WTI

Sur **1983-2024** (historique EIA complet), le WTI est en **backwardation 52 % / contango
48 %** (0,3 % plat) ; roll yield annualisé moyen **≈ 0 %** — mais la sous-période **2004-2024
était nettement en contango** (roll yield ≈ −8,6 %). Sur 2007-2024, le spread Brent-WTI est
plus large en contango (6,2 $ contre 4,3 $ en backwardation), cohérent avec un excès d'offre
pesant sur le WTI/Cushing. En revanche, la backwardation **ne prédit pas** de rendements WTI
forward différents à 21 j (t = −0,4 sans recouvrement). Ce module est un **éclairage de
régime**, pas un signal démontré.

## 7. Limites et prolongements

- **Données** : cutoff EIA (2024-04-05) ; pas de courbe Brent gratuite (term structure sur WTI
  seul) ; clôtures yfinance non institutionnelles ; **séries front-month continues non ajustées
  des rolls** ; spot FRED décorrélé des futures en fin de période.
- **Méthode** : plusieurs choix de conception (retrait du stop, time-stop, garde-fou) ont été
  faits en connaissant l'échantillon — tout est causal, mais ce n'est pas un vrai test hors
  échantillon.
- **Prolongements** : intégrer le flag de cointégration glissante dans le dimensionnement
  (désactiver quand la relation casse), calibrer le bruit d'observation du Kalman sur une
  période d'amorçage, séries ajustées des rolls, forecast GARCH glissant pour le
  vol-targeting, jambe de courbe (roll yield) comme signal complémentaire.

## 8. Conclusion

Le spread Brent-WTI est cointégré sur l'ensemble de l'échantillon (pas sur chaque
sous-période) et mean-reverting ; un signal simple en tire un **Sharpe ≈ 0,45 net de coûts,
stable avant et après 2018**. L'audit a montré qu'un seul trade — un hedge devenu absurde le
jour du WTI négatif — suffisait à masquer cet edge, et que le vol-targeting règle le niveau
de risque sans produire de performance. Le projet documente ces points et les régimes où la
stratégie échoue plutôt que de les cacher — ce qui est, en pratique, le livrable le plus utile
pour un desk de valeur relative.
