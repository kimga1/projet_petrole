"""Brent-WTI statistical arbitrage & term structure toolkit.

Package modulaire pour :
- charger et aligner les prix Brent/WTI (futures front-month + spot FRED),
- tester la cointégration et calibrer la mean-reversion,
- estimer un hedge ratio dynamique (rolling OLS / Kalman),
- générer un signal z-score et le backtester sans look-ahead (walk-forward),
- analyser la structure de courbe WTI (contango/backwardation).

Chaque choix méthodologique est documenté dans le README (section « Pourquoi »).
"""

__version__ = "0.1.0"
