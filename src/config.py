"""Configuration centrale : chemins, tickers, paramètres de stratégie, clé EIA.

Tout est centralisé ici pour que les modules, scripts et le dashboard partagent
exactement les mêmes conventions (schéma de données, paramètres par défaut).
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Chemins (relatifs au fichier -> robustes quel que soit le CWD)              #
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Sources de données                                                          #
# --------------------------------------------------------------------------- #
# Futures front-month (base tradable, live) via yfinance.
YF_TICKERS = {"wti": "CL=F", "brent": "BZ=F"}

# Spot long historique (robustesse) via FRED.
FRED_SERIES = {"wti": "DCOILWTICO", "brent": "DCOILBRENTEU"}

# Courbe WTI (contango/backwardation) via EIA : contrats 1 à 4 (Cushing, $/bbl).
# NB : publication arrêtée après 2024-04-05 -> historique borné à cette date.
EIA_CURVE_SERIES = {f"c{i}": f"PET.RCLC{i}.D" for i in range(1, 5)}
EIA_CUTOFF_DATE = "2024-04-05"

# Fenêtre d'historique par défaut.
DEFAULT_START = "2007-01-01"   # Brent future (BZ=F) dispo à partir de ~2007
DEFAULT_END = None             # None -> aujourd'hui

# Annualisation (jours de bourse).
TRADING_DAYS = 252

# --------------------------------------------------------------------------- #
# Paramètres de stratégie (surchargés dans le dashboard par des curseurs)     #
# --------------------------------------------------------------------------- #
@dataclass
class StrategyConfig:
    """Hyper-paramètres de la stratégie de mean-reversion sur le spread."""

    # --- Construction du spread / hedge ratio ---
    price_col_y: str = "brent"          # jambe dépendante (y) dans y = a + b*x
    price_col_x: str = "wti"            # jambe explicative (x)
    spread_kind: str = "resid"          # "resid" (cointégrant) | "price" | "log"
    hedge_method: str = "kalman"        # "ols" | "rolling_ols" | "kalman"
    ols_window: int = 60                # fenêtre du rolling OLS (jours)
    kalman_delta: float = 1e-4          # ratio bruit d'état / observation (Kalman)
    # NB : 1e-3 ($²) est très petit devant la variance réelle du résidu (~25 $²) ->
    # β très réactif. Le résultat y est sensible (cf. README) : ne pas le régler
    # en maximisant le Sharpe.
    kalman_obs_cov: float = 1e-3        # variance du bruit d'observation (Kalman)
    # Garde-fou d'entrée : Brent et WTI cotent à des niveaux proches -> β ≈ 1.
    # Pas d'entrée si β sort de [beta_min, beta_max] ou si un prix est <= 0
    # (WTI négatif d'avril 2020 : le Kalman y produit β < 0, la « couverture »
    # devient une position sèche sur le Brent). None = pas de borne.
    beta_min: float | None = 0.5
    beta_max: float | None = 1.5

    # --- Signal z-score ---
    # z_lookback ≈ 2× la demi-vie estimée (~19j) -> ~40j (choix principé, pas du
    # sur-fit ; cf. étude de sensibilité dans le notebook 03).
    z_lookback: int = 40                # fenêtre glissante du z-score (jours)
    z_in: float = 2.0                   # seuil d'entrée (|z| > z_in)
    z_out: float = 0.5                  # seuil de sortie (|z| < z_out)
    # Stop sur z DÉSACTIVÉ par défaut : un stop au niveau z entre en conflit avec
    # l'entrée (whipsaw stop/ré-entrée) ET coupe les grosses réversions depuis les
    # niveaux extrêmes -> empiriquement contre-productif. Le risque de queue est
    # géré par le vol-targeting + une sortie temporelle (cf. README).
    z_stop: float | None = None         # stop si divergence extrême (|z| > z_stop)
    # Sortie temporelle ≈ 4-5× la demi-vie (~19j) : si le spread n'a pas mean-reverti
    # en ~90j, l'équilibre a probablement changé (rupture de régime) -> on coupe.
    max_holding_days: int | None = 90   # sortie forcée après N jours

    # --- Coûts de transaction (par jambe, en points de base du notionnel) ---
    half_spread_bps: float = 2.0        # demi bid-ask par jambe
    commission_bps: float = 0.5         # commission par jambe

    # --- Sizing ---
    vol_target: float | None = None     # vol annualisée cible (None = pas de scaling)
    vol_lookback: int = 21              # fenêtre de vol réalisée pour le sizing
    max_leverage: float = 3.0           # borne du multiplicateur de position

    # --- Backtest ---
    # 1 = « trade-on-close » : signal calculé sur la clôture t-1 et exécuté à
    # CETTE clôture (P&L capté de t-1 à t). 2 = exécution à la clôture suivante
    # (plus prudent ; cf. sensibilité dans results/sensitivity_execution.csv).
    execution_lag: int = 1
    warmup: int = 252                   # jours d'amorçage avant le 1er trade

    @property
    def cost_bps_per_leg(self) -> float:
        """Coût total par jambe et par changement de position (bps)."""
        return self.half_spread_bps + self.commission_bps

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Régimes de marché connus (annotations pour graphiques & analyse par régime) #
# --------------------------------------------------------------------------- #
# Documenter où/pourquoi la stratégie peut casser (à défendre en entretien).
MARKET_EVENTS: list[dict] = [
    {"date": "2011-01-01", "label": "Cushing glut / discount WTI (2011-14)"},
    {"date": "2014-06-01", "label": "Krach pétrolier (offre US shale)"},
    {"date": "2015-12-18", "label": "Levée de l'export ban US"},
    {"date": "2020-04-20", "label": "WTI négatif (choc COVID)"},
    {"date": "2022-02-24", "label": "Guerre Russie-Ukraine"},
]

# Segments de régime pour la ventilation de performance (start, end, nom).
REGIME_SEGMENTS: list[dict] = [
    {"name": "Pré-glut (2007-2010)", "start": "2007-01-01", "end": "2010-12-31"},
    {"name": "WTI discount / Cushing glut (2011-2014)", "start": "2011-01-01", "end": "2014-06-30"},
    {"name": "Krach & export ban (2014-2016)", "start": "2014-07-01", "end": "2016-12-31"},
    {"name": "Normalisation (2017-2019)", "start": "2017-01-01", "end": "2019-12-31"},
    {"name": "COVID & reprise (2020-2021)", "start": "2020-01-01", "end": "2021-12-31"},
    {"name": "Guerre & inflation (2022+)", "start": "2022-01-01", "end": "2100-01-01"},
]


# --------------------------------------------------------------------------- #
# Clé API EIA : jamais en dur. Ordre : Streamlit secrets -> env -> .env       #
# --------------------------------------------------------------------------- #
def get_eia_api_key() -> str | None:
    """Récupère la clé EIA sans jamais l'écrire dans le code/versionnée.

    Priorité :
      1. st.secrets["EIA_API_KEY"] (déploiement Streamlit Cloud)
      2. variable d'environnement EIA_API_KEY
      3. fichier .env local (via python-dotenv)
    """
    # 1) Streamlit secrets (uniquement si streamlit est présent et lancé)
    try:  # pragma: no cover - dépend du runtime Streamlit
        import streamlit as st

        if "EIA_API_KEY" in st.secrets:
            key = str(st.secrets["EIA_API_KEY"]).strip()
            if key:
                return key
    except Exception:
        pass

    # 2) variable d'environnement directe
    key = os.environ.get("EIA_API_KEY")
    if key and key.strip():
        return key.strip()

    # 3) fichier .env
    try:
        from dotenv import load_dotenv

        load_dotenv(BASE_DIR / ".env")
        key = os.environ.get("EIA_API_KEY")
        if key and key.strip():
            return key.strip()
    except Exception:
        pass

    return None


# Instance par défaut réutilisable.
DEFAULT_CONFIG = StrategyConfig()
