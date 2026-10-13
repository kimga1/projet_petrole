"""Chargement, alignement et mise en cache des données Brent/WTI.

Schéma canonique (contrat partagé par tout le projet)
-----------------------------------------------------
- Prix : DataFrame indexé par un DatetimeIndex nommé ``date`` (trié, tz-naïf),
  colonnes ``["wti", "brent"]`` en USD/bbl.
- Courbe WTI : DataFrame même index, colonnes ``["c1", "c2", "c3", "c4"]``.

Sources (100 % gratuit)
-----------------------
- Futures front-month (tradable, live)  : yfinance -> CL=F (WTI), BZ=F (Brent).
- Spot long historique (robustesse)     : FRED -> DCOILWTICO, DCOILBRENTEU.
- Courbe WTI (contango/backwardation)   : EIA -> RCLC1..RCLC4 (<= 2024-04-05).

Choix « pourquoi » (à défendre)
-------------------------------
- On **aligne par intersection des jours cotés** (inner join) plutôt que de
  forward-fill : forward-fill sur des prix puis calcul de rendements crée des
  rendements nuls artificiels et introduit du look-ahead. On préfère perdre
  quelques jours fériés décalés que de contaminer les rendements.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

from . import config
from .config import DATA_DIR


# --------------------------------------------------------------------------- #
# Helpers cache parquet                                                        #
# --------------------------------------------------------------------------- #
def _cache_path(name: str) -> Path:
    return DATA_DIR / f"{name}.parquet"


def save_cache(df: pd.DataFrame, name: str) -> Path:
    path = _cache_path(name)
    df.to_parquet(path)
    return path


def load_cache(name: str) -> pd.DataFrame | None:
    path = _cache_path(name)
    if path.exists():
        return pd.read_parquet(path)
    return None


def _clean_index(df: pd.DataFrame) -> pd.DataFrame:
    """Index datetime tz-naïf, nommé 'date', trié, sans doublons."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


# --------------------------------------------------------------------------- #
# 1) Futures front-month via yfinance                                         #
# --------------------------------------------------------------------------- #
def _extract_close(raw: pd.DataFrame, ticker: str) -> pd.Series:
    """Extrait la série de clôture d'un ticker depuis un download yfinance."""
    if isinstance(raw.columns, pd.MultiIndex):
        # Colonnes (field, ticker) ; on cherche 'Close'.
        lvl0 = raw.columns.get_level_values(0)
        field = "Close" if "Close" in set(lvl0) else lvl0[0]
        sub = raw[field]
        if isinstance(sub, pd.DataFrame):
            if ticker in sub.columns:
                return sub[ticker]
            # Ticker absent : on ÉCHOUE plutôt que de renvoyer la mauvaise
            # colonne (risque de données Brent étiquetées WTI).
            raise KeyError(f"ticker {ticker!r} absent du download yfinance "
                           f"(colonnes: {list(sub.columns)})")
        return sub
    # Colonnes plates (un seul ticker).
    if "Close" in raw.columns:
        return raw["Close"]
    return raw.iloc[:, 0]


def fetch_futures(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Télécharge WTI (CL=F) et Brent (BZ=F) front-month via yfinance."""
    import yfinance as yf

    start = start or config.DEFAULT_START
    end = end or pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    tickers = list(config.YF_TICKERS.values())  # ["CL=F", "BZ=F"]

    raw = yf.download(
        tickers, start=start, end=end,
        auto_adjust=False, progress=False, group_by="column",
    )
    if raw is None or len(raw) == 0:
        raise RuntimeError("yfinance n'a renvoyé aucune donnée (réseau ? tickers ?)")

    out = pd.DataFrame({
        name: _extract_close(raw, tick) for name, tick in config.YF_TICKERS.items()
    })
    out = _clean_index(out).dropna(how="all")
    return out


# --------------------------------------------------------------------------- #
# 2) Spot FRED (historique long)                                              #
# --------------------------------------------------------------------------- #
def _fred_csv(series_id: str) -> pd.Series:
    """Télécharge une série FRED via l'endpoint CSV public (sans clé)."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    date_col = df.columns[0]              # 'observation_date' ou 'DATE'
    val_col = df.columns[1]
    s = pd.Series(
        pd.to_numeric(df[val_col], errors="coerce").values,
        index=pd.to_datetime(df[date_col]),
        name=series_id,
    )
    return s.dropna()


def fetch_fred_spot(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Spot WTI/Brent depuis FRED (pandas-datareader, fallback CSV)."""
    start = start or config.DEFAULT_START
    end = end or pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    series = config.FRED_SERIES  # {"wti": "DCOILWTICO", "brent": "DCOILBRENTEU"}

    data = None
    try:
        from pandas_datareader import data as web

        data = web.DataReader(list(series.values()), "fred", start, end)
        data = data.rename(columns={v: k for k, v in series.items()})
    except Exception:
        data = None

    if data is None:  # fallback CSV robuste
        cols = {}
        for name, sid in series.items():
            cols[name] = _fred_csv(sid)
        data = pd.DataFrame(cols)

    data = _clean_index(data)
    data = data.loc[(data.index >= pd.Timestamp(start)) & (data.index <= pd.Timestamp(end))]
    return data.dropna(how="all")


# --------------------------------------------------------------------------- #
# 3) Courbe WTI via EIA (RCLC1..4)                                            #
# --------------------------------------------------------------------------- #
def _fetch_eia_series(legacy_id: str, api_key: str, page: int = 5000) -> pd.Series:
    """Une série EIA via la route v2 /seriesid (accepte les IDs legacy).

    L'API v2 plafonne chaque réponse à 5000 lignes : on **pagine** (length +
    offset) pour récupérer tout l'historique (RCLC1 remonte à 1983), au lieu de
    tronquer silencieusement aux 5000 lignes les plus récentes.
    """
    url = f"https://api.eia.gov/v2/seriesid/{legacy_id}"
    all_rows: list[dict] = []
    offset = 0
    while True:
        resp = requests.get(
            url, params={"api_key": api_key, "length": page, "offset": offset},
            timeout=30,
        )
        resp.raise_for_status()
        response = resp.json().get("response", {})
        rows = response.get("data", [])
        all_rows.extend(rows)
        total = int(response.get("total", len(all_rows)) or 0)
        offset += page
        if len(rows) < page or offset >= total:
            break

    if not all_rows:
        return pd.Series(dtype="float64", name=legacy_id)
    df = pd.DataFrame(all_rows)
    val_col = "value" if "value" in df.columns else df.columns[-1]
    s = pd.Series(
        pd.to_numeric(df[val_col], errors="coerce").values,
        index=pd.to_datetime(df["period"]),
        name=legacy_id,
    )
    return s[~s.index.duplicated()].dropna().sort_index()


def fetch_eia_curve(api_key: str | None = None) -> pd.DataFrame | None:
    """Courbe WTI contrats 1..4. Renvoie None si pas de clé (dégradation propre)."""
    api_key = api_key or config.get_eia_api_key()
    if not api_key:
        return None
    cols = {}
    for name, sid in config.EIA_CURVE_SERIES.items():   # c1..c4 -> PET.RCLCx.D
        cols[name] = _fetch_eia_series(sid, api_key)
    curve = pd.DataFrame(cols)
    curve = _clean_index(curve)
    return curve.dropna(how="all")


# --------------------------------------------------------------------------- #
# Alignement Brent/WTI                                                         #
# --------------------------------------------------------------------------- #
def align_prices(df: pd.DataFrame, how: str = "inner") -> pd.DataFrame:
    """Aligne les colonnes de prix. ``how='inner'`` = jours cotés communs."""
    out = _clean_index(df)
    if how == "inner":
        out = out.dropna(how="any")
    else:
        out = out.dropna(how="all")
    return out


def _slice_dates(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    """Restreint à [start, end] (bornes incluses). Sert à re-découper le cache."""
    if start is not None:
        df = df.loc[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df.loc[df.index <= pd.Timestamp(end)]
    return df


# --------------------------------------------------------------------------- #
# Entrées de haut niveau (avec cache)                                          #
# --------------------------------------------------------------------------- #
def get_prices(
    source: str = "futures",
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
    align: str = "inner",
) -> pd.DataFrame:
    """Prix Brent/WTI alignés.

    Parameters
    ----------
    source : "futures" (yfinance) ou "fred" (spot long historique).
    refresh : force le retéléchargement (sinon lit le cache parquet).
    """
    name = f"prices_{source}"
    if not refresh:
        cached = load_cache(name)
        if cached is not None:
            # IMPORTANT : re-découper sur [start, end] même en cache-hit, sinon
            # une plage bornée (ex. end='2019-12-31' pour un OOS) renverrait
            # toute la plage cachée -> fuite de données futures.
            return _slice_dates(align_prices(cached, how=align), start, end)

    if source == "futures":
        raw = fetch_futures(start, end)
    elif source == "fred":
        raw = fetch_fred_spot(start, end)
    else:
        raise ValueError(f"source inconnue: {source!r}")

    save_cache(raw, name)
    return _slice_dates(align_prices(raw, how=align), start, end)


def get_curve(refresh: bool = False, api_key: str | None = None) -> pd.DataFrame | None:
    """Courbe WTI (c1..c4) avec cache. None si indisponible (pas de clé)."""
    name = "eia_curve"
    if not refresh:
        cached = load_cache(name)
        if cached is not None:
            return _clean_index(cached)

    curve = fetch_eia_curve(api_key=api_key)
    if curve is None or curve.empty:
        return None
    save_cache(curve, name)
    return curve


def data_status() -> dict:
    """État du cache local (pour le dashboard)."""
    status = {}
    for name in ("prices_futures", "prices_fred", "eia_curve"):
        df = load_cache(name)
        if df is None:
            status[name] = {"available": False}
        else:
            df = _clean_index(df)
            status[name] = {
                "available": True,
                "rows": len(df),
                "start": str(df.index.min().date()) if len(df) else None,
                "end": str(df.index.max().date()) if len(df) else None,
            }
    return status
