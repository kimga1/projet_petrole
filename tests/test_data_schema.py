"""Tests du schéma de données (lit le cache parquet, aucun accès réseau)."""
import pandas as pd
import pytest

from src import data_loader as dl


@pytest.mark.parametrize("source", ["futures", "fred"])
def test_prices_schema(source):
    df = dl.load_cache(f"prices_{source}")
    if df is None:
        pytest.skip(f"cache {source} absent (lancer scripts/fetch_data.py)")
    df = dl.align_prices(df)
    assert list(df.columns) == ["wti", "brent"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    assert not df.index.has_duplicates
    assert df.notna().all().all()          # inner join -> pas de trou


def test_cache_respects_end_date():
    """Un end borné ne doit pas fuiter de données futures (cache re-découpé)."""
    if dl.load_cache("prices_futures") is None:
        pytest.skip("cache futures absent")
    df = dl.get_prices("futures", end="2019-12-31")
    assert df.index.max() <= pd.Timestamp("2019-12-31")


def test_curve_schema():
    curve = dl.load_cache("eia_curve")
    if curve is None:
        pytest.skip("cache courbe EIA absent")
    curve = dl._clean_index(curve)
    assert set(["c1", "c2", "c3", "c4"]).issubset(curve.columns)
    assert str(curve.index.max().date()) <= "2024-04-30"   # cutoff EIA
