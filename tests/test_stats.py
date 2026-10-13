"""Tests des outils statistiques : demi-vie, cointégration, ADF, Hurst."""
import numpy as np
import pandas as pd

from src import stats_tests as st


def _dates(n):
    return pd.date_range("2010-01-01", periods=n, freq="B")


def test_half_life_ou_process():
    """Sur un OU/AR(1) de rho connu, la demi-vie estimée ≈ théorique."""
    rng = np.random.default_rng(0)
    n = 4000
    rho = 0.95
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = rho * s[t - 1] + rng.normal(0, 1)
    series = pd.Series(s, index=_dates(n))
    hl = st.half_life(series)
    theo = -np.log(2) / np.log(rho)   # ~13.5
    assert 0.6 * theo < hl < 1.5 * theo


def test_half_life_random_walk_is_large():
    """Une marche aléatoire (rho=1) n'a pas de mean-reversion -> demi-vie énorme."""
    rng = np.random.default_rng(1)
    s = pd.Series(np.cumsum(rng.normal(0, 1, 3000)), index=_dates(3000))
    hl = st.half_life(s)
    assert hl > 100 or np.isinf(hl)


def test_engle_granger_detects_cointegration():
    """y = 2x + bruit stationnaire, x marche aléatoire -> cointégrés, beta≈2."""
    rng = np.random.default_rng(2)
    n = 2000
    x = np.cumsum(rng.normal(0, 1, n)) + 50
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.9 * noise[t - 1] + rng.normal(0, 0.5)
    y = 2.0 * x + noise
    idx = _dates(n)
    res = st.engle_granger(pd.Series(y, index=idx), pd.Series(x, index=idx))
    assert res.is_cointegrated
    assert abs(res.beta - 2.0) < 0.1


def test_no_cointegration_independent_walks():
    """Deux marches aléatoires indépendantes : peu de faux positifs de coint.

    Test statistique sur plusieurs tirages (un tirage unique est flaky : le test
    a un taux d'erreur de type I ~ au niveau choisi).
    """
    rng = np.random.default_rng(3)
    n = 1500
    hits = 0
    for _ in range(20):
        x = np.cumsum(rng.normal(0, 1, n))
        y = np.cumsum(rng.normal(0, 1, n))
        idx = _dates(n)
        if st.engle_granger(pd.Series(y, index=idx), pd.Series(x, index=idx)).is_cointegrated:
            hits += 1
    assert hits <= 4   # << 20 : la plupart des paires ne sont pas cointégrées


def test_adf_stationary_vs_unit_root():
    """Taux de rejet ADF sur plusieurs tirages (robuste à la flakiness)."""
    rng = np.random.default_rng(4)
    n = 1500
    stat_hits = sum(
        st.adf_test(pd.Series(rng.normal(0, 1, n), index=_dates(n))).is_stationary
        for _ in range(20))
    walk_hits = sum(
        st.adf_test(pd.Series(np.cumsum(rng.normal(0, 1, n)), index=_dates(n))).is_stationary
        for _ in range(20))
    assert stat_hits >= 17   # un bruit i.i.d. est quasi toujours jugé stationnaire
    assert walk_hits <= 4    # une marche aléatoire l'est rarement (erreur type I)


def test_hurst_mean_reverting():
    """Le Hurst d'un bruit i.i.d. (mean-reverting) est < 0.5."""
    rng = np.random.default_rng(5)
    s = pd.Series(rng.normal(0, 1, 3000), index=_dates(3000))
    assert st.hurst_exponent(s) < 0.5
