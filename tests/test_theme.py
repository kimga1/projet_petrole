"""Tests des utilitaires de couleur secondaire du dashboard."""
import re

import numpy as np

from dashboard.theme import (
    ACCENT_PRESETS,
    DEFAULT_ACCENT,
    hex_to_rgb,
    hue_deg,
    mix_white,
    resolve_accent,
    rgb_str,
)


def test_presets_are_valid_and_legible():
    assert DEFAULT_ACCENT in ACCENT_PRESETS
    assert ACCENT_PRESETS[DEFAULT_ACCENT] == "#FFA028"   # défaut historique
    for name, color in ACCENT_PRESETS.items():
        assert re.fullmatch(r"#[0-9A-Fa-f]{6}", color), name
        # Lisible sur fond sombre ET texte noir lisible dessus (bandeau,
        # sélections) : luminance relative dans une bande médiane-haute.
        r, g, b = (c / 255 for c in hex_to_rgb(color))
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        assert 0.35 < lum < 0.85, f"{name}: luminance {lum:.2f}"


def test_hex_rgb_roundtrip():
    assert hex_to_rgb("#FFA028") == (255, 160, 40)
    assert rgb_str("#FFA028") == "255,160,40"


def test_mix_white_bounds():
    assert mix_white("#FFA028", 0.0) == "#FFA028"
    assert mix_white("#FFA028", 1.0) == "#FFFFFF"
    r0, g0, b0 = hex_to_rgb("#FFA028")
    r1, g1, b1 = hex_to_rgb(mix_white("#FFA028", 0.4))
    assert r1 >= r0 and g1 > g0 and b1 > b0            # strictement plus clair


def test_resolve_accent_fallback():
    hex_, lt, rgb = resolve_accent("Cyan")
    assert hex_ == "#22D3EE" and rgb == "34,211,238"
    assert np.all(np.array(hex_to_rgb(lt)) >= np.array(hex_to_rgb(hex_)))
    # inconnu / None (pill désélectionnée) -> défaut, jamais d'erreur
    assert resolve_accent("n'existe pas")[0] == ACCENT_PRESETS[DEFAULT_ACCENT]
    assert resolve_accent(None)[0] == ACCENT_PRESETS[DEFAULT_ACCENT]


def test_hue_deg():
    assert hue_deg("#FF0000") == 0.0
    assert hue_deg("#00FF00") == 120.0
    assert hue_deg("#0000FF") == 240.0
    assert 30 < hue_deg("#FFA028") < 40          # ambre ~33°
