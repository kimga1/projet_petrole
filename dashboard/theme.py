"""Couleur secondaire du dashboard : préréglages et utilitaires couleur.

La couleur secondaire (« accent ») est choisie parmi des préréglages testés
sur le thème sombre. Elle est propagée de deux façons :
- interface (CSS/HTML) : via la variable CSS ``--accent`` (animable -> disco) ;
- graphiques Plotly    : en hexadécimal réel au moment du rendu (le SVG ne
  résout pas les variables CSS) ; app.py re-cible ensuite en CSS les éléments
  SVG portant exactement cette couleur vers var(--accent), pour qu'ils
  suivent aussi le disco (le vert/rouge des P&L, lui, reste stable).
"""
from __future__ import annotations

import colorsys

# Préréglages : lisibles sur fond #131417, et le texte noir du bandeau/des
# sélections reste contrasté sur chacun. Les clés (françaises) sont les
# valeurs internes stables des widgets ; l'affichage passe par i18n.
ACCENT_PRESETS: dict[str, str] = {
    "Ambre": "#FFA028",       # défaut historique du thème terminal
    "Cyan": "#22D3EE",
    "Menthe": "#2EE6A6",
    "Violet": "#A78BFA",
    "Rose": "#F472B6",
    "Bleu": "#6AA5FF",
}
DEFAULT_ACCENT = "Ambre"


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    """'#RRGGBB' -> (r, g, b)."""
    c = color.lstrip("#")
    if len(c) != 6:
        raise ValueError(f"couleur hex invalide: {color!r}")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def rgb_str(color: str) -> str:
    """'#RRGGBB' -> 'r,g,b' (pour composer des rgba() côté Plotly)."""
    r, g, b = hex_to_rgb(color)
    return f"{r},{g},{b}"


def mix_white(color: str, t: float) -> str:
    """Éclaircit une couleur en la mélangeant au blanc (t dans [0, 1])."""
    t = min(1.0, max(0.0, t))
    r, g, b = (round(c + (255 - c) * t) for c in hex_to_rgb(color))
    return f"#{r:02X}{g:02X}{b:02X}"


def resolve_accent(name: str | None) -> tuple[str, str, str]:
    """Nom de préréglage -> (hex, hex éclairci, 'r,g,b').

    Nom inconnu ou None (pill désélectionnée) -> préréglage par défaut.
    """
    color = ACCENT_PRESETS.get(name or "", ACCENT_PRESETS[DEFAULT_ACCENT])
    return color, mix_white(color, 0.40), rgb_str(color)


def hue_deg(color: str) -> float:
    """Teinte HSL (0-360°) — sert au filtre hue-rotate du disco sur les rasters."""
    r, g, b = (c / 255 for c in hex_to_rgb(color))
    return colorsys.rgb_to_hls(r, g, b)[0] * 360.0
