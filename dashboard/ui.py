"""Couche présentation du dashboard : jetons de design, gabarits HTML, widgets.

Regroupe tout ce qui dessine sans rien calculer :
- jetons de couleur/typo du thème TERMINAL et template Plotly ;
- chargement de la feuille de style (style.css) et bloc CSS dynamique de la
  couleur secondaire (+ animation du mode disco) ;
- gabarits HTML (cartes KPI, badges, barres de progression) et widgets
  composés (panneau ``card``, slider à rangée label/valeur, pills avec
  repli radio pour les vieilles versions de Streamlit) ;
- bouton maison de réouverture de la sidebar (JS injecté).

app.py orchestre ; i18n.py traduit ; theme.py fournit les préréglages de
couleur ; src/ calcule.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components

from src import regime as rg

# --------------------------------------------------------------------------- #
# Jetons de design — TERMINAL (gris profonds / accent / mono)                 #
# La couleur secondaire (ACCENT…) est dynamique : résolue par app.py via      #
# theme.resolve_accent, puis propagée en CSS par accent_style().              #
# --------------------------------------------------------------------------- #
CARD_INSET = "#212429"
BORDER = "#34373E"
DIVIDER = "#24262B"
GRID = "#26282E"
GREEN = "#00C86E"
RED = "#FF4438"
NEUTRAL = "#B8B8B8"     # séries de comparaison (blanc-gris)
TXT = "#F5F5F5"
TXT_BODY = "#D9D9D9"
TXT_MUTED = "#9A9A9A"
TXT_FAINT = "#7A8089"
MONO = "'IBM Plex Mono', monospace"

LEGEND_TR = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)


def setup_plotly_template(accent: str) -> None:
    """Installe le template Plotly du thème (accent = hex de la couleur choisie)."""
    pio.templates["bw"] = go.layout.Template(layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=MONO, color=TXT_BODY, size=10.5),
        xaxis=dict(gridcolor=GRID, griddash="dot", zerolinecolor=BORDER,
                   linecolor=BORDER, tickfont=dict(color=TXT_FAINT, size=9.5)),
        yaxis=dict(gridcolor=GRID, griddash="dot", zerolinecolor=BORDER,
                   linecolor=BORDER, tickfont=dict(color=TXT_FAINT, size=9.5)),
        colorway=[accent, NEUTRAL, GREEN, RED, "#8ab4f8"],
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        hoverlabel=dict(bgcolor="#1E2126", bordercolor=BORDER,
                        font=dict(family=MONO, color="#E8E8E8", size=10.5)),
    ))
    pio.templates.default = "bw"


# --------------------------------------------------------------------------- #
# CSS : feuille statique + bloc dynamique de la couleur secondaire            #
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_base_css() -> str:
    """Feuille de style du thème (dashboard/style.css), lue une seule fois."""
    return (Path(__file__).parent / "style.css").read_text()


def accent_style(accent: str, accent_lt: str, accent_rgb: str,
                 accent_hue: float, disco: bool) -> str:
    """Bloc <style> de la couleur secondaire courante.

    Mode normal : fixe --accent / --accent-lt aux hex du préréglage.
    Mode disco  : la teinte tourne en continu côté client (propriété CSS
    enregistrée + animation, ~6 s par tour, fluide, sans rerun) ; tout ce qui
    référence var(--accent) suit.

    Graphiques : les éléments SVG de Plotly qui portent EXACTEMENT la couleur
    choisie (traits, remplissages) sont re-ciblés vers var(--accent) -> ils
    suivent la couleur ET le disco. Le vert/rouge des P&L et le gris des
    comparaisons ne matchent pas -> stables. Le heatmap de sensibilité est un
    raster (non ciblable élément par élément) : en disco, il tourne via un
    filtre hue-rotate RELATIF à la teinte choisie — il ne contient que la
    couleur d'accent, du gris et du blanc, donc aucun vert/rouge n'est touché.
    """
    rgb = accent_rgb.replace(",", ", ")     # sérialisation CSS : "r, g, b"
    chart = (
        f' [data-testid="stPlotlyChart"] [style*="stroke: rgb({rgb})"]'
        f' {{ stroke: var(--accent) !important; }}'
        f' [data-testid="stPlotlyChart"] [style*="fill: rgb({rgb})"]'
        f' {{ fill: var(--accent) !important; }}'
        f' [data-testid="stPlotlyChart"] [style*="fill: rgba({rgb}, 0.12)"]'
        f' {{ fill: color-mix(in srgb, var(--accent) 12%, transparent) !important; }}'
    )
    if not disco:
        return (f"<style>:root {{ --accent: {accent};"
                f" --accent-lt: {accent_lt}; }} {chart}</style>")
    return (
        "<style>"
        "@property --accent-h { syntax: '<angle>'; inherits: true;"
        " initial-value: 30deg; }"
        ":root { --accent: hsl(var(--accent-h) 95% 62%);"
        " --accent-lt: hsl(var(--accent-h) 95% 76%);"
        " animation: bw-disco 6s linear infinite; }"
        "@keyframes bw-disco { from { --accent-h: 0deg; }"
        " to { --accent-h: 360deg; } }"
        f"{chart}"
        f' [class*="st-key-heat_sens"] [data-testid="stPlotlyChart"]'
        f' {{ filter: hue-rotate(calc(var(--accent-h) - {accent_hue:.0f}deg)); }}'
        "</style>")


# --------------------------------------------------------------------------- #
# Gabarits HTML                                                                #
# --------------------------------------------------------------------------- #
def kpi_card(label: str, value: str, status: str | None = None,
             status_color: str = TXT_FAINT) -> str:
    s = (f'<div style="color:{status_color};font-size:10px;font-weight:600;'
         f'margin-top:3px">{status}</div>') if status else ""
    return (f'<div class="bw-card"><div class="bw-lbl">{label}</div>'
            f'<div class="bw-val">{value}</div>{s}</div>')


def progress_row(label: str, pct: float, color: str) -> str:
    return (
        f'<div style="margin:8px 0">'
        f'<div style="display:flex;justify-content:space-between;align-items:center">'
        f'<span style="color:{TXT_BODY};font-size:11.5px">{label}</span>'
        f'<span class="mono" style="color:{TXT};font-size:11.5px;font-weight:600">{pct:.1f} %</span></div>'
        f'<div style="background:{CARD_INSET};border-radius:999px;height:7px;margin-top:4px">'
        f'<div style="width:{min(pct, 100):.1f}%;height:7px;border-radius:999px;background:{color}"></div>'
        f'</div></div>')


def side_badge(side: int) -> str:
    if side > 0:
        return '<span class="bw-badge up">LONG</span>'
    return '<span class="bw-badge down">SHORT</span>'


# --------------------------------------------------------------------------- #
# Widgets composés                                                             #
# --------------------------------------------------------------------------- #
_panel_counter = {"n": 0}


def card(title: str | None = None):
    """Container-panneau ; le titre est rendu DANS le panneau.

    Chaque panneau reçoit key="panel-N" (compteur réinitialisé à chaque rerun,
    donc stable) : le CSS cible sa classe st-key-panel-N — nécessaire car le
    markup des containers bordés change selon la version de Streamlit.
    """
    _panel_counter["n"] += 1
    try:
        c = st.container(border=True, key=f"panel-{_panel_counter['n']}")
    except TypeError:                     # vieille version sans key=
        c = st.container(border=True)
    if title:
        c.markdown(f'<div class="bw-ct">{title}</div>', unsafe_allow_html=True)
    return c


def param_row(label: str, value: str) -> None:
    st.markdown(f'<div class="bw-prow"><span class="l">{label}</span>'
                f'<span class="v">{value}</span></div>', unsafe_allow_html=True)


def param_slider(label, key, mn, mx, default, step, fmt):
    """Slider façon mockup : rangée label/valeur au-dessus, widget sans label.

    La valeur affichée vient de session_state (à jour au rerun). La paire est
    isolée dans un st.container(key="pair-…") : le CSS ne resserre que ce
    conteneur (pas de marge négative -> pas de superposition possible).
    """
    with st.container(key=f"pair-{key}"):
        val = st.session_state.get(key, default)
        param_row(label, fmt(val))
        return st.slider(label, mn, mx, default, step, key=key,
                         label_visibility="collapsed")


def pills_or_radio(label, options, default, key=None, format_func=str):
    if hasattr(st, "pills"):
        return st.pills(label, options, default=default, format_func=format_func,
                        label_visibility="collapsed", key=key) or default
    return st.radio(label, options, index=options.index(default), horizontal=True,
                    label_visibility="collapsed", key=key, format_func=format_func)


def add_event_lines(fig: go.Figure) -> None:
    """Traits verticaux discrets aux événements de marché connus (config)."""
    for _, ev in rg.event_lines().iterrows():
        fig.add_vline(x=ev["date"], line_width=1, line_dash="dot",
                      line_color="#4A4A4A", opacity=0.7)


# --------------------------------------------------------------------------- #
# Réouverture de la sidebar                                                    #
# --------------------------------------------------------------------------- #
def inject_sidebar_reopener(tooltip: str = "Rouvrir les paramètres") -> None:
    """Bouton de réouverture de sidebar 100 % maison (fiable).

    Le bouton natif de Streamlit n'est monté qu'au survol du header et son
    layout est fragile. On injecte donc dans la page parente (iframe
    components -> window.parent) un vrai bouton fixe, affiché dès que la
    sidebar est repliée (MutationObserver + intervalle de secours). Son clic
    déclenche le survol du header puis clique le bouton natif, avec retries.
    """
    components.html(_REOPENER_JS.replace("__TOOLTIP__", tooltip), height=0)


_REOPENER_JS = """<script>
const P = window.parent.document;
function ensureBtn(){
  let b = P.getElementById('bw-reopen');
  if(!b){
    b = P.createElement('button');
    b.id = 'bw-reopen'; b.textContent = '\\u00bb';
    b.title = '__TOOLTIP__';
    Object.assign(b.style, {position:'fixed', top:'8px', left:'8px',
      zIndex:1000000, width:'28px', height:'26px', background:'#1A1C20',
      color:'var(--accent)', border:'1px solid var(--accent)', borderRadius:'0',
      font:"700 14px 'IBM Plex Mono', monospace", cursor:'pointer',
      display:'none', padding:'0', lineHeight:'24px'});
    b.onclick = () => {
      const hdr = P.querySelector('[data-testid="stHeader"]');
      if(hdr){
        hdr.style.pointerEvents = 'auto';
        for(const t of ['pointerover','pointerenter','mouseover','mouseenter','mousemove'])
          hdr.dispatchEvent(new MouseEvent(t, {bubbles:true, clientX:20, clientY:20}));
      }
      let tries = 0;
      const iv = setInterval(() => {
        const sb = P.querySelector('[data-testid="stSidebar"]');
        if(sb && sb.getAttribute('aria-expanded') === 'true'){ clearInterval(iv); return; }
        const nb = P.querySelector('button[data-testid="stExpandSidebarButton"]');
        if(nb) nb.click();
        if(++tries > 12) clearInterval(iv);
      }, 120);
    };
    P.body.appendChild(b);
  }
  return b;
}
function sync(){
  const sb = P.querySelector('[data-testid="stSidebar"]');
  const b = ensureBtn();
  // Replié = pas explicitement 'true' OU largeur quasi nulle (robuste aux
  // variations d'attributs entre versions de Streamlit).
  const closed = !sb || sb.getAttribute('aria-expanded') !== 'true'
                 || sb.getBoundingClientRect().width < 40;
  b.style.display = closed ? 'block' : 'none';
}
sync();
const sb = P.querySelector('[data-testid="stSidebar"]');
if(sb) new MutationObserver(sync).observe(sb, {attributes:true, attributeFilter:['aria-expanded']});
setInterval(sync, 700);
</script>"""
