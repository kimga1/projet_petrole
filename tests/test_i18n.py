"""Tests du module de traduction du dashboard (FR -> EN).

Garanties :
- toute clé littérale ``L("...")`` d'app.py existe dans TRANSLATIONS (sinon le
  fallback afficherait silencieusement du français en mode anglais) ;
- les gabarits gardent les mêmes champs ``{nom}`` des deux côtés ;
- ``make_tr`` : identité en FR, dictionnaire + fallback en EN.
"""
import ast
import re
from pathlib import Path

from dashboard.i18n import TRANSLATIONS, make_tr

APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def _literal_L_keys() -> list[str]:
    """Toutes les chaînes littérales passées à L() dans app.py (concat incluses)."""
    tree = ast.parse(APP.read_text())
    keys = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "L" and len(node.args) == 1
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            keys.append(node.args[0].value)
    assert len(keys) > 60, "l'extraction AST semble cassée"
    return keys


def test_all_literal_keys_are_translated():
    missing = sorted({k for k in _literal_L_keys() if k not in TRANSLATIONS})
    assert not missing, f"clés L() absentes de TRANSLATIONS : {missing}"


def test_placeholders_match():
    for fr, en in TRANSLATIONS.items():
        assert isinstance(en, str) and en.strip(), fr
        assert set(re.findall(r"{(\w+)}", fr)) == set(re.findall(r"{(\w+)}", en)), fr


def test_make_tr_behaviour():
    fr, en = make_tr("fr"), make_tr("en")
    assert fr("Paramètres") == "Paramètres"
    assert en("Paramètres") == "Parameters"
    assert en("clé inconnue, on garde le français") == \
        "clé inconnue, on garde le français"
