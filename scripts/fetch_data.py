"""Récupère et met en cache toutes les données (futures, spot FRED, courbe EIA).

Usage :
    python scripts/fetch_data.py            # (re)télécharge tout
    python scripts/fetch_data.py --no-eia   # ignore la courbe EIA

La clé EIA est lue depuis .env / variable d'environnement (jamais en dur).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src import data_loader as dl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-eia", action="store_true", help="ne pas télécharger la courbe EIA")
    ap.add_argument("--start", default=config.DEFAULT_START)
    args = ap.parse_args()

    print(">> Futures front-month (yfinance CL=F, BZ=F)...")
    fut = dl.get_prices("futures", start=args.start, refresh=True)
    print(f"   OK {fut.shape} {fut.index.min().date()} -> {fut.index.max().date()}")

    print(">> Spot FRED (DCOILWTICO, DCOILBRENTEU)...")
    fred = dl.get_prices("fred", start=args.start, refresh=True)
    print(f"   OK {fred.shape} {fred.index.min().date()} -> {fred.index.max().date()}")

    if not args.no_eia:
        if config.get_eia_api_key():
            print(">> Courbe WTI EIA (RCLC1..4)...")
            curve = dl.get_curve(refresh=True)
            if curve is not None:
                print(f"   OK {curve.shape} {curve.index.min().date()} -> {curve.index.max().date()}")
            else:
                print("   (aucune donnée courbe)")
        else:
            print(">> Clé EIA absente -> courbe ignorée (ajouter EIA_API_KEY dans .env).")

    print("\nCache écrit dans", config.DATA_DIR)


if __name__ == "__main__":
    main()
