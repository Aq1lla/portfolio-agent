#!/usr/bin/env python3
"""
Portfolio Agent — Inicijalizacija baze i provjera sustava.
Pokreni ovo nakon kloniranja projekta.

Korištenje:
    python scripts/setup_db.py
    python scripts/setup_db.py --config config/strategy.yaml
"""

import argparse
import sys
from pathlib import Path

# Dodaj root direktorij u path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.models import load_config
from src.db import Database


def main():
    parser = argparse.ArgumentParser(description="Inicijalizacija Portfolio Agenta")
    parser.add_argument(
        "--config",
        default="config/strategy.yaml",
        help="Putanja do konfiguracijske datoteke",
    )
    args = parser.parse_args()

    config_path = ROOT / args.config

    print("=" * 60)
    print("  Portfolio Agent — Setup")
    print("=" * 60)

    # 1. Učitaj i validiraj konfiguraciju
    print(f"\n[1/3] Učitavam konfiguraciju: {config_path}")
    try:
        config = load_config(str(config_path))
        print(f"  ✓ Portfolio: {config.portfolio.name}")
        print(f"  ✓ Benchmark: {config.portfolio.benchmark}")
        print(f"  ✓ Simboli: {', '.join(config.allocation.symbols)}")
        print(f"  ✓ Cash rezerva: {config.allocation.cash_target * 100:.1f}%")
        print(f"  ✓ DCA: {'uključen' if config.dca.enabled else 'isključen'}"
              f" (${config.dca.amount}/{config.dca.frequency.value})")
        print(f"  ✓ Paper trading: {'DA' if config.broker.paper_trading else 'NE — ŽIVIMO!'}")
    except Exception as e:
        print(f"  ✗ Greška: {e}")
        sys.exit(1)

    # 2. Inicijaliziraj bazu
    db_path = ROOT / config.data.store_path
    print(f"\n[2/3] Inicijaliziram bazu: {db_path}")
    try:
        db = Database(db_path=str(db_path))
        stats = db.get_db_stats()
        print(f"  ✓ Baza kreirana ({stats['db_size_kb']:.1f} KB)")
        print(f"  ✓ Barovi: {stats['price_bars']}")
        print(f"  ✓ Snapshoti: {stats['snapshots']}")
        print(f"  ✓ Nalozi: {stats['orders']}")
    except Exception as e:
        print(f"  ✗ Greška: {e}")
        sys.exit(1)

    # 3. Provjera dependencija
    print("\n[3/3] Provjera dependencija:")
    deps = {
        "pydantic": "pydantic",
        "yaml": "pyyaml",
        "pandas": "pandas",
        "numpy": "numpy",
    }
    optional_deps = {
        "alpaca_trade_api": "alpaca-trade-api",
        "yfinance": "yfinance",
    }

    all_ok = True
    for module, package in deps.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} — NEDOSTAJE (pip install {package})")
            all_ok = False

    for module, package in optional_deps.items():
        try:
            __import__(module)
            print(f"  ✓ {package} (opcionalno)")
        except ImportError:
            print(f"  ○ {package} (opcionalno, nije instalirano)")

    # Rezultat
    print("\n" + "=" * 60)
    if all_ok:
        print("  ✓ Setup uspješan! Sustav je spreman.")
        print("\n  Sljedeći koraci:")
        print("  1. Kopiraj config/secrets.example.yaml → config/secrets.yaml")
        print("  2. Unesi Alpaca API ključeve u secrets.yaml")
        print("  3. Pokreni testove: pytest tests/ -v")
        print("  4. Inicijaliziraj podatke (Faza 1 kompletna)")
    else:
        print("  ✗ Nedostaju dependencije. Pokreni:")
        print("    pip install -r requirements.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
