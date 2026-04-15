"""
Portfolio Agent — Runner
Automatsko periodičko pokretanje agenta putem APScheduler-a.

Raspored:
  - run_cycle():        svako X minuta dok je tržište otvoreno
  - run_daily_summary(): 21:05 UTC (nakon zatvaranja NYSE)
  - run_new_day():       14:25 UTC (5 min prije otvaranja NYSE)

Pokretanje:
  python runner.py
  python runner.py --config config/strategy.yaml --secrets config/secrets.yaml
  python runner.py --dry-run          # Bez Alpaca konekcije
  python runner.py --once             # Jedan ciklus i izlaz
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Logging setup (prije svega ostalog) ──
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Portfolio Agent — Automatizirani Portfolio Manager"
    )
    parser.add_argument(
        "--config", default="config/strategy.yaml",
        help="Putanja do konfiguracijske datoteke (default: config/strategy.yaml)"
    )
    parser.add_argument(
        "--secrets", default="config/secrets.yaml",
        help="Putanja do secrets datoteke (default: config/secrets.yaml)"
    )
    parser.add_argument(
        "--interval", type=int, default=5,
        help="Interval ciklusa u minutama (default: 5)"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Pokreni jedan ciklus i izađi"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Pokreni bez Alpaca konekcije (dry-run mod)"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Razina logiranja (default: INFO)"
    )
    return parser.parse_args()


def create_agent(args):
    """Kreiraj i inicijaliziraj agenta."""
    from src.agent import Agent

    secrets_path = args.secrets if not args.dry_run else None

    agent = Agent.from_config(
        config_path=args.config,
        secrets_path=secrets_path,
    )
    return agent


def run_once(agent):
    """Pokreni jedan ciklus."""
    logger.info("Pokrećem jednokratni ciklus...")
    result = agent.initialize()
    logger.info("Inicijalizacija: %s", result["status"])

    cycle = agent.run_cycle()
    logger.info(
        "Ciklus završen: market_open=%s, naloga=%d/%d, greške=%d",
        cycle["market_open"], cycle["orders_executed"],
        cycle["orders_generated"], len(cycle["errors"])
    )

    summary = agent.run_daily_summary()
    print("\n" + summary)
    return cycle


def run_scheduled(agent, interval_minutes: int):
    """Pokreni agenta s APScheduler-om za 24/7 rad."""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.triggers.interval import IntervalTrigger
    except ImportError:
        logger.error(
            "APScheduler nije instaliran. Pokreni: pip install apscheduler"
        )
        sys.exit(1)

    scheduler = BlockingScheduler()

    # ── Inicijaliziraj agenta ──
    init_result = agent.initialize()
    logger.info("Agent inicijaliziran: %s", init_result["status"])

    # ── Job 1: Glavni ciklus — svako X minuta, pon-pet, tržišni sati ──
    # NYSE: 9:30-16:00 ET = 14:30-21:00 UTC
    # Pokrećemo od 14:30 do 21:00 UTC s malo margine
    def cycle_job():
        try:
            result = agent.run_cycle()
            if result["market_open"]:
                logger.info(
                    "Ciklus #%d: naloga=%d/%d",
                    result["cycle"], result["orders_executed"],
                    result["orders_generated"]
                )
            else:
                logger.debug("Ciklus #%d: tržište zatvoreno", result["cycle"])
        except Exception as e:
            logger.error("Greška u ciklusu: %s", e, exc_info=True)

    scheduler.add_job(
        cycle_job,
        IntervalTrigger(minutes=interval_minutes),
        id="main_cycle",
        name=f"Glavni ciklus (svako {interval_minutes} min)",
        max_instances=1,
        coalesce=True,
    )

    # ── Job 2: Dnevni sažetak — 21:05 UTC (5 min nakon zatvaranja) ──
    def daily_summary_job():
        try:
            logger.info("Generiram dnevni sažetak...")
            summary = agent.run_daily_summary()
            logger.info("Dnevni sažetak poslan.")
        except Exception as e:
            logger.error("Greška u dnevnom sažetku: %s", e, exc_info=True)

    scheduler.add_job(
        daily_summary_job,
        CronTrigger(hour=21, minute=5, day_of_week="mon-fri", timezone="UTC"),
        id="daily_summary",
        name="Dnevni sažetak (21:05 UTC)",
    )

    # ── Job 3: Novi dan — 14:25 UTC (5 min prije otvaranja) ──
    def new_day_job():
        try:
            logger.info("Priprema za novi tržišni dan...")
            agent.run_new_day()
            logger.info("Novi dan pripremljen.")
        except Exception as e:
            logger.error("Greška u pripremi novog dana: %s", e, exc_info=True)

    scheduler.add_job(
        new_day_job,
        CronTrigger(hour=14, minute=25, day_of_week="mon-fri", timezone="UTC"),
        id="new_day",
        name="Priprema novog dana (14:25 UTC)",
    )

    # ── Job 4: Heartbeat — svako 15 minuta (provjera da agent radi) ──
    def heartbeat_job():
        status = agent.get_status()
        logger.debug(
            "Heartbeat: equity=$%.2f, cycles=%d, halted=%s",
            status.get("equity", 0),
            status.get("cycle_count", 0),
            status.get("risk", {}).get("halted", False),
        )

    scheduler.add_job(
        heartbeat_job,
        IntervalTrigger(minutes=15),
        id="heartbeat",
        name="Heartbeat (svako 15 min)",
    )

    # ── Graceful shutdown ──
    def shutdown_handler(signum, frame):
        logger.info("Primljen signal za gašenje (%s)...", signum)
        agent.shutdown()
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # ── Start ──
    logger.info("=" * 60)
    logger.info("Portfolio Agent Runner pokrenut")
    logger.info("  Interval ciklusa: %d min", interval_minutes)
    logger.info("  Dnevni sažetak: 21:05 UTC")
    logger.info("  Priprema novog dana: 14:25 UTC")
    logger.info("  Heartbeat: svako 15 min")
    logger.info("  Zaustavljanje: Ctrl+C")
    logger.info("=" * 60)

    # Pokreni prvi ciklus odmah
    cycle_job()

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Agent zaustavljen.")
        agent.shutdown()


def main():
    args = parse_args()

    # Setup logging
    setup_logging(level=args.log_level)

    logger.info("Portfolio Agent v0.1.0")
    logger.info("Config: %s", args.config)
    logger.info("Secrets: %s", args.secrets if not args.dry_run else "(dry-run)")
    logger.info("Interval: %d min", args.interval)

    # Provjeri da config postoji
    if not Path(args.config).exists():
        logger.error("Konfiguracijska datoteka ne postoji: %s", args.config)
        sys.exit(1)

    if not args.dry_run and not Path(args.secrets).exists():
        logger.error("Secrets datoteka ne postoji: %s", args.secrets)
        logger.info("Pokreni s --dry-run za rad bez Alpaca konekcije.")
        sys.exit(1)

    # Kreiraj agenta
    agent = create_agent(args)

    if args.once:
        run_once(agent)
    else:
        run_scheduled(agent, args.interval)


if __name__ == "__main__":
    main()
