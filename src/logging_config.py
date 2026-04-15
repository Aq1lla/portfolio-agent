"""
Portfolio Agent — Logging konfiguracija
Centralizirano logiranje za sve komponente.

Logovi se pišu na:
  - Konzolu (uvijek)
  - Datoteku logs/agent.log (rotacija: 5MB, 5 backup datoteka)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_dir: str = "logs",
    log_file: str = "agent.log",
    max_bytes: int = 5 * 1024 * 1024,   # 5 MB
    backup_count: int = 5,
) -> None:
    """
    Konfiguriraj logging za cijelu aplikaciju.

    Args:
        level: Razina logiranja (DEBUG, INFO, WARNING, ERROR)
        log_dir: Direktorij za log datoteke
        log_file: Ime log datoteke
        max_bytes: Max veličina log datoteke prije rotacije
        backup_count: Broj backup datoteka
    """
    from logging.handlers import RotatingFileHandler

    # Kreiraj log direktorij
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Očisti postojeće handlere (za re-konfiguraciju)
    root_logger.handlers.clear()

    # Format
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fmt_short = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Handler 1: Konzola ──
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(fmt_short)
    root_logger.addHandler(console)

    # ── Handler 2: Datoteka s rotacijom ──
    file_handler = RotatingFileHandler(
        filename=str(log_path / log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)  # Datoteka hvata sve
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)

    # ── Smanji buku od biblioteka ──
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)

    logging.getLogger(__name__).debug(
        "Logging konfiguriran: level=%s, datoteka=%s",
        level, log_path / log_file
    )
