"""
Portfolio Agent — Modul baze podataka (SQLite)
Upravljanje shemom, upisima i čitanjem podataka.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator, List, Optional

from .models import PortfolioSnapshot, Position, PriceBar, TradeOrder


class Database:
    """SQLite baza podataka za Portfolio Agent."""

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str = "./data/market.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager za konekciju s auto-commit i error handling."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")      # Write-Ahead Logging za performanse
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Kreiraj tablice ako ne postoje."""
        with self._conn() as conn:
            conn.executescript("""
                -- ══════════════════════════════════════
                -- Metapodaci baze
                -- ══════════════════════════════════════
                CREATE TABLE IF NOT EXISTS meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                -- ══════════════════════════════════════
                -- Historijski podaci o cijenama (OHLCV)
                -- ══════════════════════════════════════
                CREATE TABLE IF NOT EXISTS price_bars (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol    TEXT    NOT NULL,
                    timestamp TEXT    NOT NULL,
                    open      REAL    NOT NULL,
                    high      REAL    NOT NULL,
                    low       REAL    NOT NULL,
                    close     REAL    NOT NULL,
                    volume    INTEGER NOT NULL,
                    vwap      REAL,
                    UNIQUE(symbol, timestamp)
                );

                CREATE INDEX IF NOT EXISTS idx_bars_symbol_ts
                    ON price_bars(symbol, timestamp DESC);

                -- ══════════════════════════════════════
                -- Snapshoti portfelja
                -- ══════════════════════════════════════
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     TEXT    NOT NULL,
                    equity        REAL    NOT NULL,
                    cash          REAL    NOT NULL,
                    buying_power  REAL    NOT NULL,
                    day_pl        REAL    NOT NULL DEFAULT 0,
                    day_pl_pct    REAL    NOT NULL DEFAULT 0,
                    total_pl      REAL    NOT NULL DEFAULT 0,
                    total_pl_pct  REAL    NOT NULL DEFAULT 0,
                    positions_json TEXT   NOT NULL DEFAULT '[]'
                );

                CREATE INDEX IF NOT EXISTS idx_snapshots_ts
                    ON portfolio_snapshots(timestamp DESC);

                -- ══════════════════════════════════════
                -- Nalozi za trgovanje
                -- ══════════════════════════════════════
                CREATE TABLE IF NOT EXISTS trade_orders (
                    id            TEXT    PRIMARY KEY,
                    symbol        TEXT    NOT NULL,
                    side          TEXT    NOT NULL,
                    qty           REAL    NOT NULL,
                    order_type    TEXT    NOT NULL DEFAULT 'limit',
                    limit_price   REAL,
                    status        TEXT    NOT NULL DEFAULT 'pending',
                    created_at    TEXT    NOT NULL,
                    filled_at     TEXT,
                    filled_price  REAL,
                    reason        TEXT    NOT NULL DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_orders_status
                    ON trade_orders(status);
                CREATE INDEX IF NOT EXISTS idx_orders_created
                    ON trade_orders(created_at DESC);

                -- ══════════════════════════════════════
                -- Dnevni sažeci (za brze upite)
                -- ══════════════════════════════════════
                CREATE TABLE IF NOT EXISTS daily_summary (
                    date          TEXT    PRIMARY KEY,
                    open_equity   REAL,
                    close_equity  REAL,
                    high_equity   REAL,
                    low_equity    REAL,
                    day_pl        REAL,
                    day_pl_pct    REAL,
                    orders_count  INTEGER DEFAULT 0,
                    rebalanced    INTEGER DEFAULT 0
                );

                -- ══════════════════════════════════════
                -- Log događaja (audit trail)
                -- ══════════════════════════════════════
                CREATE TABLE IF NOT EXISTS event_log (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT    NOT NULL,
                    level     TEXT    NOT NULL,
                    component TEXT    NOT NULL,
                    message   TEXT    NOT NULL,
                    details   TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_log_ts
                    ON event_log(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_log_level
                    ON event_log(level);
            """)

            # Spremi verziju sheme
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("schema_version", str(self.SCHEMA_VERSION))
            )

    # ──────────────────────────────────────────────────────
    # Price Bars
    # ──────────────────────────────────────────────────────

    def insert_price_bars(self, bars: List[PriceBar]) -> int:
        """
        Umetni OHLCV zapise. Preskače duplikate (UPSERT).
        Vraća broj umetnutih zapisa.
        """
        if not bars:
            return 0

        with self._conn() as conn:
            cursor = conn.executemany(
                """
                INSERT OR IGNORE INTO price_bars
                    (symbol, timestamp, open, high, low, close, volume, vwap)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        b.symbol,
                        b.timestamp.isoformat(),
                        b.open, b.high, b.low, b.close,
                        b.volume, b.vwap,
                    )
                    for b in bars
                ]
            )
            return cursor.rowcount

    def get_price_bars(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[PriceBar]:
        """Dohvati OHLCV zapise za simbol."""
        query = "SELECT * FROM price_bars WHERE symbol = ?"
        params: list = [symbol]

        if start:
            query += " AND timestamp >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND timestamp <= ?"
            params.append(end.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            PriceBar(
                symbol=row["symbol"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                vwap=row["vwap"],
            )
            for row in reversed(rows)  # Kronološki redoslijed
        ]

    def get_latest_price(self, symbol: str) -> Optional[PriceBar]:
        """Dohvati zadnji poznati OHLCV zapis za simbol."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM price_bars WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1",
                (symbol,)
            ).fetchone()

        if row is None:
            return None

        return PriceBar(
            symbol=row["symbol"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            vwap=row["vwap"],
        )

    def count_bars(self, symbol: Optional[str] = None) -> int:
        """Prebroji zapise cijena."""
        with self._conn() as conn:
            if symbol:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM price_bars WHERE symbol = ?",
                    (symbol,)
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) as cnt FROM price_bars").fetchone()
            return row["cnt"]

    # ──────────────────────────────────────────────────────
    # Portfolio Snapshots
    # ──────────────────────────────────────────────────────

    def insert_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        """Spremi snapshot portfelja."""
        positions_json = json.dumps(
            [p.model_dump() for p in snapshot.positions],
            default=str
        )

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO portfolio_snapshots
                    (timestamp, equity, cash, buying_power, day_pl, day_pl_pct,
                     total_pl, total_pl_pct, positions_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.timestamp.isoformat(),
                    snapshot.equity, snapshot.cash, snapshot.buying_power,
                    snapshot.day_pl, snapshot.day_pl_pct,
                    snapshot.total_pl, snapshot.total_pl_pct,
                    positions_json,
                )
            )

    def get_latest_snapshot(self) -> Optional[PortfolioSnapshot]:
        """Dohvati zadnji snapshot portfelja."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()

        if row is None:
            return None

        positions_data = json.loads(row["positions_json"])
        positions = [Position(**p) for p in positions_data]

        return PortfolioSnapshot(
            timestamp=datetime.fromisoformat(row["timestamp"]),
            equity=row["equity"],
            cash=row["cash"],
            buying_power=row["buying_power"],
            day_pl=row["day_pl"],
            day_pl_pct=row["day_pl_pct"],
            total_pl=row["total_pl"],
            total_pl_pct=row["total_pl_pct"],
            positions=positions,
        )

    def get_snapshots(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[PortfolioSnapshot]:
        """Dohvati snapshote portfelja."""
        query = "SELECT * FROM portfolio_snapshots WHERE 1=1"
        params: list = []

        if start:
            query += " AND timestamp >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND timestamp <= ?"
            params.append(end.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()

        results = []
        for row in reversed(rows):
            positions_data = json.loads(row["positions_json"])
            positions = [Position(**p) for p in positions_data]
            results.append(PortfolioSnapshot(
                timestamp=datetime.fromisoformat(row["timestamp"]),
                equity=row["equity"],
                cash=row["cash"],
                buying_power=row["buying_power"],
                day_pl=row["day_pl"],
                day_pl_pct=row["day_pl_pct"],
                total_pl=row["total_pl"],
                total_pl_pct=row["total_pl_pct"],
                positions=positions,
            ))
        return results

    def get_peak_equity(self) -> float:
        """Dohvati najveću zabilježenu vrijednost portfelja (za drawdown)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MAX(equity) as peak FROM portfolio_snapshots"
            ).fetchone()
        return row["peak"] or 0.0

    # ──────────────────────────────────────────────────────
    # Trade Orders
    # ──────────────────────────────────────────────────────

    def insert_order(self, order: TradeOrder) -> None:
        """Spremi nalog."""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO trade_orders
                    (id, symbol, side, qty, order_type, limit_price,
                     status, created_at, filled_at, filled_price, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order.id, order.symbol, order.side.value,
                    order.qty, order.order_type.value, order.limit_price,
                    order.status.value, order.created_at.isoformat(),
                    order.filled_at.isoformat() if order.filled_at else None,
                    order.filled_price, order.reason,
                )
            )

    def update_order_status(
        self,
        order_id: str,
        status: str,
        filled_at: Optional[datetime] = None,
        filled_price: Optional[float] = None,
    ) -> None:
        """Ažuriraj status naloga."""
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE trade_orders
                SET status = ?, filled_at = ?, filled_price = ?
                WHERE id = ?
                """,
                (
                    status,
                    filled_at.isoformat() if filled_at else None,
                    filled_price,
                    order_id,
                )
            )

    def get_orders_today(self) -> List[TradeOrder]:
        """Dohvati sve današnje naloge."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trade_orders WHERE created_at LIKE ?",
                (f"{today}%",)
            ).fetchall()

        return [self._row_to_order(row) for row in rows]

    def count_orders_today(self) -> int:
        """Prebroji današnje naloge."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM trade_orders WHERE created_at LIKE ?",
                (f"{today}%",)
            ).fetchone()
        return row["cnt"]

    @staticmethod
    def _row_to_order(row: sqlite3.Row) -> TradeOrder:
        return TradeOrder(
            id=row["id"],
            symbol=row["symbol"],
            side=row["side"],
            qty=row["qty"],
            order_type=row["order_type"],
            limit_price=row["limit_price"],
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
            filled_at=datetime.fromisoformat(row["filled_at"]) if row["filled_at"] else None,
            filled_price=row["filled_price"],
            reason=row["reason"],
        )

    # ──────────────────────────────────────────────────────
    # Event Log
    # ──────────────────────────────────────────────────────

    def log_event(
        self,
        level: str,
        component: str,
        message: str,
        details: Optional[dict] = None,
    ) -> None:
        """Spremi događaj u log tablicu."""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO event_log (timestamp, level, component, message, details)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    level, component, message,
                    json.dumps(details) if details else None,
                )
            )

    def get_recent_events(
        self,
        limit: int = 50,
        level: Optional[str] = None,
        component: Optional[str] = None,
    ) -> list:
        """Dohvati nedavne događaje iz loga."""
        query = "SELECT * FROM event_log WHERE 1=1"
        params: list = []

        if level:
            query += " AND level = ?"
            params.append(level)
        if component:
            query += " AND component = ?"
            params.append(component)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            return conn.execute(query, params).fetchall()

    # ──────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────

    def get_db_stats(self) -> dict:
        """Statistike baze podataka."""
        with self._conn() as conn:
            bars = conn.execute("SELECT COUNT(*) as c FROM price_bars").fetchone()["c"]
            snaps = conn.execute("SELECT COUNT(*) as c FROM portfolio_snapshots").fetchone()["c"]
            orders = conn.execute("SELECT COUNT(*) as c FROM trade_orders").fetchone()["c"]
            events = conn.execute("SELECT COUNT(*) as c FROM event_log").fetchone()["c"]

        return {
            "price_bars": bars,
            "snapshots": snaps,
            "orders": orders,
            "events": events,
            "db_path": str(self.db_path),
            "db_size_kb": self.db_path.stat().st_size / 1024 if self.db_path.exists() else 0,
        }
