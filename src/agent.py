"""
Portfolio Agent — Glavni orkestracija modul
Povezuje sve komponente u funkcionalan sustav.

Životni ciklus jednog ciklusa:
  1. Provjeri je li tržište otvoreno
  2. Dohvati najnovije cijene i snapshot portfelja
  3. Provjeri treba li rebalansiranje
  4. Generiraj naloge (rebalance + DCA po rasporedu)
  5. Validiraj naloge kroz Risk Manager
  6. Izvrši odobrene naloge
  7. Pošalji obavijesti
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import yaml

from .data_engine import DataEngine, BaseDataProvider
from .db import Database
from .execution_engine import ExecutionEngine
from .models import (
    OrderStatus,
    PortfolioSnapshot,
    StrategyConfig,
    TradeOrder,
    load_config,
)
from .reporter import Reporter
from .risk_manager import RiskManager
from .strategy_engine import StrategyEngine

logger = logging.getLogger(__name__)


class Agent:
    """
    Glavni agent koji orkestrira sve komponente.

    Korištenje:
        agent = Agent.from_config("config/strategy.yaml", "config/secrets.yaml")
        agent.initialize()
        agent.run_cycle()   # Jedan ciklus — poziva se periodički
    """

    def __init__(
        self,
        config: StrategyConfig,
        db: Database,
        data_engine: DataEngine,
        strategy_engine: StrategyEngine,
        risk_manager: RiskManager,
        execution_engine: ExecutionEngine,
        reporter: Reporter,
    ):
        self.config = config
        self.db = db
        self.data = data_engine
        self.strategy = strategy_engine
        self.risk = risk_manager
        self.execution = execution_engine
        self.reporter = reporter

        self._initialized = False
        self._cycle_count = 0
        self._last_snapshot: Optional[PortfolioSnapshot] = None
        self._last_prices: Dict[str, float] = {}

    # ══════════════════════════════════════════════════════
    # Factory metode
    # ══════════════════════════════════════════════════════

    @classmethod
    def from_config(
        cls,
        config_path: str,
        secrets_path: Optional[str] = None,
        provider: Optional[BaseDataProvider] = None,
        broker_client=None,
    ) -> "Agent":
        """
        Kreiraj Agent iz konfiguracijskih datoteka.

        Args:
            config_path: Putanja do strategy.yaml
            secrets_path: Putanja do secrets.yaml (opcionalno)
            provider: Eksplicitni data provider (za testiranje)
            broker_client: Alpaca TradingClient (opcionalno)
        """
        config = load_config(config_path)

        # Učitaj secrets
        secrets = {}
        if secrets_path:
            try:
                with open(secrets_path, "r", encoding="utf-8") as f:
                    secrets = yaml.safe_load(f) or {}
            except FileNotFoundError:
                logger.warning("Secrets datoteka ne postoji: %s", secrets_path)

        # Inicijaliziraj komponente
        db = Database(db_path=config.data.store_path)
        data_engine = DataEngine(config, db, secrets=secrets, provider=provider)
        strategy_engine = StrategyEngine(config)
        risk_manager = RiskManager(config, db)
        execution_engine = ExecutionEngine(
            config, db, risk_manager,
            broker_client=broker_client,
        )
        reporter = Reporter(config, db)

        return cls(
            config=config,
            db=db,
            data_engine=data_engine,
            strategy_engine=strategy_engine,
            risk_manager=risk_manager,
            execution_engine=execution_engine,
            reporter=reporter,
        )

    # ══════════════════════════════════════════════════════
    # Inicijalizacija
    # ══════════════════════════════════════════════════════

    def initialize(self) -> Dict:
        """
        Inicijaliziraj agent: dohvati historijske podatke i prvi snapshot.
        Poziva se jednom pri prvom pokretanju.
        """
        logger.info("=" * 60)
        logger.info("Portfolio Agent — Inicijalizacija")
        logger.info("Portfolio: %s", self.config.portfolio.name)
        logger.info("Benchmark: %s", self.config.portfolio.benchmark)
        logger.info("Simboli: %s", ", ".join(self.config.allocation.symbols))
        logger.info("Paper trading: %s", self.config.broker.paper_trading)
        logger.info("Dry run: %s", self.execution.dry_run)
        logger.info("=" * 60)

        # Dohvati historijske podatke
        history = self.data.initialize_history()
        logger.info("Historija inicijalizirana: %s", history)

        # Dohvati prvi snapshot
        self._last_snapshot = self.data.get_snapshot()
        self._last_prices = self.data.get_all_prices()

        self._initialized = True

        result = {
            "status": "initialized",
            "history": history,
            "snapshot": self._last_snapshot is not None,
            "prices_count": len(self._last_prices),
            "health": self.data.health_check(),
        }

        self.db.log_event(
            "INFO", "agent", "Agent inicijaliziran", result
        )
        return result

    # ══════════════════════════════════════════════════════
    # Glavni ciklus
    # ══════════════════════════════════════════════════════

    def run_cycle(self) -> Dict:
        """
        Izvršava jedan kompletan ciklus agenta.

        Vraća sažetak rezultata ciklusa.
        """
        if not self._initialized:
            self.initialize()

        self._cycle_count += 1
        cycle_start = datetime.now(timezone.utc)
        result = {
            "cycle": self._cycle_count,
            "timestamp": cycle_start.isoformat(),
            "market_open": False,
            "orders_generated": 0,
            "orders_executed": 0,
            "errors": [],
        }

        try:
            # ── 1. Provjeri je li tržište otvoreno ──
            market_open = self.data.is_market_open()
            result["market_open"] = market_open

            if not market_open:
                logger.debug("Ciklus %d: tržište zatvoreno.", self._cycle_count)
                return result

            # ── 2. Ažuriraj cijene i snapshot ──
            self._last_prices = self.data.update_prices()
            price_map = {
                s: bar.close
                for s, bar in self._last_prices.items()
            } if isinstance(self._last_prices, dict) and self._last_prices else {}

            # Ako update_prices vraća PriceBar objekte
            if price_map and not isinstance(list(self._last_prices.values())[0], (int, float)):
                price_map = {
                    s: bar.close if hasattr(bar, 'close') else bar
                    for s, bar in self._last_prices.items()
                }

            snapshot = self.data.get_snapshot()
            if snapshot:
                self._last_snapshot = snapshot
            elif self._last_snapshot is None:
                result["errors"].append("Nema dostupnog snapshota portfelja")
                return result

            # ── 3. Generiraj naloge ──
            all_orders: List[TradeOrder] = []

            # Rebalansiranje
            rebalance_orders = self.strategy.generate_rebalance_orders(
                self._last_snapshot, price_map
            )
            all_orders.extend(rebalance_orders)

            # DCA (svaki ciklus provjerava, Strategy Engine filtrira po rasporedu)
            dca_orders = self.strategy.generate_dca_orders(
                self._last_snapshot, price_map
            )
            all_orders.extend(dca_orders)

            result["orders_generated"] = len(all_orders)

            if not all_orders:
                logger.debug("Ciklus %d: nema naloga za izvršenje.", self._cycle_count)
                return result

            # ── 4. Izvrši naloge ──
            executed = self.execution.execute_orders(
                orders=all_orders,
                snapshot=self._last_snapshot,
                prices=price_map,
                market_open=market_open,
            )
            result["orders_executed"] = len(executed)

            # ── 5. Pošalji obavijesti za izvršene naloge ──
            for order in executed:
                if order.status == OrderStatus.FILLED:
                    self.reporter.notify_trade(order)

        except Exception as e:
            logger.error("Greška u ciklusu %d: %s", self._cycle_count, e, exc_info=True)
            result["errors"].append(str(e))
            self.db.log_event(
                "ERROR", "agent",
                f"Greška u ciklusu {self._cycle_count}: {e}",
            )

        return result

    # ══════════════════════════════════════════════════════
    # Dnevne operacije
    # ══════════════════════════════════════════════════════

    def run_daily_summary(self) -> str:
        """Generiraj i pošalji dnevni sažetak."""
        if not self._last_snapshot:
            return "Nema dostupnog snapshota."

        orders_today = self.db.get_orders_today()
        risk_status = self.risk.get_risk_status(self._last_snapshot)

        self.reporter.notify_daily_summary(
            self._last_snapshot, orders_today, risk_status
        )

        return self.reporter.generate_daily_summary(
            self._last_snapshot, orders_today, risk_status
        )

    def run_new_day(self) -> None:
        """Resetiraj dnevne limitere. Poziva se na početku svakog dana."""
        self.execution.reset_idempotency()
        if self.risk.is_halted:
            logger.info("Novi dan — provjeravam treba li sustav nastaviti...")
            # Automatski resume samo ako je halt bio zbog dnevnog gubitka
            # Drawdown halt zahtijeva ručnu intervenciju

    # ══════════════════════════════════════════════════════
    # Status i dijagnostika
    # ══════════════════════════════════════════════════════

    def get_status(self) -> Dict:
        """Kompletan status agenta."""
        status = {
            "initialized": self._initialized,
            "cycle_count": self._cycle_count,
            "portfolio": self.config.portfolio.name,
        }

        if self._last_snapshot:
            status["equity"] = self._last_snapshot.equity
            status["cash"] = self._last_snapshot.cash
            status["positions"] = len(self._last_snapshot.positions)
            status["day_pl"] = self._last_snapshot.day_pl

        status["data"] = self.data.health_check()
        status["risk"] = (
            self.risk.get_risk_status(self._last_snapshot)
            if self._last_snapshot
            else {"halted": self.risk.is_halted}
        )
        status["execution"] = self.execution.get_status()
        status["db"] = self.db.get_db_stats()

        return status

    # ══════════════════════════════════════════════════════
    # Kontrole
    # ══════════════════════════════════════════════════════

    def halt(self, reason: str = "Ručni halt") -> None:
        """Zaustavi agenta."""
        self.risk.halt(reason)
        self.execution.cancel_all_orders()
        logger.critical("Agent zaustavljen: %s", reason)

    def resume(self) -> None:
        """Nastavi rad agenta."""
        self.risk.resume()
        logger.info("Agent nastavlja rad.")

    def shutdown(self) -> None:
        """Sigurno gašenje agenta."""
        logger.info("Agent se gasi...")
        self.execution.cancel_all_orders()
        if self._last_snapshot:
            self.run_daily_summary()
        self.db.log_event("INFO", "agent", "Agent ugašen.")
        logger.info("Agent ugašen.")
