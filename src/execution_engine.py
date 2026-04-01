"""
Portfolio Agent — Execution Engine
Sigurno i pouzdano slanje naloga prema brokeru.

Ključne značajke:
  - Dry-run mod: logira naloge bez stvarnog izvršenja
  - Retry logika s eksponencijalnim backoff-om
  - Idempotentnost: dupli nalozi se automatski odbijaju
  - Samo LIMIT nalozi (zaštita od slippage-a)
  - Praćenje statusa naloga do ispunjenja ili isteka
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from .db import Database
from .models import (
    OrderSide,
    OrderStatus,
    OrderType,
    PortfolioSnapshot,
    RiskAction,
    StrategyConfig,
    TradeOrder,
)
from .risk_manager import RiskManager

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Izvršava naloge prema brokeru nakon Risk Manager validacije.

    Tok izvršenja:
      1. Primi listu naloga od Strategy Engine-a
      2. Svaki nalog prođe Risk Manager validaciju
      3. Odobreni nalozi se šalju brokeru (ili logiraju u dry-run modu)
      4. Status naloga se prati do ispunjenja
      5. Rezultati se spremaju u bazu
    """

    # Konstante za retry logiku
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 1.0   # sekunde
    RETRY_MAX_DELAY = 10.0   # sekunde

    def __init__(
        self,
        config: StrategyConfig,
        db: Database,
        risk_manager: RiskManager,
        broker_client=None,
    ):
        self.config = config
        self.db = db
        self.risk_manager = risk_manager
        self.broker_client = broker_client  # Alpaca TradingClient ili None
        self.dry_run = config.broker.paper_trading and broker_client is None
        self._executed_ids: Set[str] = set()  # Za idempotentnost

        if self.dry_run:
            logger.info("Execution Engine pokrenut u DRY-RUN modu.")
        else:
            logger.info("Execution Engine pokrenut u LIVE modu.")

    # ══════════════════════════════════════════════════════
    # Glavno sučelje
    # ══════════════════════════════════════════════════════

    def execute_orders(
        self,
        orders: List[TradeOrder],
        snapshot: PortfolioSnapshot,
        prices: Dict[str, float],
        vix: Optional[float] = None,
        market_open: bool = True,
    ) -> List[TradeOrder]:
        """
        Izvršava listu naloga. Svaki nalog prolazi risk validaciju.

        Vraća listu izvršenih (ili dry-run logiranih) naloga
        s ažuriranim statusima.
        """
        if not orders:
            return []

        executed = []
        for order in orders:
            # ── Idempotentnost ──
            if order.id in self._executed_ids:
                logger.warning(
                    "Dupli nalog preskočen: %s %s %s (ID: %s)",
                    order.side.value, order.qty, order.symbol, order.id
                )
                continue

            # ── Risk validacija ──
            check = self.risk_manager.validate_order(
                order, snapshot, prices, vix=vix, market_open=market_open
            )
            if check.action != RiskAction.ALLOW:
                order.status = OrderStatus.REJECTED
                order.reason = f"{order.reason} | REJECTED: {check.reason}"
                self.db.insert_order(order)
                self._log("WARNING",
                    f"Nalog odbijen: {order.side.value} {order.qty} "
                    f"{order.symbol} — {check.reason}",
                    {"order_id": order.id}
                )
                continue

            # ── Izvršenje ──
            if self.dry_run:
                result = self._execute_dry_run(order)
            else:
                result = self._execute_live(order)

            if result:
                self._executed_ids.add(order.id)
                executed.append(result)

        logger.info(
            "Izvršenje završeno: %d/%d naloga uspješno.",
            len(executed), len(orders)
        )
        return executed

    def execute_single(
        self,
        order: TradeOrder,
        snapshot: PortfolioSnapshot,
        prices: Dict[str, float],
        vix: Optional[float] = None,
        market_open: bool = True,
    ) -> Optional[TradeOrder]:
        """Izvršava pojedinačni nalog."""
        results = self.execute_orders(
            [order], snapshot, prices, vix=vix, market_open=market_open
        )
        return results[0] if results else None

    # ══════════════════════════════════════════════════════
    # Dry-run izvršenje
    # ══════════════════════════════════════════════════════

    def _execute_dry_run(self, order: TradeOrder) -> TradeOrder:
        """
        Simuliraj izvršenje naloga — logira ali ne šalje brokeru.
        Nalog se smatra odmah ispunjenim po limit cijeni.
        """
        now = datetime.now(timezone.utc)
        order.status = OrderStatus.FILLED
        order.filled_at = now
        order.filled_price = order.limit_price

        self.db.insert_order(order)
        self._log("INFO",
            f"[DRY-RUN] {order.side.value.upper()} {order.qty} {order.symbol} "
            f"@ ${order.filled_price:.2f} "
            f"(ukupno: ${order.estimated_value:,.2f}) — {order.reason}",
            {"order_id": order.id, "dry_run": True}
        )
        return order

    # ══════════════════════════════════════════════════════
    # Live izvršenje (Alpaca)
    # ══════════════════════════════════════════════════════

    def _execute_live(self, order: TradeOrder) -> Optional[TradeOrder]:
        """
        Pošalji nalog Alpaca brokeru s retry logikom.
        """
        if not self.broker_client:
            logger.error("Broker klijent nije inicijaliziran za live trading.")
            order.status = OrderStatus.REJECTED
            order.reason += " | Broker klijent nedostupan"
            self.db.insert_order(order)
            return None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                broker_order = self._submit_to_alpaca(order)
                if broker_order:
                    return broker_order
            except Exception as e:
                delay = min(
                    self.RETRY_BASE_DELAY * (2 ** (attempt - 1)),
                    self.RETRY_MAX_DELAY
                )
                logger.warning(
                    "Pokušaj %d/%d za %s %s %s neuspješan: %s. "
                    "Ponavljam za %.1fs...",
                    attempt, self.MAX_RETRIES,
                    order.side.value, order.qty, order.symbol,
                    str(e), delay
                )
                if attempt < self.MAX_RETRIES:
                    time.sleep(delay)

        # Svi pokušaji neuspješni
        order.status = OrderStatus.REJECTED
        order.reason += f" | Svi pokušaji ({self.MAX_RETRIES}) neuspješni"
        self.db.insert_order(order)
        self._log("ERROR",
            f"Nalog propao nakon {self.MAX_RETRIES} pokušaja: "
            f"{order.side.value} {order.qty} {order.symbol}",
            {"order_id": order.id}
        )
        return None

    def _submit_to_alpaca(self, order: TradeOrder) -> Optional[TradeOrder]:
        """Pošalji nalog Alpaca API-ju."""
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
        )
        from alpaca.trading.enums import OrderSide as AlpacaSide
        from alpaca.trading.enums import TimeInForce

        side = AlpacaSide.BUY if order.side == OrderSide.BUY else AlpacaSide.SELL

        if order.order_type == OrderType.LIMIT:
            request = LimitOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=order.limit_price,
            )
        else:
            request = MarketOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )

        response = self.broker_client.submit_order(request)

        # Ažuriraj nalog s broker response-om
        order.id = str(response.id) if response.id else order.id
        order.status = OrderStatus.SUBMITTED
        self.db.insert_order(order)

        self._log("INFO",
            f"Nalog poslan: {order.side.value.upper()} {order.qty} "
            f"{order.symbol} @ ${order.limit_price:.2f} "
            f"(broker ID: {order.id})",
            {"order_id": order.id}
        )
        return order

    # ══════════════════════════════════════════════════════
    # Praćenje naloga
    # ══════════════════════════════════════════════════════

    def check_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Provjeri status naloga kod brokera."""
        if self.dry_run:
            return OrderStatus.FILLED

        if not self.broker_client:
            return None

        try:
            response = self.broker_client.get_order_by_id(order_id)
            status_map = {
                "new": OrderStatus.SUBMITTED,
                "accepted": OrderStatus.SUBMITTED,
                "partially_filled": OrderStatus.PARTIALLY_FILLED,
                "filled": OrderStatus.FILLED,
                "canceled": OrderStatus.CANCELLED,
                "expired": OrderStatus.EXPIRED,
                "rejected": OrderStatus.REJECTED,
                "pending_new": OrderStatus.PENDING,
            }
            raw_status = str(response.status).lower()
            status = status_map.get(raw_status, OrderStatus.PENDING)

            # Ažuriraj u bazi
            filled_price = (
                float(response.filled_avg_price)
                if response.filled_avg_price else None
            )
            filled_at = response.filled_at

            self.db.update_order_status(
                order_id, status.value,
                filled_at=filled_at,
                filled_price=filled_price,
            )
            return status

        except Exception as e:
            logger.error("Greška pri provjeri statusa naloga %s: %s", order_id, e)
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Otkaži nalog kod brokera."""
        if self.dry_run:
            self.db.update_order_status(order_id, OrderStatus.CANCELLED.value)
            logger.info("[DRY-RUN] Nalog otkazan: %s", order_id)
            return True

        if not self.broker_client:
            return False

        try:
            self.broker_client.cancel_order_by_id(order_id)
            self.db.update_order_status(order_id, OrderStatus.CANCELLED.value)
            logger.info("Nalog otkazan: %s", order_id)
            return True
        except Exception as e:
            logger.error("Greška pri otkazivanju naloga %s: %s", order_id, e)
            return False

    def cancel_all_orders(self) -> int:
        """Otkaži sve otvorene naloge."""
        if self.dry_run:
            logger.info("[DRY-RUN] Svi nalozi otkazani.")
            return 0

        if not self.broker_client:
            return 0

        try:
            responses = self.broker_client.cancel_orders()
            count = len(responses) if responses else 0
            logger.info("Otkazano %d naloga.", count)
            return count
        except Exception as e:
            logger.error("Greška pri otkazivanju svih naloga: %s", e)
            return 0

    # ══════════════════════════════════════════════════════
    # Status i dijagnostika
    # ══════════════════════════════════════════════════════

    def get_status(self) -> Dict[str, any]:
        """Trenutni status Execution Engine-a."""
        return {
            "dry_run": self.dry_run,
            "broker_connected": self.broker_client is not None,
            "executed_count": len(self._executed_ids),
            "orders_today": self.db.count_orders_today(),
        }

    def reset_idempotency(self) -> None:
        """Resetiraj set izvršenih ID-jeva (za novi dan)."""
        self._executed_ids.clear()
        logger.info("Idempotentnost resetirana.")

    # ══════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════

    def _log(self, level: str, message: str, details: dict = None):
        self.db.log_event(
            level=level, component="execution_engine",
            message=message, details=details,
        )
