"""
Portfolio Agent — Risk Manager
Sigurnosni sloj koji validira svaki nalog prije izvršenja.

Hijerarhija provjera (svaki nalog prolazi SVE razine):
  1. Pre-trade validacija: nalog sukladan konfiguraciji?
  2. Portfolio-level provjera: dnevni gubitak, drawdown, pozicija
  3. Tržišna provjera: tržište otvoreno, VIX, likvidnost
  4. Izvršna provjera: dnevni limit naloga, idempotentnost

Ako BILO KOJA provjera ne prođe, nalog se ODBIJA.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .db import Database
from .models import (
    OrderSide,
    OrderType,
    PortfolioSnapshot,
    RiskAction,
    RiskCheck,
    StrategyConfig,
    TradeOrder,
)

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Svaki nalog mora proći validate_order() prije izvršenja.
    Risk Manager nikada ne modificira naloge — samo ih odobrava ili odbija.
    """

    def __init__(self, config: StrategyConfig, db: Database):
        self.config = config
        self.risk = config.risk
        self.db = db
        self._halted = False
        self._buying_paused = False
        self._halt_reason = ""

    # ══════════════════════════════════════════════════════
    # Glavno sučelje
    # ══════════════════════════════════════════════════════

    def validate_order(
        self,
        order: TradeOrder,
        snapshot: PortfolioSnapshot,
        prices: Dict[str, float],
        vix: Optional[float] = None,
        market_open: bool = True,
    ) -> RiskCheck:
        """
        Provedi sve razine provjere na nalogu.
        Vraća RiskCheck s action=ALLOW ili action=REJECT i razlogom.
        """
        # ── Razina 0: Globalni halt ──
        if self._halted:
            return self._reject(
                f"Sustav je zaustavljen: {self._halt_reason}",
                {"halt_reason": self._halt_reason},
            )

        # ── Razina 1: Pre-trade validacija ──
        check = self._validate_pre_trade(order)
        if check.action != RiskAction.ALLOW:
            return check

        # ── Razina 2: Portfolio-level provjera ──
        check = self._validate_portfolio(order, snapshot)
        if check.action != RiskAction.ALLOW:
            return check

        # ── Razina 3: Tržišna provjera ──
        check = self._validate_market(order, market_open, vix)
        if check.action != RiskAction.ALLOW:
            return check

        # ── Razina 4: Izvršna provjera ──
        check = self._validate_execution(order)
        if check.action != RiskAction.ALLOW:
            return check

        # Sve provjere prošle
        self._log_event(
            "INFO",
            f"Nalog odobren: {order.side.value} {order.qty} {order.symbol}",
            {"order_id": order.id},
        )
        return RiskCheck(
            action=RiskAction.ALLOW,
            reason="Sve provjere prošle.",
        )

    def validate_orders_batch(
        self,
        orders: List[TradeOrder],
        snapshot: PortfolioSnapshot,
        prices: Dict[str, float],
        vix: Optional[float] = None,
        market_open: bool = True,
    ) -> List[tuple[TradeOrder, RiskCheck]]:
        """
        Validiraj batch naloga. Vraća listu (nalog, rezultat) parova.
        """
        results = []
        for order in orders:
            check = self.validate_order(
                order, snapshot, prices, vix, market_open
            )
            results.append((order, check))
        return results

    def get_approved_orders(
        self,
        orders: List[TradeOrder],
        snapshot: PortfolioSnapshot,
        prices: Dict[str, float],
        vix: Optional[float] = None,
        market_open: bool = True,
    ) -> List[TradeOrder]:
        """
        Filtriraj samo odobrene naloge iz batcha.
        """
        results = self.validate_orders_batch(
            orders, snapshot, prices, vix, market_open
        )
        approved = []
        for order, check in results:
            if check.action == RiskAction.ALLOW:
                approved.append(order)
            else:
                logger.warning(
                    "Nalog odbijen: %s %s %s — %s",
                    order.side.value, order.qty, order.symbol, check.reason
                )
        return approved

    # ══════════════════════════════════════════════════════
    # Razina 1: Pre-trade validacija
    # ══════════════════════════════════════════════════════

    def _validate_pre_trade(self, order: TradeOrder) -> RiskCheck:
        """Provjeri da je nalog sukladan osnovnim pravilima."""

        # Količina mora biti pozitivna
        if order.qty <= 0:
            return self._reject(
                f"Količina mora biti pozitivna (dobiveno: {order.qty}).",
                {"qty": order.qty},
            )

        # Simbol mora biti u konfiguraciji ili benchmarku
        allowed_symbols = (
            self.config.allocation.symbols
            + [self.config.portfolio.benchmark]
        )
        if order.symbol not in allowed_symbols:
            return self._reject(
                f"Simbol {order.symbol} nije u konfiguraciji portfelja.",
                {"symbol": order.symbol, "allowed": allowed_symbols},
            )

        # Order type mora biti LIMIT (nikad market)
        if self.risk.order_type == OrderType.LIMIT and order.order_type != OrderType.LIMIT:
            return self._reject(
                "Samo LIMIT nalozi su dozvoljeni (konfiguracija).",
                {"order_type": order.order_type.value},
            )

        # Limit price mora postojati za LIMIT nalog
        if order.order_type == OrderType.LIMIT and (
            order.limit_price is None or order.limit_price <= 0
        ):
            return self._reject(
                "LIMIT nalog zahtijeva pozitivnu limit cijenu.",
                {"limit_price": order.limit_price},
            )

        return self._allow()

    # ══════════════════════════════════════════════════════
    # Razina 2: Portfolio-level provjera
    # ══════════════════════════════════════════════════════

    def _validate_portfolio(
        self,
        order: TradeOrder,
        snapshot: PortfolioSnapshot,
    ) -> RiskCheck:
        """Provjeri nalog u kontekstu portfelja."""

        equity = snapshot.equity
        if equity <= 0:
            return self._reject(
                "Portfelj je prazan (equity=0).",
                {"equity": equity},
            )

        # ── Provjera dnevnog gubitka ──
        max_daily_loss = (self.risk.max_daily_loss_pct / 100) * equity
        if snapshot.day_pl < -max_daily_loss:
            self._halt(
                f"Dnevni gubitak ({snapshot.day_pl:.2f}) prelazi limit "
                f"(-{max_daily_loss:.2f})"
            )
            return self._reject(
                f"Dnevni gubitak prelazi {self.risk.max_daily_loss_pct}% limita. "
                f"Sustav pauziran do sutra.",
                {
                    "day_pl": snapshot.day_pl,
                    "max_daily_loss": max_daily_loss,
                    "max_daily_loss_pct": self.risk.max_daily_loss_pct,
                },
            )

        # ── Provjera max drawdown-a ──
        peak_equity = self.db.get_peak_equity()
        if peak_equity > 0:
            drawdown_pct = ((peak_equity - equity) / peak_equity) * 100
            if drawdown_pct > self.risk.max_drawdown_pct:
                self._halt(
                    f"Drawdown ({drawdown_pct:.1f}%) prelazi limit "
                    f"({self.risk.max_drawdown_pct}%)"
                )
                return self._reject(
                    f"Drawdown ({drawdown_pct:.1f}%) prelazi "
                    f"{self.risk.max_drawdown_pct}% limita. "
                    f"Sustav zaustavljen — potrebna ručna intervencija.",
                    {
                        "drawdown_pct": round(drawdown_pct, 2),
                        "peak_equity": peak_equity,
                        "current_equity": equity,
                        "max_drawdown_pct": self.risk.max_drawdown_pct,
                    },
                )

        # ── Provjera veličine pozicije (samo za BUY) ──
        if order.side == OrderSide.BUY:
            current_value = 0.0
            pos = snapshot.get_position(order.symbol)
            if pos:
                current_value = pos.market_value

            new_value = current_value + order.estimated_value
            max_position = (self.risk.max_position_pct / 100) * equity

            if new_value > max_position:
                return self._reject(
                    f"Pozicija {order.symbol} bi nakon naloga bila "
                    f"${new_value:,.2f} ({new_value / equity * 100:.1f}%), "
                    f"a max je {self.risk.max_position_pct}%.",
                    {
                        "symbol": order.symbol,
                        "current_value": current_value,
                        "order_value": order.estimated_value,
                        "new_value": new_value,
                        "max_position": max_position,
                        "max_position_pct": self.risk.max_position_pct,
                    },
                )

            # ── Provjera cash rezerve (samo za BUY) ──
            min_cash = (self.risk.min_cash_reserve_pct / 100) * equity
            remaining_cash = snapshot.cash - order.estimated_value
            if remaining_cash < min_cash:
                return self._reject(
                    f"Nalog bi smanjio cash na ${remaining_cash:,.2f}, "
                    f"ispod minimuma ${min_cash:,.2f} "
                    f"({self.risk.min_cash_reserve_pct}%).",
                    {
                        "current_cash": snapshot.cash,
                        "order_value": order.estimated_value,
                        "remaining_cash": remaining_cash,
                        "min_cash": min_cash,
                    },
                )

        # ── Provjera da imamo dovoljno za prodaju ──
        if order.side == OrderSide.SELL:
            pos = snapshot.get_position(order.symbol)
            if pos is None:
                return self._reject(
                    f"Nema pozicije u {order.symbol} za prodaju.",
                    {"symbol": order.symbol},
                )
            if order.qty > pos.qty:
                return self._reject(
                    f"Pokušaj prodaje {order.qty} {order.symbol}, "
                    f"ali imamo samo {pos.qty}.",
                    {
                        "symbol": order.symbol,
                        "requested_qty": order.qty,
                        "available_qty": pos.qty,
                    },
                )

        return self._allow()

    # ══════════════════════════════════════════════════════
    # Razina 3: Tržišna provjera
    # ══════════════════════════════════════════════════════

    def _validate_market(
        self,
        order: TradeOrder,
        market_open: bool,
        vix: Optional[float] = None,
    ) -> RiskCheck:
        """Provjeri tržišne uvjete."""

        # ── Tržište mora biti otvoreno ──
        if not market_open:
            return self._reject(
                "Tržište je zatvoreno.",
                {"market_open": False},
            )

        # ── VIX circuit breaker (samo za BUY) ──
        if vix is not None and order.side == OrderSide.BUY:
            if vix > self.risk.circuit_breaker_vix:
                self._buying_paused = True
                return self._reject(
                    f"VIX ({vix:.1f}) iznad circuit breaker praga "
                    f"({self.risk.circuit_breaker_vix}). "
                    f"Kupnja pauzirana.",
                    {
                        "vix": vix,
                        "circuit_breaker_vix": self.risk.circuit_breaker_vix,
                    },
                )
            else:
                self._buying_paused = False

        # ── Kupnja pauzirana? ──
        if self._buying_paused and order.side == OrderSide.BUY:
            return self._reject(
                "Kupnja je trenutno pauzirana (VIX circuit breaker aktivan).",
                {"buying_paused": True},
            )

        return self._allow()

    # ══════════════════════════════════════════════════════
    # Razina 4: Izvršna provjera
    # ══════════════════════════════════════════════════════

    def _validate_execution(self, order: TradeOrder) -> RiskCheck:
        """Provjeri operativne limite."""

        # ── Dnevni limit naloga ──
        orders_today = self.db.count_orders_today()
        if orders_today >= self.risk.max_orders_per_day:
            return self._reject(
                f"Dosegnut dnevni limit naloga "
                f"({orders_today}/{self.risk.max_orders_per_day}).",
                {
                    "orders_today": orders_today,
                    "max_orders_per_day": self.risk.max_orders_per_day,
                },
            )

        return self._allow()

    # ══════════════════════════════════════════════════════
    # Kontrole sustava
    # ══════════════════════════════════════════════════════

    def halt(self, reason: str) -> None:
        """Ručno zaustavi sustav (external trigger)."""
        self._halt(reason)

    def _halt(self, reason: str) -> None:
        """Interno zaustavljanje — logira i postavlja flag."""
        self._halted = True
        self._halt_reason = reason
        logger.critical("HALT: %s", reason)
        self._log_event("CRITICAL", f"Sustav zaustavljen: {reason}")

    def resume(self) -> None:
        """Ručno pokreni sustav nakon halt-a."""
        self._halted = False
        self._buying_paused = False
        self._halt_reason = ""
        logger.info("Sustav pokrenut (resume).")
        self._log_event("INFO", "Sustav pokrenut nakon halt-a.")

    def pause_buying(self, reason: str = "") -> None:
        """Pauziraj kupnju."""
        self._buying_paused = True
        logger.warning("Kupnja pauzirana: %s", reason)

    def resume_buying(self) -> None:
        """Nastavi kupnju."""
        self._buying_paused = False
        logger.info("Kupnja nastavljena.")

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def is_buying_paused(self) -> bool:
        return self._buying_paused

    # ══════════════════════════════════════════════════════
    # Status i izvještavanje
    # ══════════════════════════════════════════════════════

    def get_risk_status(
        self, snapshot: PortfolioSnapshot
    ) -> Dict[str, any]:
        """Trenutni status risk parametara."""
        equity = snapshot.equity
        peak = self.db.get_peak_equity()
        drawdown_pct = (
            ((peak - equity) / peak * 100) if peak > 0 else 0.0
        )
        orders_today = self.db.count_orders_today()

        return {
            "halted": self._halted,
            "halt_reason": self._halt_reason,
            "buying_paused": self._buying_paused,
            "equity": equity,
            "peak_equity": peak,
            "drawdown_pct": round(drawdown_pct, 2),
            "max_drawdown_pct": self.risk.max_drawdown_pct,
            "day_pl": snapshot.day_pl,
            "day_pl_pct": snapshot.day_pl_pct,
            "max_daily_loss_pct": self.risk.max_daily_loss_pct,
            "orders_today": orders_today,
            "max_orders_per_day": self.risk.max_orders_per_day,
            "cash": snapshot.cash,
            "cash_pct": snapshot.cash_pct,
            "min_cash_reserve_pct": self.risk.min_cash_reserve_pct,
        }

    # ══════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════

    @staticmethod
    def _allow() -> RiskCheck:
        return RiskCheck(action=RiskAction.ALLOW, reason="OK")

    @staticmethod
    def _reject(reason: str, details: dict = None) -> RiskCheck:
        return RiskCheck(
            action=RiskAction.REJECT,
            reason=reason,
            details=details or {},
        )

    def _log_event(
        self, level: str, message: str, details: dict = None
    ) -> None:
        """Spremi u event log."""
        self.db.log_event(
            level=level,
            component="risk_manager",
            message=message,
            details=details,
        )
