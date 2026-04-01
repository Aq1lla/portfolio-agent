"""
Portfolio Agent — Reporter
Generira izvještaje i šalje obavijesti.

Podržani kanali:
  - Konzola (uvijek aktivan)
  - Telegram bot (opcionalno)

Vrste izvještaja:
  - Dnevni sažetak: stanje, P&L, izvršeni nalozi
  - Tjedni izvještaj: performanse vs benchmark, alokacija
  - Trade obavijest: svaki izvršeni nalog
  - Risk obavijest: aktivacija risk limita
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from .db import Database
from .models import (
    OrderStatus,
    PortfolioSnapshot,
    RiskCheck,
    StrategyConfig,
    TradeOrder,
)

logger = logging.getLogger(__name__)


class Reporter:
    """Generira izvještaje i šalje obavijesti."""

    def __init__(self, config: StrategyConfig, db: Database):
        self.config = config
        self.db = db
        self.telegram_enabled = config.notifications.telegram.enabled
        self._telegram_bot = None

        if self.telegram_enabled:
            self._init_telegram()

    def _init_telegram(self) -> None:
        """Inicijaliziraj Telegram bot (lazy)."""
        try:
            import telegram
            token = self.config.notifications.telegram.bot_token
            if token:
                self._telegram_bot = telegram.Bot(token=token)
                logger.info("Telegram bot inicijaliziran.")
            else:
                logger.warning("Telegram token nije konfiguriran.")
                self.telegram_enabled = False
        except ImportError:
            logger.warning(
                "python-telegram-bot nije instaliran. "
                "Telegram obavijesti su isključene."
            )
            self.telegram_enabled = False

    # ══════════════════════════════════════════════════════
    # Dnevni sažetak
    # ══════════════════════════════════════════════════════

    def generate_daily_summary(
        self,
        snapshot: PortfolioSnapshot,
        orders_today: List[TradeOrder],
        risk_status: Dict,
    ) -> str:
        """Generiraj dnevni sažetak portfelja."""
        filled = [o for o in orders_today if o.status == OrderStatus.FILLED]
        rejected = [o for o in orders_today if o.status == OrderStatus.REJECTED]

        pl_emoji = "📈" if snapshot.day_pl >= 0 else "📉"
        status_emoji = "🟢" if not risk_status.get("halted") else "🔴"

        lines = [
            f"{status_emoji} DNEVNI SAŽETAK — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
            "",
            f"💰 Equity: ${snapshot.equity:,.2f}",
            f"💵 Cash: ${snapshot.cash:,.2f} ({snapshot.cash_pct:.1f}%)",
            f"{pl_emoji} Dnevni P&L: ${snapshot.day_pl:+,.2f} ({snapshot.day_pl_pct:+.2f}%)",
        ]

        # Pozicije
        if snapshot.positions:
            lines.append("")
            lines.append("📊 POZICIJE:")
            for pos in snapshot.positions:
                pl_sign = "+" if pos.unrealized_pl >= 0 else ""
                lines.append(
                    f"  {pos.symbol}: {pos.qty} × ${pos.current_price:.2f} "
                    f"= ${pos.market_value:,.2f} "
                    f"({pl_sign}{pos.unrealized_pl_pct:.1f}%)"
                )

        # Nalozi
        if filled or rejected:
            lines.append("")
            lines.append(f"📋 NALOZI: {len(filled)} izvršenih, {len(rejected)} odbijenih")
            for order in filled:
                lines.append(
                    f"  ✅ {order.side.value.upper()} {order.qty} {order.symbol} "
                    f"@ ${order.filled_price or order.limit_price:.2f}"
                )
            for order in rejected:
                lines.append(
                    f"  ❌ {order.side.value.upper()} {order.qty} {order.symbol} "
                    f"— {order.reason[:60]}"
                )
        else:
            lines.append("")
            lines.append("📋 Bez naloga danas.")

        # Risk status
        if risk_status.get("halted"):
            lines.append("")
            lines.append(f"🔴 SUSTAV ZAUSTAVLJEN: {risk_status.get('halt_reason', 'N/A')}")

        drawdown = risk_status.get("drawdown_pct", 0)
        if drawdown > 5:
            lines.append(f"⚠️ Drawdown: {drawdown:.1f}% (limit: {risk_status.get('max_drawdown_pct', 15)}%)")

        return "\n".join(lines)

    # ══════════════════════════════════════════════════════
    # Trade obavijest
    # ══════════════════════════════════════════════════════

    def format_trade_alert(self, order: TradeOrder) -> str:
        """Formatiraj obavijest o izvršenom nalogu."""
        emoji = "🟢" if order.side.value == "buy" else "🔴"
        price = order.filled_price or order.limit_price
        value = order.qty * price if price else 0

        return (
            f"{emoji} {order.side.value.upper()} {order.qty} {order.symbol} "
            f"@ ${price:.2f} (${value:,.2f})\n"
            f"Razlog: {order.reason}"
        )

    # ══════════════════════════════════════════════════════
    # Risk obavijest
    # ══════════════════════════════════════════════════════

    def format_risk_alert(self, check: RiskCheck) -> str:
        """Formatiraj obavijest o risk događaju."""
        return f"⚠️ RISK ALERT: {check.reason}"

    # ══════════════════════════════════════════════════════
    # Strategija sažetak
    # ══════════════════════════════════════════════════════

    def format_strategy_summary(
        self, strategy_summary: Dict
    ) -> str:
        """Formatiraj sažetak strategije."""
        lines = [
            "📊 STRATEGIJA:",
            f"  Rebalansiranje: {'POTREBNO' if strategy_summary['needs_rebalancing'] else 'OK'}",
            f"  Prag: {strategy_summary['rebalance_threshold']}%",
            f"  Max drift: {strategy_summary['max_drift_pct']}% ({strategy_summary['max_drift_symbol']})",
            f"  DCA: {'uključen' if strategy_summary['dca_enabled'] else 'isključen'}"
            f" (${strategy_summary['dca_amount']}/ciklus)",
        ]

        if strategy_summary.get("allocation_targets"):
            lines.append("  Ciljna alokacija:")
            for symbol, pct in strategy_summary["allocation_targets"].items():
                lines.append(f"    {symbol}: {pct}%")

        return "\n".join(lines)

    # ══════════════════════════════════════════════════════
    # Slanje obavijesti
    # ══════════════════════════════════════════════════════

    def send_notification(self, message: str) -> bool:
        """
        Pošalji obavijest.
        Uvijek logira na konzolu. Šalje na Telegram ako je omogućen.
        """
        logger.info("NOTIFICATION:\n%s", message)

        if self.telegram_enabled and self._telegram_bot:
            return self._send_telegram(message)

        return True

    def _send_telegram(self, message: str) -> bool:
        """Pošalji poruku na Telegram."""
        try:
            chat_id = self.config.notifications.telegram.chat_id
            if not chat_id:
                logger.warning("Telegram chat_id nije konfiguriran.")
                return False

            self._telegram_bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=None,
            )
            return True
        except Exception as e:
            logger.error("Greška pri slanju Telegram poruke: %s", e)
            return False

    # ══════════════════════════════════════════════════════
    # Koordinirane obavijesti
    # ══════════════════════════════════════════════════════

    def notify_daily_summary(
        self,
        snapshot: PortfolioSnapshot,
        orders_today: List[TradeOrder],
        risk_status: Dict,
    ) -> None:
        """Generiraj i pošalji dnevni sažetak."""
        if not self.config.notifications.daily_summary:
            return
        message = self.generate_daily_summary(snapshot, orders_today, risk_status)
        self.send_notification(message)

    def notify_trade(self, order: TradeOrder) -> None:
        """Pošalji obavijest o izvršenom nalogu."""
        if not self.config.notifications.trade_alerts:
            return
        message = self.format_trade_alert(order)
        self.send_notification(message)

    def notify_risk_event(self, check: RiskCheck) -> None:
        """Pošalji obavijest o risk događaju."""
        if not self.config.notifications.risk_alerts:
            return
        message = self.format_risk_alert(check)
        self.send_notification(message)
