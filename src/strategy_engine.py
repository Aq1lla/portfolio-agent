"""
Portfolio Agent — Strategy Engine
Implementira investicijska pravila korisnika: rebalansiranje i DCA.

Ovaj modul generira prijedloge naloga (TradeOrder) na temelju:
  - Trenutnog stanja portfelja (PortfolioSnapshot)
  - Ciljne alokacije (config/strategy.yaml)
  - Pravila rebalansiranja i DCA rasporeda

Strategy Engine NIKADA ne izvršava naloge direktno — samo ih predlaže.
Svaki nalog mora proći Risk Manager validaciju prije izvršenja.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from .models import (
    DCADistribution,
    Frequency,
    OrderSide,
    OrderType,
    PortfolioSnapshot,
    RebalanceMethod,
    StrategyConfig,
    TradeOrder,
)

logger = logging.getLogger(__name__)


class StrategyEngine:
    """
    Srce sustava — generira naloge na temelju korisnikovih pravila.

    Dva glavna mehanizma:
      1. Rebalansiranje: usklađivanje trenutne alokacije s ciljnom
      2. DCA: periodičko ulaganje fiksnog iznosa
    """

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.allocation = config.allocation
        self.rebalancing = config.rebalancing
        self.dca = config.dca
        self.risk = config.risk

    # ══════════════════════════════════════════════════════
    # Analiza portfelja
    # ══════════════════════════════════════════════════════

    def calculate_drift(
        self,
        snapshot: PortfolioSnapshot,
    ) -> Dict[str, Dict[str, float]]:
        """
        Izračunaj odstupanje (drift) svake pozicije od ciljne alokacije.

        Vraća dict:
        {
            "VTI": {
                "target_pct": 35.0,
                "current_pct": 38.2,
                "drift_pct": 3.2,       # Koliko odstupa (apsolutno)
                "drift_amount": 320.0,   # Koliko $ treba kupiti/prodati
                "action": "sell",        # "buy", "sell", ili "hold"
            },
            ...
        }
        """
        if snapshot.equity <= 0:
            logger.warning("Portfelj je prazan (equity=0), nema drifta.")
            return {}

        result = {}
        equity = snapshot.equity

        for symbol, target_pct_raw in self.allocation.targets.items():
            target_pct = target_pct_raw * 100  # 0.35 → 35.0%
            current_pct = snapshot.position_pct(symbol)
            drift_pct = current_pct - target_pct
            drift_amount = (drift_pct / 100) * equity

            if drift_pct > 0.5:
                action = "sell"
            elif drift_pct < -0.5:
                action = "buy"
            else:
                action = "hold"

            result[symbol] = {
                "target_pct": target_pct,
                "current_pct": round(current_pct, 2),
                "drift_pct": round(drift_pct, 2),
                "drift_amount": round(drift_amount, 2),
                "action": action,
            }

        # Dodaj CASH analizu
        cash_target_pct = self.allocation.cash_target * 100
        cash_current_pct = snapshot.cash_pct
        result["CASH"] = {
            "target_pct": cash_target_pct,
            "current_pct": round(cash_current_pct, 2),
            "drift_pct": round(cash_current_pct - cash_target_pct, 2),
            "drift_amount": round(
                ((cash_current_pct - cash_target_pct) / 100) * equity, 2
            ),
            "action": "hold",
        }

        return result

    def needs_rebalancing(
        self,
        snapshot: PortfolioSnapshot,
    ) -> Tuple[bool, List[str]]:
        """
        Provjeri treba li portfelj rebalansiranje.

        Vraća (True/False, [lista simbola koji odstupaju]).
        """
        drift = self.calculate_drift(snapshot)
        threshold = self.rebalancing.threshold_pct
        drifted_symbols = []

        for symbol, info in drift.items():
            if symbol == "CASH":
                continue
            if abs(info["drift_pct"]) > threshold:
                drifted_symbols.append(symbol)

        needs = len(drifted_symbols) > 0
        if needs:
            logger.info(
                "Rebalansiranje potrebno: %s (prag: %.1f%%)",
                ", ".join(drifted_symbols), threshold
            )
        else:
            logger.debug(
                "Rebalansiranje nije potrebno (prag: %.1f%%)", threshold
            )

        return needs, drifted_symbols

    # ══════════════════════════════════════════════════════
    # Generiranje naloga — Rebalansiranje
    # ══════════════════════════════════════════════════════

    def generate_rebalance_orders(
        self,
        snapshot: PortfolioSnapshot,
        prices: Dict[str, float],
    ) -> List[TradeOrder]:
        """
        Generiraj naloge za rebalansiranje portfelja.

        Logika:
        1. Izračunaj drift za svaku poziciju
        2. Generiraj SELL naloge za pozicije iznad cilja
        3. Generiraj BUY naloge za pozicije ispod cilja
        4. Poštuj min_cash_reserve i max_position_pct

        Nalozi se sortiraju: SELL prije BUY (oslobodi cash prije kupnje).
        """
        if snapshot.equity <= 0:
            logger.warning("Nema equity-ja za rebalansiranje.")
            return []

        needs, drifted = self.needs_rebalancing(snapshot)
        if not needs:
            return []

        drift = self.calculate_drift(snapshot)
        equity = snapshot.equity
        orders: List[TradeOrder] = []

        # ── Faza 1: SELL nalozi (pozicije iznad cilja) ──
        sell_orders = []
        for symbol in drifted:
            info = drift[symbol]
            if info["action"] != "sell":
                continue

            price = prices.get(symbol)
            if not price or price <= 0:
                logger.warning("Nema cijene za %s, preskačem.", symbol)
                continue

            # Koliko $ treba prodati da se vrati na cilj
            sell_amount = abs(info["drift_amount"])
            qty = sell_amount / price

            # Zaokruži na 2 decimale (Alpaca podržava fractional shares)
            qty = round(qty, 2)
            if qty < 0.01:
                continue

            # Provjeri da imamo dovoljno za prodati
            pos = snapshot.get_position(symbol)
            if pos and qty > pos.qty:
                qty = pos.qty

            limit_price = round(
                price * (1 - self.risk.limit_offset_pct / 100), 2
            )

            order = TradeOrder(
                id=self._generate_order_id(),
                symbol=symbol,
                side=OrderSide.SELL,
                qty=qty,
                order_type=OrderType.LIMIT,
                limit_price=limit_price,
                reason=f"rebalance: {info['current_pct']:.1f}% → {info['target_pct']:.1f}%",
            )
            sell_orders.append(order)

        # ── Faza 2: BUY nalozi (pozicije ispod cilja) ──
        buy_orders = []

        # Procijeni koliko cash-a će biti dostupno nakon SELL naloga
        estimated_cash = snapshot.cash
        for order in sell_orders:
            estimated_cash += order.estimated_value

        # Zadrži minimalnu cash rezervu
        min_cash = (self.risk.min_cash_reserve_pct / 100) * equity
        available_cash = max(0, estimated_cash - min_cash)

        for symbol in drifted:
            info = drift[symbol]
            if info["action"] != "buy":
                continue

            price = prices.get(symbol)
            if not price or price <= 0:
                logger.warning("Nema cijene za %s, preskačem.", symbol)
                continue

            # Koliko $ treba kupiti da se vrati na cilj
            buy_amount = min(abs(info["drift_amount"]), available_cash)
            if buy_amount < 1.0:
                continue

            qty = buy_amount / price
            qty = round(qty, 2)
            if qty < 0.01:
                continue

            # Provjeri max_position_pct
            current_value = snapshot.position_pct(symbol) / 100 * equity
            new_value = current_value + (qty * price)
            max_allowed = (self.risk.max_position_pct / 100) * equity
            if new_value > max_allowed:
                qty = round((max_allowed - current_value) / price, 2)
                if qty < 0.01:
                    continue

            limit_price = round(
                price * (1 + self.risk.limit_offset_pct / 100), 2
            )

            order = TradeOrder(
                id=self._generate_order_id(),
                symbol=symbol,
                side=OrderSide.BUY,
                qty=qty,
                order_type=OrderType.LIMIT,
                limit_price=limit_price,
                reason=f"rebalance: {info['current_pct']:.1f}% → {info['target_pct']:.1f}%",
            )
            buy_orders.append(order)
            available_cash -= qty * price

        # SELL prije BUY
        orders = sell_orders + buy_orders

        logger.info(
            "Rebalansiranje: generirano %d naloga (%d sell, %d buy)",
            len(orders), len(sell_orders), len(buy_orders)
        )
        return orders

    # ══════════════════════════════════════════════════════
    # Generiranje naloga — DCA
    # ══════════════════════════════════════════════════════

    def generate_dca_orders(
        self,
        snapshot: PortfolioSnapshot,
        prices: Dict[str, float],
    ) -> List[TradeOrder]:
        """
        Generiraj naloge za Dollar-Cost Averaging.

        DCA ulaže fiksni iznos raspodijeljen prema odabranoj metodi:
          - target: prema ciljnoj alokaciji
          - equal: ravnomjerno po svim simbolima
          - underweight: prioritet pozicijama ispod cilja
        """
        if not self.dca.enabled:
            logger.debug("DCA je isključen.")
            return []

        amount = self.dca.amount
        if amount <= 0:
            return []

        # Provjeri da imamo dovoljno cash-a
        min_cash = (self.risk.min_cash_reserve_pct / 100) * snapshot.equity
        available = snapshot.cash - min_cash
        if available < amount:
            # Smanji DCA iznos na dostupni cash
            amount = max(0, available)
            if amount < 1.0:
                logger.info(
                    "Nedovoljno cash-a za DCA (dostupno: $%.2f, potrebno: $%.2f). "
                    "Pokrećem cash restoration.",
                    available, self.dca.amount,
                )
                return self._generate_cash_restoration(snapshot, prices)
            logger.info(
                "DCA iznos smanjen na $%.2f (cash ograničenje).", amount
            )

        # Rasporedi iznos prema metodi
        allocations = self._distribute_dca(amount, snapshot, prices)
        orders: List[TradeOrder] = []

        for symbol, alloc_amount in allocations.items():
            price = prices.get(symbol)
            if not price or price <= 0 or alloc_amount < 1.0:
                continue

            qty = round(alloc_amount / price, 2)
            if qty < 0.01:
                continue

            # Provjeri max_position_pct
            current_value = snapshot.position_pct(symbol) / 100 * snapshot.equity
            new_value = current_value + (qty * price)
            max_allowed = (self.risk.max_position_pct / 100) * snapshot.equity
            if new_value > max_allowed:
                qty = round((max_allowed - current_value) / price, 2)
                if qty < 0.01:
                    continue

            limit_price = round(
                price * (1 + self.risk.limit_offset_pct / 100), 2
            )

            order = TradeOrder(
                id=self._generate_order_id(),
                symbol=symbol,
                side=OrderSide.BUY,
                qty=qty,
                order_type=OrderType.LIMIT,
                limit_price=limit_price,
                reason=f"DCA ${alloc_amount:.2f} ({self.dca.distribute_by.value})",
            )
            orders.append(order)

        logger.info(
            "DCA: generirano %d naloga za ukupno $%.2f",
            len(orders), amount
        )
        return orders

    def _distribute_dca(
        self,
        amount: float,
        snapshot: PortfolioSnapshot,
        prices: Dict[str, float],
    ) -> Dict[str, float]:
        """Rasporedi DCA iznos po simbolima prema odabranoj metodi."""
        symbols = self.allocation.symbols

        if self.dca.distribute_by == DCADistribution.EQUAL:
            # Ravnomjerno
            per_symbol = amount / len(symbols) if symbols else 0
            return {s: per_symbol for s in symbols}

        elif self.dca.distribute_by == DCADistribution.UNDERWEIGHT:
            # Prioritet pozicijama ispod cilja
            drift = self.calculate_drift(snapshot)
            underweight = {}
            for symbol in symbols:
                info = drift.get(symbol, {})
                d = info.get("drift_pct", 0)
                if d < 0:  # Ispod cilja
                    underweight[symbol] = abs(d)

            if not underweight:
                # Nitko nije ispod cilja — fallback na target
                return self._distribute_by_target(amount, symbols)

            total_drift = sum(underweight.values())
            return {
                s: amount * (d / total_drift)
                for s, d in underweight.items()
            }

        else:
            # Default: target — prema ciljnoj alokaciji
            return self._distribute_by_target(amount, symbols)

    def _distribute_by_target(
        self, amount: float, symbols: List[str]
    ) -> Dict[str, float]:
        """Rasporedi iznos prema ciljnoj alokaciji."""
        total_target = sum(
            self.allocation.targets.get(s, 0) for s in symbols
        )
        if total_target <= 0:
            return {}

        return {
            s: amount * (self.allocation.targets.get(s, 0) / total_target)
            for s in symbols
        }

    # ══════════════════════════════════════════════════════
    # Cash restoration
    # ══════════════════════════════════════════════════════

    def _generate_cash_restoration(
        self,
        snapshot: PortfolioSnapshot,
        prices: Dict[str, float],
    ) -> List[TradeOrder]:
        """
        Prodaj minimalnu količinu najprekomjernije pozicije da se financira
        sljedeći DCA ciklus.

        Logika:
          - needed = min_cash_reserve + dca_amount - trenutni_cash
          - prodaje se simbol s najvećim drift_pct (najprecizajniji)
          - qty se ograničava na stvarnu veličinu pozicije
        """
        needed = (
            (self.risk.min_cash_reserve_pct / 100) * snapshot.equity
            + self.dca.amount
            - snapshot.cash
        )
        if needed <= 0:
            return []

        drift = self.calculate_drift(snapshot)

        # Sortiraj po drift_pct opadajuće — najprecizajniji simbol na vrhu
        candidates = sorted(
            [
                (symbol, info)
                for symbol, info in drift.items()
                if symbol != "CASH" and prices.get(symbol, 0) > 0
            ],
            key=lambda x: x[1]["drift_pct"],
            reverse=True,
        )

        if not candidates:
            logger.warning("Cash restoration: nema kandidata za prodaju.")
            return []

        symbol, info = candidates[0]
        price = prices[symbol]

        qty = round(needed / price, 2)
        if qty < 0.01:
            return []

        pos = snapshot.get_position(symbol)
        if pos and qty > pos.qty:
            qty = round(pos.qty, 2)
        if qty < 0.01:
            return []

        limit_price = round(price * (1 - self.risk.limit_offset_pct / 100), 2)

        order = TradeOrder(
            id=self._generate_order_id(),
            symbol=symbol,
            side=OrderSide.SELL,
            qty=qty,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
            reason=(
                f"cash_restore: potrebno ${needed:.0f} "
                f"({symbol} drift={info['drift_pct']:+.1f}%)"
            ),
        )
        logger.info(
            "Cash restoration: prodajem %.2f %s ($%.2f) da financiram DCA ciklus.",
            qty, symbol, qty * price,
        )
        return [order]

    # ══════════════════════════════════════════════════════
    # Utility
    # ══════════════════════════════════════════════════════

    @staticmethod
    def _generate_order_id() -> str:
        """Generiraj jedinstveni ID naloga."""
        return f"pa-{uuid.uuid4().hex[:12]}"

    def get_strategy_summary(
        self, snapshot: PortfolioSnapshot
    ) -> Dict[str, any]:
        """Sažetak stanja strategije."""
        drift = self.calculate_drift(snapshot)
        needs, drifted = self.needs_rebalancing(snapshot)

        max_drift = 0.0
        max_drift_symbol = ""
        for symbol, info in drift.items():
            if symbol == "CASH":
                continue
            if abs(info["drift_pct"]) > abs(max_drift):
                max_drift = info["drift_pct"]
                max_drift_symbol = symbol

        return {
            "equity": snapshot.equity,
            "cash": snapshot.cash,
            "cash_pct": snapshot.cash_pct,
            "positions_count": len(snapshot.positions),
            "needs_rebalancing": needs,
            "drifted_symbols": drifted,
            "max_drift_pct": round(max_drift, 2),
            "max_drift_symbol": max_drift_symbol,
            "dca_enabled": self.dca.enabled,
            "dca_amount": self.dca.amount,
            "rebalance_threshold": self.rebalancing.threshold_pct,
            "allocation_targets": {
                s: round(p * 100, 1)
                for s, p in self.allocation.targets.items()
            },
        }
