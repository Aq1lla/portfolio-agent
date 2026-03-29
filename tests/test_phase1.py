"""
Portfolio Agent — Testovi za Fazu 1
Pokrivaju: modele (Pydantic), bazu podataka, i Data Engine.

Pokretanje:  pytest tests/ -v
"""

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# ──────────────────────────────────────────────────────
# Dodaj src/ u path
# ──────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    AllocationConfig,
    DCAConfig,
    Frequency,
    OrderSide,
    OrderStatus,
    OrderType,
    PortfolioSnapshot,
    Position,
    PriceBar,
    RebalancingConfig,
    RiskConfig,
    StrategyConfig,
    TradeOrder,
    load_config,
)
from src.db import Database
from src.data_engine import BaseDataProvider, DataEngine


# ══════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_db():
    """Kreiraj privremenu bazu za svaki test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = Database(db_path=db_path)
        yield db


@pytest.fixture
def sample_config():
    """Primjer validne konfiguracije."""
    return StrategyConfig(
        allocation=AllocationConfig(targets={
            "VTI": 0.35,
            "VXUS": 0.25,
            "BND": 0.20,
            "BNDX": 0.10,
            "GLD": 0.05,
        }),
        risk=RiskConfig(min_cash_reserve_pct=5.0),
    )


@pytest.fixture
def sample_bars():
    """Primjer OHLCV podataka."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [
        PriceBar(
            symbol="VTI",
            timestamp=base + timedelta(days=i),
            open=200.0 + i,
            high=205.0 + i,
            low=198.0 + i,
            close=202.0 + i,
            volume=1000000 + i * 10000,
            vwap=201.5 + i,
        )
        for i in range(30)
    ]


@pytest.fixture
def sample_snapshot():
    """Primjer snapshota portfelja."""
    return PortfolioSnapshot(
        timestamp=datetime.now(timezone.utc),
        equity=10000.0,
        cash=500.0,
        buying_power=500.0,
        positions=[
            Position(
                symbol="VTI",
                qty=20.0,
                avg_entry_price=200.0,
                current_price=210.0,
                market_value=4200.0,
                unrealized_pl=200.0,
                unrealized_pl_pct=5.0,
            ),
            Position(
                symbol="VXUS",
                qty=30.0,
                avg_entry_price=55.0,
                current_price=57.0,
                market_value=1710.0,
                unrealized_pl=60.0,
                unrealized_pl_pct=3.6,
            ),
        ],
        day_pl=50.0,
        day_pl_pct=0.5,
    )


class MockDataProvider(BaseDataProvider):
    """Mock provider za testiranje bez API-ja."""

    def __init__(self, bars: List[PriceBar] = None):
        self._bars = bars or []
        self._market_open = True

    def get_historical_bars(self, symbol, start, end, timeframe="1Day"):
        return [b for b in self._bars if b.symbol == symbol]

    def get_latest_bar(self, symbol):
        symbol_bars = [b for b in self._bars if b.symbol == symbol]
        return symbol_bars[-1] if symbol_bars else None

    def get_latest_bars_multi(self, symbols):
        result = {}
        for s in symbols:
            bar = self.get_latest_bar(s)
            if bar:
                result[s] = bar
        return result

    def get_portfolio_snapshot(self):
        return PortfolioSnapshot(
            timestamp=datetime.now(timezone.utc),
            equity=10000.0,
            cash=500.0,
            buying_power=500.0,
            positions=[],
        )

    def is_market_open(self):
        return self._market_open


# ══════════════════════════════════════════════════════════════
# TEST: Pydantic modeli
# ══════════════════════════════════════════════════════════════

class TestModels:

    def test_allocation_valid(self):
        """Validna alokacija (zbroj <= 1.0)."""
        config = AllocationConfig(targets={"VTI": 0.5, "BND": 0.3})
        assert config.cash_target == pytest.approx(0.2)
        assert config.symbols == ["VTI", "BND"]

    def test_allocation_over_100_pct(self):
        """Alokacija > 100% mora baciti grešku."""
        with pytest.raises(ValueError, match="<= 1.0"):
            AllocationConfig(targets={"VTI": 0.6, "BND": 0.5})

    def test_allocation_negative(self):
        """Negativna alokacija mora baciti grešku."""
        with pytest.raises(ValueError, match=">= 0"):
            AllocationConfig(targets={"VTI": -0.1})

    def test_allocation_cash_auto_calc(self):
        """Cash se automatski računa kao ostatak."""
        config = AllocationConfig(targets={"VTI": 0.95})
        assert config.cash_target == pytest.approx(0.05)

    def test_risk_config_bounds(self):
        """Risk parametri moraju biti unutar granica."""
        # Validno
        risk = RiskConfig(max_daily_loss_pct=5.0, max_drawdown_pct=20.0)
        assert risk.max_daily_loss_pct == 5.0

        # Previsoki daily loss
        with pytest.raises(ValueError):
            RiskConfig(max_daily_loss_pct=15.0)

        # Prenizak drawdown
        with pytest.raises(ValueError):
            RiskConfig(max_drawdown_pct=3.0)

    def test_strategy_config_cash_validation(self):
        """Konfiguracija mora imati dovoljno cash-a za min_cash_reserve."""
        # Ovo treba baciti grešku jer 100% alocirano, 0% cash, a min je 5%
        with pytest.raises(ValueError, match="Cash alokacija"):
            StrategyConfig(
                allocation=AllocationConfig(targets={"VTI": 1.0}),
                risk=RiskConfig(min_cash_reserve_pct=5.0),
            )

    def test_portfolio_snapshot_properties(self, sample_snapshot):
        """Provjera computed propertija na PortfolioSnapshot."""
        snap = sample_snapshot
        assert snap.invested == 9500.0  # 10000 - 500
        assert snap.cash_pct == pytest.approx(5.0)
        assert snap.get_position("VTI").qty == 20.0
        assert snap.get_position("NONEXIST") is None
        assert snap.position_pct("VTI") == pytest.approx(42.0)

    def test_price_bar_mid(self):
        """PriceBar.mid = (high + low) / 2."""
        bar = PriceBar(
            symbol="VTI",
            timestamp=datetime.now(timezone.utc),
            open=100, high=110, low=90, close=105, volume=1000
        )
        assert bar.mid == 100.0

    def test_trade_order_estimated_value(self):
        """TradeOrder estimated value = qty * limit_price."""
        order = TradeOrder(
            symbol="VTI",
            side=OrderSide.BUY,
            qty=10,
            limit_price=200.0,
        )
        assert order.estimated_value == 2000.0

    def test_load_config_from_yaml(self, tmp_path):
        """Učitaj konfiguraciju iz YAML datoteke."""
        yaml_content = """
portfolio:
  name: "Test Portfolio"
  benchmark: SPY
allocation:
  VTI: 0.50
  BND: 0.30
rebalancing:
  threshold_pct: 5.0
  frequency: monthly
dca:
  enabled: true
  amount: 100.0
risk:
  max_daily_loss_pct: 3.0
  min_cash_reserve_pct: 5.0
data:
  provider: yahoo
  store_path: "./test.db"
broker:
  paper_trading: true
"""
        yaml_file = tmp_path / "test_strategy.yaml"
        yaml_file.write_text(yaml_content)

        config = load_config(str(yaml_file))
        assert config.portfolio.name == "Test Portfolio"
        assert config.allocation.targets["VTI"] == 0.50
        assert config.allocation.cash_target == pytest.approx(0.20)
        assert config.dca.amount == 100.0


# ══════════════════════════════════════════════════════════════
# TEST: Baza podataka
# ══════════════════════════════════════════════════════════════

class TestDatabase:

    def test_schema_creation(self, tmp_db):
        """Baza se inicijalizira s ispravnom shemom."""
        stats = tmp_db.get_db_stats()
        assert stats["price_bars"] == 0
        assert stats["snapshots"] == 0
        assert stats["orders"] == 0
        assert stats["events"] == 0

    def test_insert_and_get_price_bars(self, tmp_db, sample_bars):
        """Umetanje i dohvat OHLCV podataka."""
        inserted = tmp_db.insert_price_bars(sample_bars)
        assert inserted == 30

        bars = tmp_db.get_price_bars("VTI", limit=100)
        assert len(bars) == 30
        assert bars[0].symbol == "VTI"
        # Kronološki redoslijed
        assert bars[0].timestamp < bars[-1].timestamp

    def test_price_bars_no_duplicates(self, tmp_db, sample_bars):
        """Dupli zapisi se ignoriraju (UPSERT)."""
        tmp_db.insert_price_bars(sample_bars)
        tmp_db.insert_price_bars(sample_bars)  # Isti podaci
        assert tmp_db.count_bars("VTI") == 30

    def test_get_latest_price(self, tmp_db, sample_bars):
        """Dohvati zadnji poznati bar."""
        tmp_db.insert_price_bars(sample_bars)
        latest = tmp_db.get_latest_price("VTI")
        assert latest is not None
        assert latest.close == 202.0 + 29  # Zadnji bar

    def test_get_latest_price_empty(self, tmp_db):
        """Zadnja cijena za nepostojeći simbol je None."""
        assert tmp_db.get_latest_price("NONEXIST") is None

    def test_price_bars_date_filter(self, tmp_db, sample_bars):
        """Filtriranje barova po datumu."""
        tmp_db.insert_price_bars(sample_bars)

        start = datetime(2024, 1, 10, tzinfo=timezone.utc)
        end = datetime(2024, 1, 20, tzinfo=timezone.utc)
        bars = tmp_db.get_price_bars("VTI", start=start, end=end)

        for bar in bars:
            assert start <= bar.timestamp <= end

    def test_insert_and_get_snapshot(self, tmp_db, sample_snapshot):
        """Umetanje i dohvat snapshota portfelja."""
        tmp_db.insert_snapshot(sample_snapshot)

        latest = tmp_db.get_latest_snapshot()
        assert latest is not None
        assert latest.equity == 10000.0
        assert latest.cash == 500.0
        assert len(latest.positions) == 2
        assert latest.positions[0].symbol == "VTI"

    def test_get_snapshot_empty(self, tmp_db):
        """Snapshot iz prazne baze je None."""
        assert tmp_db.get_latest_snapshot() is None

    def test_peak_equity(self, tmp_db):
        """Peak equity prati maksimum."""
        for equity in [10000, 11000, 10500, 12000, 11500]:
            snap = PortfolioSnapshot(
                timestamp=datetime.now(timezone.utc),
                equity=equity, cash=500, buying_power=500,
            )
            tmp_db.insert_snapshot(snap)

        assert tmp_db.get_peak_equity() == 12000

    def test_insert_and_update_order(self, tmp_db):
        """Umetanje naloga i ažuriranje statusa."""
        order = TradeOrder(
            id="test-order-001",
            symbol="VTI",
            side=OrderSide.BUY,
            qty=10,
            order_type=OrderType.LIMIT,
            limit_price=200.0,
            reason="DCA",
        )
        tmp_db.insert_order(order)

        # Ažuriraj na 'filled'
        fill_time = datetime.now(timezone.utc)
        tmp_db.update_order_status(
            "test-order-001",
            status="filled",
            filled_at=fill_time,
            filled_price=199.50,
        )

        orders = tmp_db.get_orders_today()
        assert len(orders) == 1
        assert orders[0].status == OrderStatus.FILLED
        assert orders[0].filled_price == 199.50

    def test_count_orders_today(self, tmp_db):
        """Brojač dnevnih naloga."""
        for i in range(5):
            order = TradeOrder(
                id=f"order-{i}",
                symbol="VTI",
                side=OrderSide.BUY,
                qty=1,
                reason="test",
            )
            tmp_db.insert_order(order)

        assert tmp_db.count_orders_today() == 5

    def test_event_log(self, tmp_db):
        """Logiranje i čitanje događaja."""
        tmp_db.log_event("INFO", "test", "Testna poruka", {"key": "value"})
        tmp_db.log_event("ERROR", "test", "Greška!")

        events = tmp_db.get_recent_events(limit=10)
        assert len(events) == 2

        errors = tmp_db.get_recent_events(level="ERROR")
        assert len(errors) == 1

    def test_db_stats(self, tmp_db, sample_bars, sample_snapshot):
        """Statistike baze."""
        tmp_db.insert_price_bars(sample_bars)
        tmp_db.insert_snapshot(sample_snapshot)
        tmp_db.log_event("INFO", "test", "Test")

        stats = tmp_db.get_db_stats()
        assert stats["price_bars"] == 30
        assert stats["snapshots"] == 1
        assert stats["events"] == 1
        assert stats["db_size_kb"] > 0


# ══════════════════════════════════════════════════════════════
# TEST: Data Engine
# ══════════════════════════════════════════════════════════════

class TestDataEngine:

    @pytest.fixture
    def engine(self, tmp_db, sample_config, sample_bars):
        """DataEngine s mock providerom."""
        mock_provider = MockDataProvider(bars=sample_bars)
        engine = DataEngine(
            config=sample_config,
            db=tmp_db,
            provider=mock_provider,
        )
        return engine

    def test_initialize_history(self, engine):
        """Inicijalizacija historije dohvaća podatke za sve simbole."""
        results = engine.initialize_history()
        # VTI ima barove u mocku, ostali nemaju
        assert results["VTI"] == 30
        # Ostali simboli nemaju podatke u mocku
        for symbol in ["VXUS", "BND", "BNDX", "GLD"]:
            assert results[symbol] == 0

    def test_initialize_history_skip_existing(self, engine, sample_bars):
        """Preskače simbole koji već imaju podatke."""
        # Prvo umetnemo ručno
        engine.db.insert_price_bars(sample_bars)

        results = engine.initialize_history()
        assert results["VTI"] == 30  # Postojeći, ne dohvaća ponovo

    def test_update_prices(self, engine):
        """Update cijena dohvaća i sprema najnovije barove."""
        bars = engine.update_prices()
        assert "VTI" in bars
        assert bars["VTI"].symbol == "VTI"

        # Provjeri da je spremljeno u bazu
        latest = engine.db.get_latest_price("VTI")
        assert latest is not None

    def test_get_snapshot(self, engine):
        """Dohvat i spremanje snapshota portfelja."""
        snapshot = engine.get_snapshot()
        assert snapshot is not None
        assert snapshot.equity == 10000.0

        # Provjeri da je spremljeno u bazu
        db_snap = engine.db.get_latest_snapshot()
        assert db_snap is not None
        assert db_snap.equity == 10000.0

    def test_is_market_open(self, engine):
        """Provjera statusa tržišta."""
        assert engine.is_market_open() is True

        engine.provider._market_open = False
        assert engine.is_market_open() is False

    def test_get_price(self, engine, sample_bars):
        """Dohvat zadnje cijene iz baze."""
        engine.db.insert_price_bars(sample_bars)
        price = engine.get_price("VTI")
        assert price is not None
        assert price == 202.0 + 29

    def test_get_price_nonexistent(self, engine):
        """Cijena za nepostojeći simbol je None."""
        assert engine.get_price("NONEXIST") is None

    def test_get_all_prices(self, engine, sample_bars):
        """Dohvat svih cijena."""
        engine.db.insert_price_bars(sample_bars)
        prices = engine.get_all_prices()
        assert "VTI" in prices

    def test_health_check(self, engine, sample_bars):
        """Health check vraća ispravne statistike."""
        engine.db.insert_price_bars(sample_bars)
        health = engine.health_check()
        assert health["status"] == "ok"
        assert health["symbols_tracked"] == 6  # 5 ETF + 1 benchmark
        assert health["total_bars"] == 30

    def test_health_check_empty(self, engine):
        """Health check na praznoj bazi."""
        health = engine.health_check()
        assert health["status"] == "empty"
        assert health["total_bars"] == 0

    def test_get_history(self, engine, sample_bars):
        """Dohvat historije iz baze."""
        engine.db.insert_price_bars(sample_bars)
        history = engine.get_history("VTI", days=365 * 3)
        assert len(history) == 30


# ══════════════════════════════════════════════════════════════
# TEST: Integracija Config → DB → Engine
# ══════════════════════════════════════════════════════════════

class TestIntegration:

    def test_full_pipeline(self, tmp_path):
        """End-to-end test: YAML → Config → DB → Engine → Data."""
        # 1. Kreiraj YAML
        yaml_content = """
portfolio:
  name: "Integration Test"
  benchmark: SPY
allocation:
  VTI: 0.60
  BND: 0.30
rebalancing:
  threshold_pct: 5.0
dca:
  enabled: true
  amount: 100.0
risk:
  max_daily_loss_pct: 3.0
  min_cash_reserve_pct: 5.0
data:
  provider: yahoo
  store_path: "{db_path}"
broker:
  paper_trading: true
"""
        db_path = str(tmp_path / "integration.db").replace("\\", "/")
        yaml_file = tmp_path / "strategy.yaml"
        yaml_file.write_text(yaml_content.format(db_path=db_path))

        # 2. Učitaj config
        config = load_config(str(yaml_file))
        assert config.portfolio.name == "Integration Test"
        assert config.allocation.cash_target == pytest.approx(0.10)

        # 3. Kreiraj bazu
        db = Database(db_path=db_path)
        assert db.get_db_stats()["price_bars"] == 0

        # 4. Kreiraj engine s mock providerom
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        mock_bars = [
            PriceBar(
                symbol="VTI",
                timestamp=base + timedelta(days=i),
                open=200 + i, high=205 + i, low=198 + i,
                close=202 + i, volume=1000000,
            )
            for i in range(10)
        ]
        mock_provider = MockDataProvider(bars=mock_bars)
        engine = DataEngine(config=config, db=db, provider=mock_provider)

        # 5. Inicijaliziraj historiju
        results = engine.initialize_history()
        assert results["VTI"] == 10

        # 6. Dohvati snapshot
        snapshot = engine.get_snapshot()
        assert snapshot.equity == 10000.0

        # 7. Health check
        health = engine.health_check()
        assert health["status"] == "ok"
        assert health["total_bars"] == 10
        assert health["total_snapshots"] == 1

        # 8. Provjeri da je sve u bazi
        stats = db.get_db_stats()
        assert stats["price_bars"] == 10
        assert stats["snapshots"] == 1


# ══════════════════════════════════════════════════════════════
# Pokretanje
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
