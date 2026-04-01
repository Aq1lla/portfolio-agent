"""
Portfolio Agent — Testovi za Fazu 3
Pokrivaju: Execution Engine, Reporter, i Agent (end-to-end).

Pokretanje:  pytest tests/test_phase3.py -v
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import pytest
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    AllocationConfig,
    DCAConfig,
    DCADistribution,
    Frequency,
    OrderSide,
    OrderStatus,
    OrderType,
    PortfolioSnapshot,
    Position,
    PriceBar,
    RebalancingConfig,
    RiskAction,
    RiskConfig,
    StrategyConfig,
    TradeOrder,
)
from src.db import Database
from src.data_engine import BaseDataProvider, DataEngine
from src.strategy_engine import StrategyEngine
from src.risk_manager import RiskManager
from src.execution_engine import ExecutionEngine
from src.reporter import Reporter
from src.agent import Agent


# ══════════════════════════════════════════════════════════════
# Mock Provider (reusable)
# ══════════════════════════════════════════════════════════════

class MockDataProvider(BaseDataProvider):
    def __init__(self, bars=None, market_open=True, snapshot=None):
        self._bars = bars or []
        self._market_open = market_open
        self._snapshot = snapshot or PortfolioSnapshot(
            timestamp=datetime.now(timezone.utc),
            equity=10000.0, cash=1000.0, buying_power=1000.0,
            positions=[
                Position(symbol="VTI", qty=12.0, avg_entry_price=250.0,
                         current_price=250.0, market_value=3000.0,
                         unrealized_pl=0.0, unrealized_pl_pct=0.0),
                Position(symbol="VXUS", qty=37.5, avg_entry_price=60.0,
                         current_price=60.0, market_value=2250.0,
                         unrealized_pl=0.0, unrealized_pl_pct=0.0),
                Position(symbol="BND", qty=25.0, avg_entry_price=70.0,
                         current_price=70.0, market_value=1750.0,
                         unrealized_pl=0.0, unrealized_pl_pct=0.0),
                Position(symbol="BNDX", qty=16.0, avg_entry_price=50.0,
                         current_price=50.0, market_value=800.0,
                         unrealized_pl=0.0, unrealized_pl_pct=0.0),
                Position(symbol="GLD", qty=1.0, avg_entry_price=200.0,
                         current_price=200.0, market_value=200.0,
                         unrealized_pl=0.0, unrealized_pl_pct=0.0),
            ],
        )

    def get_historical_bars(self, symbol, start, end, timeframe="1Day"):
        return [b for b in self._bars if b.symbol == symbol]

    def get_latest_bar(self, symbol):
        bars = [b for b in self._bars if b.symbol == symbol]
        return bars[-1] if bars else None

    def get_latest_bars_multi(self, symbols):
        result = {}
        for s in symbols:
            bar = self.get_latest_bar(s)
            if bar:
                result[s] = bar
        return result

    def get_portfolio_snapshot(self):
        return self._snapshot

    def is_market_open(self):
        return self._market_open


# ══════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Database(db_path=os.path.join(tmpdir, "test.db"))


@pytest.fixture
def config():
    return StrategyConfig(
        allocation=AllocationConfig(targets={
            "VTI": 0.35, "VXUS": 0.25, "BND": 0.20,
            "BNDX": 0.10, "GLD": 0.05,
        }),
        rebalancing=RebalancingConfig(threshold_pct=5.0),
        dca=DCAConfig(enabled=True, amount=200.0),
        risk=RiskConfig(
            max_daily_loss_pct=3.0, max_drawdown_pct=15.0,
            max_position_pct=40.0, circuit_breaker_vix=35.0,
            min_cash_reserve_pct=5.0, max_orders_per_day=10,
        ),
    )


@pytest.fixture
def prices():
    return {
        "VTI": 250.0, "VXUS": 60.0, "BND": 70.0,
        "BNDX": 50.0, "GLD": 200.0, "SPY": 500.0,
    }


@pytest.fixture
def snapshot():
    return PortfolioSnapshot(
        timestamp=datetime.now(timezone.utc),
        equity=10000.0, cash=1000.0, buying_power=1000.0,
        positions=[
            Position(symbol="VTI", qty=12.0, avg_entry_price=250.0,
                     current_price=250.0, market_value=3000.0,
                     unrealized_pl=0.0, unrealized_pl_pct=0.0),
            Position(symbol="VXUS", qty=37.5, avg_entry_price=60.0,
                     current_price=60.0, market_value=2250.0,
                     unrealized_pl=0.0, unrealized_pl_pct=0.0),
            Position(symbol="BND", qty=25.0, avg_entry_price=70.0,
                     current_price=70.0, market_value=1750.0,
                     unrealized_pl=0.0, unrealized_pl_pct=0.0),
            Position(symbol="BNDX", qty=16.0, avg_entry_price=50.0,
                     current_price=50.0, market_value=800.0,
                     unrealized_pl=0.0, unrealized_pl_pct=0.0),
            Position(symbol="GLD", qty=1.0, avg_entry_price=200.0,
                     current_price=200.0, market_value=200.0,
                     unrealized_pl=0.0, unrealized_pl_pct=0.0),
        ],
    )


@pytest.fixture
def risk_mgr(config, tmp_db):
    return RiskManager(config, tmp_db)


@pytest.fixture
def exec_engine(config, tmp_db, risk_mgr):
    """Execution Engine u dry-run modu."""
    return ExecutionEngine(config, tmp_db, risk_mgr, broker_client=None)


@pytest.fixture
def reporter(config, tmp_db):
    return Reporter(config, tmp_db)


# ══════════════════════════════════════════════════════════════
# TEST: Execution Engine — Dry Run
# ══════════════════════════════════════════════════════════════

class TestExecutionDryRun:

    def test_dry_run_mode(self, exec_engine):
        """Engine je u dry-run modu bez broker klijenta."""
        assert exec_engine.dry_run is True

    def test_execute_single_dry_run(self, exec_engine, snapshot, prices):
        """Pojedinačni nalog se izvršava u dry-run modu."""
        order = TradeOrder(
            id="exec-001", symbol="VTI", side=OrderSide.SELL,
            qty=1.0, order_type=OrderType.LIMIT, limit_price=250.0,
            reason="test",
        )
        result = exec_engine.execute_single(order, snapshot, prices)
        assert result is not None
        assert result.status == OrderStatus.FILLED
        assert result.filled_price == 250.0
        assert result.filled_at is not None

    def test_execute_batch_dry_run(self, exec_engine, snapshot, prices):
        """Batch naloga se izvršava u dry-run modu."""
        orders = [
            TradeOrder(
                id=f"batch-{i}", symbol="VTI", side=OrderSide.SELL,
                qty=0.5, order_type=OrderType.LIMIT, limit_price=250.0,
                reason="test batch",
            )
            for i in range(3)
        ]
        results = exec_engine.execute_orders(orders, snapshot, prices)
        assert len(results) == 3
        for r in results:
            assert r.status == OrderStatus.FILLED

    def test_idempotency(self, exec_engine, snapshot, prices):
        """Dupli nalog se preskače."""
        order = TradeOrder(
            id="idem-001", symbol="VTI", side=OrderSide.SELL,
            qty=1.0, order_type=OrderType.LIMIT, limit_price=250.0,
            reason="test",
        )
        r1 = exec_engine.execute_single(order, snapshot, prices)
        r2 = exec_engine.execute_single(order, snapshot, prices)

        assert r1 is not None
        assert r2 is None  # Dupli se preskače

    def test_reset_idempotency(self, exec_engine, snapshot, prices):
        """Reset idempotentnosti dozvoljava ponovni ID."""
        order = TradeOrder(
            id="reset-001", symbol="VTI", side=OrderSide.SELL,
            qty=1.0, order_type=OrderType.LIMIT, limit_price=250.0,
            reason="test",
        )
        exec_engine.execute_single(order, snapshot, prices)
        exec_engine.reset_idempotency()

        # Sada isti ID može proći
        r2 = exec_engine.execute_single(order, snapshot, prices)
        assert r2 is not None

    def test_rejected_order_saved(self, exec_engine, snapshot, prices):
        """Odbijeni nalog se sprema u bazu."""
        order = TradeOrder(
            id="reject-001", symbol="AAPL", side=OrderSide.BUY,
            qty=1.0, order_type=OrderType.LIMIT, limit_price=200.0,
            reason="test — treba biti odbijen",
        )
        results = exec_engine.execute_orders([order], snapshot, prices)
        assert len(results) == 0  # Odbijen, nije izvršen

    def test_orders_saved_to_db(self, exec_engine, tmp_db, snapshot, prices):
        """Izvršeni nalozi se spremaju u bazu."""
        order = TradeOrder(
            id="db-001", symbol="VTI", side=OrderSide.SELL,
            qty=1.0, order_type=OrderType.LIMIT, limit_price=250.0,
            reason="test DB",
        )
        exec_engine.execute_single(order, snapshot, prices)
        assert tmp_db.count_orders_today() >= 1

    def test_risk_rejected_not_executed(self, exec_engine, snapshot, prices):
        """Nalog koji ne prođe risk check se ne izvršava."""
        # Kupnja koja bi probila max position
        order = TradeOrder(
            id="risk-001", symbol="VTI", side=OrderSide.BUY,
            qty=20.0, order_type=OrderType.LIMIT, limit_price=250.0,
            reason="test — previsoka pozicija",
        )
        results = exec_engine.execute_orders([order], snapshot, prices)
        assert len(results) == 0

    def test_halted_rejects_all(self, exec_engine, risk_mgr, snapshot, prices):
        """Halted sustav odbija sve naloge."""
        risk_mgr.halt("Test halt")
        order = TradeOrder(
            id="halt-001", symbol="VTI", side=OrderSide.SELL,
            qty=1.0, order_type=OrderType.LIMIT, limit_price=250.0,
            reason="test",
        )
        results = exec_engine.execute_orders([order], snapshot, prices)
        assert len(results) == 0


class TestExecutionStatus:

    def test_get_status(self, exec_engine):
        """Status vraća ispravne podatke."""
        status = exec_engine.get_status()
        assert status["dry_run"] is True
        assert status["broker_connected"] is False
        assert status["executed_count"] == 0

    def test_check_order_status_dry_run(self, exec_engine):
        """U dry-run modu status je uvijek FILLED."""
        status = exec_engine.check_order_status("any-id")
        assert status == OrderStatus.FILLED

    def test_cancel_order_dry_run(self, exec_engine, tmp_db):
        """Cancel u dry-run modu ažurira bazu."""
        # Ubaci nalog u bazu
        order = TradeOrder(
            id="cancel-001", symbol="VTI", side=OrderSide.SELL,
            qty=1.0, order_type=OrderType.LIMIT, limit_price=250.0,
            reason="test",
        )
        tmp_db.insert_order(order)

        result = exec_engine.cancel_order("cancel-001")
        assert result is True


# ══════════════════════════════════════════════════════════════
# TEST: Reporter
# ══════════════════════════════════════════════════════════════

class TestReporter:

    def test_daily_summary_format(self, reporter, snapshot):
        """Dnevni sažetak sadrži ključne informacije."""
        risk_status = {"halted": False, "drawdown_pct": 2.0, "max_drawdown_pct": 15.0}
        summary = reporter.generate_daily_summary(snapshot, [], risk_status)

        assert "DNEVNI SAŽETAK" in summary
        assert "$10,000.00" in summary
        assert "POZICIJE" in summary
        assert "VTI" in summary

    def test_daily_summary_with_orders(self, reporter, snapshot):
        """Sažetak uključuje informacije o nalozima."""
        orders = [
            TradeOrder(
                id="s-1", symbol="VTI", side=OrderSide.SELL,
                qty=1.0, order_type=OrderType.LIMIT, limit_price=250.0,
                status=OrderStatus.FILLED, filled_price=250.0,
                reason="rebalance",
            ),
        ]
        risk_status = {"halted": False, "drawdown_pct": 0}
        summary = reporter.generate_daily_summary(snapshot, orders, risk_status)

        assert "1 izvršenih" in summary
        assert "SELL" in summary

    def test_daily_summary_halted(self, reporter, snapshot):
        """Sažetak prikazuje halt status."""
        risk_status = {"halted": True, "halt_reason": "Drawdown", "drawdown_pct": 20}
        summary = reporter.generate_daily_summary(snapshot, [], risk_status)

        assert "ZAUSTAVLJEN" in summary
        assert "Drawdown" in summary

    def test_trade_alert_buy(self, reporter):
        """Trade alert za BUY nalog."""
        order = TradeOrder(
            id="t-1", symbol="VTI", side=OrderSide.BUY,
            qty=2.0, order_type=OrderType.LIMIT, limit_price=250.0,
            status=OrderStatus.FILLED, filled_price=249.50,
            reason="DCA",
        )
        alert = reporter.format_trade_alert(order)
        assert "BUY" in alert
        assert "VTI" in alert
        assert "249.50" in alert

    def test_trade_alert_sell(self, reporter):
        """Trade alert za SELL nalog."""
        order = TradeOrder(
            id="t-2", symbol="VXUS", side=OrderSide.SELL,
            qty=5.0, order_type=OrderType.LIMIT, limit_price=60.0,
            status=OrderStatus.FILLED, filled_price=60.0,
            reason="rebalance",
        )
        alert = reporter.format_trade_alert(order)
        assert "SELL" in alert
        assert "VXUS" in alert

    def test_risk_alert(self, reporter):
        """Risk alert formatiranje."""
        from src.models import RiskCheck, RiskAction
        check = RiskCheck(
            action=RiskAction.REJECT,
            reason="Dnevni gubitak prelazi 3% limita.",
        )
        alert = reporter.format_risk_alert(check)
        assert "RISK ALERT" in alert
        assert "3%" in alert

    def test_strategy_summary(self, reporter):
        """Strategy summary formatiranje."""
        summary_data = {
            "needs_rebalancing": True,
            "rebalance_threshold": 5.0,
            "max_drift_pct": 8.5,
            "max_drift_symbol": "VTI",
            "dca_enabled": True,
            "dca_amount": 200.0,
            "allocation_targets": {"VTI": 35.0, "VXUS": 25.0},
        }
        text = reporter.format_strategy_summary(summary_data)
        assert "POTREBNO" in text
        assert "VTI" in text

    def test_send_notification_console(self, reporter):
        """Notification logira na konzolu."""
        result = reporter.send_notification("Test poruka")
        assert result is True


# ══════════════════════════════════════════════════════════════
# TEST: Agent — End-to-End
# ══════════════════════════════════════════════════════════════

class TestAgent:

    @pytest.fixture
    def agent_setup(self, tmp_path):
        """Kreiraj Agent s mock providerom i privremenim datotekama."""
        # Kreiraj YAML konfiguraciju
        config_yaml = {
            "portfolio": {"name": "Test Agent", "benchmark": "SPY"},
            "allocation": {"VTI": 0.35, "VXUS": 0.25, "BND": 0.20, "BNDX": 0.10, "GLD": 0.05},
            "rebalancing": {"threshold_pct": 5.0},
            "dca": {"enabled": True, "amount": 200.0},
            "risk": {
                "max_daily_loss_pct": 3.0, "max_drawdown_pct": 15.0,
                "max_position_pct": 40.0, "min_cash_reserve_pct": 5.0,
                "max_orders_per_day": 10,
            },
            "data": {"provider": "yahoo", "store_path": str(tmp_path / "test.db").replace("\\", "/")},
            "broker": {"paper_trading": True},
        }
        config_path = tmp_path / "strategy.yaml"
        config_path.write_text(yaml.dump(config_yaml))

        # Mock barovi za inicijalizaciju
        base = datetime(2024, 6, 1, tzinfo=timezone.utc)
        bars = []
        for symbol in ["VTI", "VXUS", "BND", "BNDX", "GLD", "SPY"]:
            for i in range(10):
                bars.append(PriceBar(
                    symbol=symbol,
                    timestamp=base + timedelta(days=i),
                    open=100.0, high=105.0, low=98.0,
                    close=102.0, volume=1000000,
                ))

        provider = MockDataProvider(bars=bars, market_open=True)

        agent = Agent.from_config(
            config_path=str(config_path),
            provider=provider,
        )
        return agent

    def test_agent_from_config(self, agent_setup):
        """Agent se kreira iz konfiguracijskih datoteka."""
        agent = agent_setup
        assert agent.config.portfolio.name == "Test Agent"
        assert not agent._initialized

    def test_agent_initialize(self, agent_setup):
        """Agent se inicijalizira — dohvaća historiju i snapshot."""
        agent = agent_setup
        result = agent.initialize()

        assert result["status"] == "initialized"
        assert result["snapshot"] is True
        assert result["prices_count"] > 0
        assert agent._initialized

    def test_agent_run_cycle(self, agent_setup):
        """Agent izvršava ciklus."""
        agent = agent_setup
        agent.initialize()

        result = agent.run_cycle()
        assert result["cycle"] == 1
        assert result["market_open"] is True
        assert "errors" in result
        assert len(result["errors"]) == 0

    def test_agent_multiple_cycles(self, agent_setup):
        """Agent može izvršiti više ciklusa."""
        agent = agent_setup
        agent.initialize()

        for i in range(3):
            result = agent.run_cycle()
            assert result["cycle"] == i + 1

    def test_agent_status(self, agent_setup):
        """Agent status vraća sve informacije."""
        agent = agent_setup
        agent.initialize()

        status = agent.get_status()
        assert status["initialized"] is True
        assert status["portfolio"] == "Test Agent"
        assert "equity" in status
        assert "data" in status
        assert "risk" in status
        assert "execution" in status
        assert "db" in status

    def test_agent_halt_resume(self, agent_setup):
        """Agent halt/resume funkcionira."""
        agent = agent_setup
        agent.initialize()

        agent.halt("Test halt")
        assert agent.risk.is_halted

        # Ciklus ne bi trebao izvršiti naloge
        result = agent.run_cycle()

        agent.resume()
        assert not agent.risk.is_halted

    def test_agent_daily_summary(self, agent_setup):
        """Agent generira dnevni sažetak."""
        agent = agent_setup
        agent.initialize()

        summary = agent.run_daily_summary()
        assert "DNEVNI SAŽETAK" in summary
        assert "$10,000.00" in summary

    def test_agent_shutdown(self, agent_setup):
        """Agent se sigurno gasi."""
        agent = agent_setup
        agent.initialize()
        agent.shutdown()
        # Ne bi trebao baciti iznimku

    def test_agent_auto_initialize(self, agent_setup):
        """Agent se auto-inicijalizira pri prvom run_cycle."""
        agent = agent_setup
        assert not agent._initialized

        result = agent.run_cycle()
        assert agent._initialized
        assert result["cycle"] == 1

    def test_agent_market_closed(self, tmp_path):
        """Agent ne izvršava naloge kad je tržište zatvoreno."""
        config_yaml = {
            "portfolio": {"name": "Closed Market Test", "benchmark": "SPY"},
            "allocation": {"VTI": 0.90},
            "risk": {"min_cash_reserve_pct": 5.0},
            "data": {"provider": "yahoo", "store_path": str(tmp_path / "closed.db").replace("\\", "/")},
            "broker": {"paper_trading": True},
        }
        config_path = tmp_path / "strategy.yaml"
        config_path.write_text(yaml.dump(config_yaml))

        provider = MockDataProvider(market_open=False)
        agent = Agent.from_config(str(config_path), provider=provider)
        agent.initialize()

        result = agent.run_cycle()
        assert result["market_open"] is False
        assert result["orders_generated"] == 0


# ══════════════════════════════════════════════════════════════
# TEST: Integracija — puni pipeline
# ══════════════════════════════════════════════════════════════

class TestFullPipeline:

    def test_drifted_portfolio_rebalances(self, tmp_path):
        """
        End-to-end: portfelj s driftom → rebalance nalozi →
        risk provjera → dry-run izvršenje → zapis u bazu.
        """
        config_yaml = {
            "portfolio": {"name": "Pipeline Test", "benchmark": "SPY"},
            "allocation": {"VTI": 0.35, "VXUS": 0.25, "BND": 0.20, "BNDX": 0.10, "GLD": 0.05},
            "rebalancing": {"threshold_pct": 5.0},
            "dca": {"enabled": False},  # Isključi DCA za čist test
            "risk": {
                "max_daily_loss_pct": 3.0, "max_drawdown_pct": 15.0,
                "max_position_pct": 50.0, "min_cash_reserve_pct": 3.0,
                "max_orders_per_day": 20,
            },
            "data": {"provider": "yahoo", "store_path": str(tmp_path / "pipe.db").replace("\\", "/")},
            "broker": {"paper_trading": True},
        }
        config_path = tmp_path / "strategy.yaml"
        config_path.write_text(yaml.dump(config_yaml))

        # Drifted snapshot: VTI previsoko, BND prenisko
        drifted = PortfolioSnapshot(
            timestamp=datetime.now(timezone.utc),
            equity=10000.0, cash=1500.0, buying_power=1500.0,
            positions=[
                Position(symbol="VTI", qty=18.0, avg_entry_price=250.0,
                         current_price=250.0, market_value=4500.0,
                         unrealized_pl=0.0, unrealized_pl_pct=0.0),
                Position(symbol="VXUS", qty=15.0, avg_entry_price=60.0,
                         current_price=60.0, market_value=900.0,
                         unrealized_pl=0.0, unrealized_pl_pct=0.0),
                Position(symbol="BND", qty=5.0, avg_entry_price=70.0,
                         current_price=70.0, market_value=350.0,
                         unrealized_pl=0.0, unrealized_pl_pct=0.0),
                Position(symbol="BNDX", qty=26.0, avg_entry_price=50.0,
                         current_price=50.0, market_value=1300.0,
                         unrealized_pl=0.0, unrealized_pl_pct=0.0),
                Position(symbol="GLD", qty=7.25, avg_entry_price=200.0,
                         current_price=200.0, market_value=1450.0,
                         unrealized_pl=0.0, unrealized_pl_pct=0.0),
            ],
        )

        base = datetime(2024, 6, 1, tzinfo=timezone.utc)
        bars = []
        for symbol, price in [("VTI", 250.0), ("VXUS", 60.0), ("BND", 70.0),
                               ("BNDX", 50.0), ("GLD", 200.0), ("SPY", 500.0)]:
            for i in range(5):
                bars.append(PriceBar(
                    symbol=symbol, timestamp=base + timedelta(days=i),
                    open=price, high=price+5, low=price-2,
                    close=price, volume=1000000,
                ))

        provider = MockDataProvider(bars=bars, market_open=True, snapshot=drifted)
        agent = Agent.from_config(str(config_path), provider=provider)
        agent.initialize()

        # Pokreni ciklus
        result = agent.run_cycle()

        assert result["market_open"] is True
        assert result["orders_generated"] > 0
        assert result["orders_executed"] > 0
        assert len(result["errors"]) == 0

        # Provjeri da su nalozi u bazi
        db = agent.db
        orders = db.get_orders_today()
        assert len(orders) > 0

        # Svi izvršeni nalozi trebaju biti FILLED (dry-run)
        filled = [o for o in orders if o.status == OrderStatus.FILLED]
        assert len(filled) > 0

        # Provjeri dnevni sažetak
        summary = agent.run_daily_summary()
        assert "DNEVNI SAŽETAK" in summary
        assert "izvršenih" in summary

        # Provjeri status
        status = agent.get_status()
        assert status["initialized"] is True
        assert status["cycle_count"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
