"""
Portfolio Agent — Testovi za Fazu 4
Pokrivaju: Runner logika, Logging konfiguracija, Agent lifecycle.

Pokretanje:  pytest tests/test_phase4.py -v
"""

import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    AllocationConfig,
    DCAConfig,
    OrderStatus,
    PortfolioSnapshot,
    Position,
    PriceBar,
    RebalancingConfig,
    RiskConfig,
    StrategyConfig,
)
from src.db import Database
from src.data_engine import BaseDataProvider
from src.agent import Agent
from src.logging_config import setup_logging


# ══════════════════════════════════════════════════════════════
# Mock Provider
# ══════════════════════════════════════════════════════════════

class MockProvider(BaseDataProvider):
    def __init__(self, market_open=True):
        self._market_open = market_open
        base = datetime(2024, 6, 1, tzinfo=timezone.utc)
        self._bars = []
        for symbol in ["VTI", "VXUS", "BND", "BNDX", "GLD", "SPY"]:
            for i in range(10):
                self._bars.append(PriceBar(
                    symbol=symbol,
                    timestamp=base + timedelta(days=i),
                    open=100.0, high=105.0, low=98.0,
                    close=102.0, volume=1000000,
                ))

    def get_historical_bars(self, symbol, start, end, timeframe="1Day"):
        return [b for b in self._bars if b.symbol == symbol]

    def get_latest_bar(self, symbol):
        bars = [b for b in self._bars if b.symbol == symbol]
        return bars[-1] if bars else None

    def get_latest_bars_multi(self, symbols):
        return {s: self.get_latest_bar(s) for s in symbols if self.get_latest_bar(s)}

    def get_portfolio_snapshot(self):
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

    def is_market_open(self):
        return self._market_open


# ══════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════

@pytest.fixture
def config_dir(tmp_path):
    """Kreiraj privremeni direktorij s konfiguracijskim datotekama."""
    config_yaml = {
        "portfolio": {"name": "Phase 4 Test", "benchmark": "SPY"},
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

    return tmp_path, str(config_path)


@pytest.fixture
def agent(config_dir):
    """Agent s mock providerom."""
    tmp_path, config_path = config_dir
    provider = MockProvider(market_open=True)
    return Agent.from_config(config_path=config_path, provider=provider)


# ══════════════════════════════════════════════════════════════
# TEST: Logging konfiguracija
# ══════════════════════════════════════════════════════════════

class TestLogging:

    def test_setup_logging(self, tmp_path):
        """Logging se konfigurira bez grešaka."""
        setup_logging(
            level="DEBUG",
            log_dir=str(tmp_path / "logs"),
            log_file="test.log",
        )
        logger = logging.getLogger("test_logging")
        logger.info("Test poruka")

        log_file = tmp_path / "logs" / "test.log"
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test poruka" in content

    def test_log_levels(self, tmp_path):
        """Svi log levelovi rade."""
        setup_logging(level="DEBUG", log_dir=str(tmp_path / "logs"))
        logger = logging.getLogger("test_levels")

        logger.debug("Debug poruka")
        logger.info("Info poruka")
        logger.warning("Warning poruka")
        logger.error("Error poruka")

        log_file = tmp_path / "logs" / "agent.log"
        content = log_file.read_text()
        assert "Debug poruka" in content
        assert "Info poruka" in content
        assert "Warning poruka" in content
        assert "Error poruka" in content

    def test_log_directory_created(self, tmp_path):
        """Log direktorij se automatski kreira."""
        log_dir = tmp_path / "nested" / "log" / "dir"
        setup_logging(log_dir=str(log_dir))
        assert log_dir.exists()


# ══════════════════════════════════════════════════════════════
# TEST: Agent Lifecycle
# ══════════════════════════════════════════════════════════════

class TestAgentLifecycle:

    def test_full_lifecycle(self, agent):
        """Agent prolazi kompletan životni ciklus."""
        # Inicijalizacija
        result = agent.initialize()
        assert result["status"] == "initialized"

        # Više ciklusa
        for i in range(5):
            cycle = agent.run_cycle()
            assert cycle["cycle"] == i + 1
            assert len(cycle["errors"]) == 0

        # Dnevni sažetak
        summary = agent.run_daily_summary()
        assert "DNEVNI SAŽETAK" in summary

        # Status
        status = agent.get_status()
        assert status["cycle_count"] == 5

        # Shutdown
        agent.shutdown()

    def test_new_day_reset(self, agent):
        """run_new_day resetira dnevne limitere."""
        agent.initialize()
        agent.run_cycle()

        # Simuliraj novi dan
        agent.run_new_day()
        # Idempotentnost je resetirana
        assert agent.execution._executed_ids == set()

    def test_halt_prevents_execution(self, agent):
        """Halted agent ne izvršava naloge."""
        agent.initialize()

        # Pokreni normalan ciklus
        result1 = agent.run_cycle()

        # Halt
        agent.halt("Test halt")
        assert agent.risk.is_halted

        # Ciklus radi ali ne izvršava naloge
        result2 = agent.run_cycle()
        assert result2["orders_executed"] == 0

        # Resume
        agent.resume()
        assert not agent.risk.is_halted

    def test_market_closed_no_orders(self, config_dir):
        """Zatvoreno tržište = nema naloga."""
        tmp_path, config_path = config_dir
        provider = MockProvider(market_open=False)
        agent = Agent.from_config(config_path=config_path, provider=provider)
        agent.initialize()

        result = agent.run_cycle()
        assert result["market_open"] is False
        assert result["orders_generated"] == 0
        assert result["orders_executed"] == 0

    def test_error_handling(self, config_dir):
        """Agent nastavlja raditi nakon greške u ciklusu."""
        tmp_path, config_path = config_dir

        class FailingProvider(MockProvider):
            def __init__(self):
                super().__init__(market_open=True)
                self._cycle = 0

            def get_portfolio_snapshot(self):
                self._cycle += 1
                if self._cycle == 2:  # Fail on second call (first run_cycle)
                    raise ConnectionError("Simulirana greška")
                return super().get_portfolio_snapshot()

        provider = FailingProvider()
        agent = Agent.from_config(config_path=config_path, provider=provider)
        agent.initialize()  # _cycle becomes 1, succeeds

        # First cycle has error (_cycle becomes 2, fails)
        result1 = agent.run_cycle()
        assert len(result1["errors"]) > 0

        # Second cycle works (_cycle becomes 3, succeeds)
        result2 = agent.run_cycle()
        assert len(result2["errors"]) == 0

    def test_multiple_days_simulation(self, agent):
        """Simuliraj više tržišnih dana."""
        agent.initialize()

        for day in range(3):
            # Početak dana
            agent.run_new_day()

            # 5 ciklusa po danu
            for _ in range(5):
                agent.run_cycle()

            # Kraj dana
            agent.run_daily_summary()

        assert agent._cycle_count == 15
        status = agent.get_status()
        assert status["cycle_count"] == 15


# ══════════════════════════════════════════════════════════════
# TEST: Deployment Readiness
# ══════════════════════════════════════════════════════════════

class TestDeploymentReadiness:

    def test_config_files_exist(self):
        """Provjeri da ključne datoteke postoje."""
        root = Path(__file__).parent.parent
        assert (root / "runner.py").exists()
        assert (root / "src" / "agent.py").exists()
        assert (root / "src" / "logging_config.py").exists()
        assert (root / "config" / "strategy.yaml").exists()
        assert (root / "config" / "secrets.example.yaml").exists()

    def test_deployment_files_exist(self):
        """Provjeri deployment datoteke."""
        root = Path(__file__).parent.parent
        assert (root / "Dockerfile").exists()
        assert (root / "docker-compose.yaml").exists()
        assert (root / "deploy" / "portfolio-agent.service").exists()
        assert (root / "deploy" / "setup_vps.sh").exists()

    def test_gitignore_protects_secrets(self):
        """Provjeri da .gitignore štiti secrets."""
        root = Path(__file__).parent.parent
        gitignore = (root / ".gitignore").read_text()
        assert "secrets.yaml" in gitignore
        assert "*.db" in gitignore
        assert "*.log" in gitignore

    def test_requirements_complete(self):
        """Provjeri da requirements.txt ima sve potrebno."""
        root = Path(__file__).parent.parent
        req = (root / "requirements.txt").read_text()
        assert "pydantic" in req
        assert "pyyaml" in req
        assert "apscheduler" in req


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
