"""
Portfolio Agent — Pydantic modeli
Validacija konfiguracije i podatkovne strukture.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ══════════════════════════════════════════════════════════════
# Enumeracije
# ══════════════════════════════════════════════════════════════

class Frequency(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"


class DayOfWeek(str, Enum):
    MONDAY = "monday"
    TUESDAY = "tuesday"
    WEDNESDAY = "wednesday"
    THURSDAY = "thursday"
    FRIDAY = "friday"


class RebalanceMethod(str, Enum):
    THRESHOLD = "threshold"
    CALENDAR = "calendar"
    HYBRID = "hybrid"


class DCADistribution(str, Enum):
    TARGET = "target"
    EQUAL = "equal"
    UNDERWEIGHT = "underweight"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"


class DataProvider(str, Enum):
    ALPACA = "alpaca"
    YAHOO = "yahoo"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class RiskAction(str, Enum):
    """Radnje koje Risk Manager može poduzeti."""
    ALLOW = "allow"
    REJECT = "reject"
    PAUSE_BUYING = "pause_buying"
    HALT_ALL = "halt_all"


# ══════════════════════════════════════════════════════════════
# Konfiguracijski modeli (mapiraju strategy.yaml)
# ══════════════════════════════════════════════════════════════

class PortfolioConfig(BaseModel):
    name: str = "Moj Portfolio"
    base_currency: str = "USD"
    benchmark: str = "SPY"


class AllocationConfig(BaseModel):
    """
    Mapa simbol → ciljni postotak.
    Validira da ukupni zbroj ne prelazi 1.0.
    """
    targets: Dict[str, float] = Field(default_factory=dict)

    @field_validator("targets")
    @classmethod
    def validate_allocation_sum(cls, v: Dict[str, float]) -> Dict[str, float]:
        total = sum(v.values())
        if total > 1.0 + 1e-9:
            raise ValueError(
                f"Zbroj alokacija je {total:.4f}, a mora biti <= 1.0. "
                f"Ostatak do 1.0 automatski se smatra cash rezervom."
            )
        if any(pct < 0 for pct in v.values()):
            raise ValueError("Svi postoci alokacije moraju biti >= 0.")
        return v

    @property
    def cash_target(self) -> float:
        """Automatski izračunaj cash kao ostatak do 1.0."""
        return max(0.0, 1.0 - sum(self.targets.values()))

    @property
    def symbols(self) -> List[str]:
        return list(self.targets.keys())


class RebalancingConfig(BaseModel):
    threshold_pct: float = Field(default=5.0, ge=1.0, le=25.0)
    frequency: Frequency = Frequency.MONTHLY
    method: RebalanceMethod = RebalanceMethod.THRESHOLD
    day_of_month: int = Field(default=1, ge=1, le=28)


class DCAConfig(BaseModel):
    enabled: bool = True
    amount: float = Field(default=200.0, ge=10.0)
    frequency: Frequency = Frequency.WEEKLY
    day: DayOfWeek = DayOfWeek.MONDAY
    distribute_by: DCADistribution = DCADistribution.TARGET


class RiskConfig(BaseModel):
    max_daily_loss_pct: float = Field(default=3.0, ge=0.5, le=10.0)
    max_drawdown_pct: float = Field(default=15.0, ge=5.0, le=50.0)
    max_position_pct: float = Field(default=40.0, ge=10.0, le=80.0)
    circuit_breaker_vix: float = Field(default=35.0, ge=15.0, le=80.0)
    min_cash_reserve_pct: float = Field(default=5.0, ge=0.0, le=30.0)
    max_orders_per_day: int = Field(default=10, ge=1, le=100)
    order_type: OrderType = OrderType.LIMIT
    limit_offset_pct: float = Field(default=0.1, ge=0.01, le=2.0)


class TelegramConfig(BaseModel):
    enabled: bool = False
    chat_id: str = ""
    bot_token: str = ""


class NotificationsConfig(BaseModel):
    telegram: TelegramConfig = TelegramConfig()
    daily_summary: bool = True
    trade_alerts: bool = True
    risk_alerts: bool = True
    weekly_report: bool = True


class DataConfig(BaseModel):
    provider: DataProvider = DataProvider.ALPACA
    history_days: int = Field(default=365, ge=30, le=3650)
    snapshot_interval_sec: int = Field(default=60, ge=10, le=3600)
    store_path: str = "./data/market.db"


class BrokerConfig(BaseModel):
    provider: str = "alpaca"
    paper_trading: bool = True
    base_url: str = "https://paper-api.alpaca.markets"


class StrategyConfig(BaseModel):
    """Glavni konfiguracijski model — mapira cijeli strategy.yaml."""
    portfolio: PortfolioConfig = PortfolioConfig()
    allocation: AllocationConfig = AllocationConfig()
    rebalancing: RebalancingConfig = RebalancingConfig()
    dca: DCAConfig = DCAConfig()
    risk: RiskConfig = RiskConfig()
    notifications: NotificationsConfig = NotificationsConfig()
    data: DataConfig = DataConfig()
    broker: BrokerConfig = BrokerConfig()

    @model_validator(mode="after")
    def validate_cash_reserve(self) -> "StrategyConfig":
        """Provjeri da je cash alokacija >= min_cash_reserve_pct."""
        cash = self.allocation.cash_target * 100
        min_cash = self.risk.min_cash_reserve_pct
        if cash < min_cash:
            raise ValueError(
                f"Cash alokacija ({cash:.1f}%) je manja od minimalne "
                f"cash rezerve ({min_cash:.1f}%). Smanji alokaciju ili "
                f"smanji min_cash_reserve_pct."
            )
        return self


# ══════════════════════════════════════════════════════════════
# Podatkovni modeli (runtime)
# ══════════════════════════════════════════════════════════════

class PriceBar(BaseModel):
    """Jedan OHLCV zapis."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2


class Position(BaseModel):
    """Trenutna pozicija u portfelju."""
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_pl_pct: float
    side: str = "long"

    @property
    def cost_basis(self) -> float:
        return self.qty * self.avg_entry_price


class PortfolioSnapshot(BaseModel):
    """Snapshot portfelja u jednom trenutku."""
    timestamp: datetime
    equity: float                    # Ukupna vrijednost portfelja
    cash: float                      # Slobodan cash
    buying_power: float              # Kupovna moć
    positions: List[Position] = Field(default_factory=list)
    day_pl: float = 0.0             # Dnevni P&L
    day_pl_pct: float = 0.0         # Dnevni P&L (%)
    total_pl: float = 0.0           # Ukupni P&L
    total_pl_pct: float = 0.0       # Ukupni P&L (%)

    @property
    def invested(self) -> float:
        return self.equity - self.cash

    @property
    def cash_pct(self) -> float:
        if self.equity == 0:
            return 100.0
        return (self.cash / self.equity) * 100

    def get_position(self, symbol: str) -> Optional[Position]:
        for pos in self.positions:
            if pos.symbol == symbol:
                return pos
        return None

    def position_pct(self, symbol: str) -> float:
        """Postotak portfelja u danom symbolu."""
        pos = self.get_position(symbol)
        if pos is None or self.equity == 0:
            return 0.0
        return (pos.market_value / self.equity) * 100


class TradeOrder(BaseModel):
    """Nalog za trgovanje."""
    id: Optional[str] = None
    symbol: str
    side: OrderSide
    qty: float = Field(ge=0)
    order_type: OrderType = OrderType.LIMIT
    limit_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    reason: str = ""                 # Zašto je nalog kreiran (rebalance, DCA, ...)

    @property
    def estimated_value(self) -> float:
        price = self.limit_price or 0.0
        return self.qty * price


class RiskCheck(BaseModel):
    """Rezultat provjere Risk Managera."""
    action: RiskAction
    reason: str
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ══════════════════════════════════════════════════════════════
# Helper funkcije za učitavanje konfiguracije
# ══════════════════════════════════════════════════════════════

def load_config(path: str) -> StrategyConfig:
    """Učitaj i validiraj strategy.yaml."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Pretvoriti allocation mapu u AllocationConfig objekt
    # Podržava oba formata: flat {VTI: 0.35} i novi {targets: {VTI: 0.35}}
    allocation_raw = raw.get("allocation", {})
    if "targets" not in allocation_raw:
        raw["allocation"] = {"targets": allocation_raw}

    return StrategyConfig(**raw)
