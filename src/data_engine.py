"""
Portfolio Agent — Data Engine
Prikupljanje, normalizacija i pohrana tržišnih podataka.

Podržani provideri:
  - Alpaca Markets API via alpaca-py (primarni)
  - Yahoo Finance (fallback, bez API ključa)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from .db import Database
from .models import (
    DataProvider,
    PortfolioSnapshot,
    Position,
    PriceBar,
    StrategyConfig,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Apstraktni Data Provider
# ══════════════════════════════════════════════════════════════

class BaseDataProvider(ABC):
    """Sučelje za sve data providere."""

    @abstractmethod
    def get_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1Day",
    ) -> List[PriceBar]:
        """Dohvati historijske OHLCV podatke."""
        ...

    @abstractmethod
    def get_latest_bar(self, symbol: str) -> Optional[PriceBar]:
        """Dohvati zadnji OHLCV zapis."""
        ...

    @abstractmethod
    def get_latest_bars_multi(self, symbols: List[str]) -> Dict[str, PriceBar]:
        """Dohvati zadnje OHLCV zapise za više simbola odjednom."""
        ...

    @abstractmethod
    def get_portfolio_snapshot(self) -> Optional[PortfolioSnapshot]:
        """Dohvati snapshot portfelja s brokera."""
        ...

    @abstractmethod
    def is_market_open(self) -> bool:
        """Je li tržište trenutno otvoreno?"""
        ...


# ══════════════════════════════════════════════════════════════
# Alpaca Data Provider (alpaca-py)
# ══════════════════════════════════════════════════════════════

class AlpacaDataProvider(BaseDataProvider):
    """
    Alpaca Markets provider koristeći alpaca-py SDK.
    Zahtijeva: pip install alpaca-py
    """

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        try:
            from alpaca.data import StockHistoricalDataClient
            from alpaca.trading import TradingClient
        except ImportError:
            raise ImportError(
                "alpaca-py nije instaliran. "
                "Pokreni: pip install alpaca-py"
            )

        # Data klijent — za tržišne podatke (cijene, barovi)
        self.data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key,
        )

        # Trading klijent — za portfelj, pozicije, naloge, status tržišta
        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )

        mode = "PAPER" if paper else "LIVE"
        logger.info("Alpaca klijent inicijaliziran (mod: %s)", mode)

    def get_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1Day",
    ) -> List[PriceBar]:
        """Dohvati historijske OHLCV podatke putem Alpaca API-ja."""
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        # Mapiraj string timeframe na TimeFrame objekt
        tf_map = {
            "1Min": TimeFrame.Minute,
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day,
            "1Week": TimeFrame.Week,
            "1Month": TimeFrame.Month,
        }
        tf = tf_map.get(timeframe, TimeFrame.Day)

        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                end=end,
                limit=10000,
            )

            bars_response = self.data_client.get_stock_bars(request)

            result = []
            # alpaca-py vraća dict {symbol: [Bar, ...]}
            symbol_bars = bars_response.get(symbol, [])
            if hasattr(bars_response, 'data'):
                symbol_bars = bars_response.data.get(symbol, [])
            elif isinstance(bars_response, dict):
                symbol_bars = bars_response.get(symbol, [])
            else:
                # BarSet objekt — pristup po simbolu
                try:
                    symbol_bars = bars_response[symbol]
                except (KeyError, TypeError):
                    symbol_bars = []

            for bar in symbol_bars:
                result.append(PriceBar(
                    symbol=symbol,
                    timestamp=bar.timestamp,
                    open=float(bar.open),
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                    volume=int(bar.volume),
                    vwap=float(bar.vwap) if hasattr(bar, 'vwap') and bar.vwap else None,
                ))

            logger.info(
                "Dohvaćeno %d barova za %s (%s → %s)",
                len(result), symbol, start.date(), end.date()
            )
            return result

        except Exception as e:
            logger.error("Greška pri dohvatu barova za %s: %s", symbol, e)
            return []

    def get_latest_bar(self, symbol: str) -> Optional[PriceBar]:
        """Dohvati zadnji OHLCV zapis."""
        from alpaca.data.requests import StockLatestBarRequest

        try:
            request = StockLatestBarRequest(symbol_or_symbols=symbol)
            response = self.data_client.get_stock_latest_bar(request)

            # Response je dict {symbol: Bar} ili objekt s pristupom po ključu
            bar = None
            if isinstance(response, dict):
                bar = response.get(symbol)
            else:
                try:
                    bar = response[symbol]
                except (KeyError, TypeError):
                    bar = response

            if bar is None:
                return None

            return PriceBar(
                symbol=symbol,
                timestamp=bar.timestamp,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=int(bar.volume),
                vwap=float(bar.vwap) if hasattr(bar, 'vwap') and bar.vwap else None,
            )

        except Exception as e:
            logger.error("Greška pri dohvatu zadnjeg bara za %s: %s", symbol, e)
            return None

    def get_latest_bars_multi(self, symbols: List[str]) -> Dict[str, PriceBar]:
        """Dohvati zadnje barove za više simbola odjednom."""
        from alpaca.data.requests import StockLatestBarRequest

        result = {}
        try:
            request = StockLatestBarRequest(symbol_or_symbols=symbols)
            response = self.data_client.get_stock_latest_bar(request)

            # Iteriraj kroz simbole
            for symbol in symbols:
                bar = None
                if isinstance(response, dict):
                    bar = response.get(symbol)
                else:
                    try:
                        bar = response[symbol]
                    except (KeyError, TypeError):
                        continue

                if bar is not None:
                    result[symbol] = PriceBar(
                        symbol=symbol,
                        timestamp=bar.timestamp,
                        open=float(bar.open),
                        high=float(bar.high),
                        low=float(bar.low),
                        close=float(bar.close),
                        volume=int(bar.volume),
                        vwap=float(bar.vwap) if hasattr(bar, 'vwap') and bar.vwap else None,
                    )

        except Exception as e:
            logger.error("Greška pri dohvatu multi-barova: %s", e)

        return result

    def get_portfolio_snapshot(self) -> Optional[PortfolioSnapshot]:
        """Dohvati snapshot portfelja s Alpaca brokera."""
        try:
            account = self.trading_client.get_account()
            raw_positions = self.trading_client.get_all_positions()

            positions = []
            for pos in raw_positions:
                positions.append(Position(
                    symbol=pos.symbol,
                    qty=float(pos.qty),
                    avg_entry_price=float(pos.avg_entry_price),
                    current_price=float(pos.current_price),
                    market_value=float(pos.market_value),
                    unrealized_pl=float(pos.unrealized_pl),
                    unrealized_pl_pct=float(pos.unrealized_plpc) * 100,
                    side=pos.side.value if hasattr(pos.side, 'value') else str(pos.side),
                ))

            equity = float(account.equity)
            last_equity = float(account.last_equity)

            return PortfolioSnapshot(
                timestamp=datetime.now(timezone.utc),
                equity=equity,
                cash=float(account.cash),
                buying_power=float(account.buying_power),
                positions=positions,
                day_pl=equity - last_equity,
                day_pl_pct=(
                    (equity - last_equity) / last_equity * 100
                    if last_equity > 0 else 0.0
                ),
            )

        except Exception as e:
            logger.error("Greška pri dohvatu snapshota portfelja: %s", e)
            return None

    def is_market_open(self) -> bool:
        """Provjeri je li tržište otvoreno."""
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error("Greška pri provjeri statusa tržišta: %s", e)
            return False


# ══════════════════════════════════════════════════════════════
# Yahoo Finance Fallback Provider
# ══════════════════════════════════════════════════════════════

class YahooDataProvider(BaseDataProvider):
    """
    Yahoo Finance fallback provider.
    Zahtijeva: pip install yfinance
    NAPOMENA: Samo za podatke o cijenama, ne podržava trading.
    """

    def __init__(self):
        try:
            import yfinance  # noqa: F401
        except ImportError:
            raise ImportError(
                "yfinance nije instaliran. Pokreni: pip install yfinance"
            )
        logger.info("Yahoo Finance provider inicijaliziran (samo za podatke)")

    def get_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1Day",
    ) -> List[PriceBar]:
        """Dohvati historijske podatke putem Yahoo Finance."""
        import yfinance as yf

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
            )

            result = []
            for idx, row in df.iterrows():
                result.append(PriceBar(
                    symbol=symbol,
                    timestamp=idx.to_pydatetime(),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=int(row["Volume"]),
                ))

            logger.info("Yahoo: dohvaćeno %d barova za %s", len(result), symbol)
            return result

        except Exception as e:
            logger.error("Yahoo: greška za %s: %s", symbol, e)
            return []

    def get_latest_bar(self, symbol: str) -> Optional[PriceBar]:
        """Dohvati zadnji bar putem Yahoo Finance."""
        import yfinance as yf

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1d")
            if df.empty:
                return None

            row = df.iloc[-1]
            return PriceBar(
                symbol=symbol,
                timestamp=df.index[-1].to_pydatetime(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=int(row["Volume"]),
            )
        except Exception as e:
            logger.error("Yahoo: greška za latest %s: %s", symbol, e)
            return None

    def get_latest_bars_multi(self, symbols: List[str]) -> Dict[str, PriceBar]:
        """Dohvati zadnje barove za više simbola."""
        result = {}
        for symbol in symbols:
            bar = self.get_latest_bar(symbol)
            if bar:
                result[symbol] = bar
        return result

    def get_portfolio_snapshot(self) -> Optional[PortfolioSnapshot]:
        """Yahoo ne podržava portfolio — vrati None."""
        logger.warning("Yahoo Finance ne podržava dohvat portfelja.")
        return None

    def is_market_open(self) -> bool:
        """Pojednostavljena provjera (US market hours)."""
        now = datetime.now(timezone.utc)
        # NYSE: 14:30 - 21:00 UTC (9:30 AM - 4:00 PM ET)
        if now.weekday() >= 5:  # Vikend
            return False
        market_open = now.replace(hour=14, minute=30, second=0)
        market_close = now.replace(hour=21, minute=0, second=0)
        return market_open <= now <= market_close


# ══════════════════════════════════════════════════════════════
# Data Engine — Orkestracija
# ══════════════════════════════════════════════════════════════

class DataEngine:
    """
    Glavni podatkovni modul.
    Koordinira dohvat podataka od providera i pohranu u bazu.
    """

    def __init__(
        self,
        config: StrategyConfig,
        db: Database,
        secrets: dict = None,
        provider: Optional[BaseDataProvider] = None,
    ):
        self.config = config
        self.db = db
        self.symbols = config.allocation.symbols + [config.portfolio.benchmark]
        # Ako je provider eksplicitno dan, koristi ga (za testiranje)
        self.provider = provider or self._init_provider(config, secrets or {})

    def _init_provider(
        self, config: StrategyConfig, secrets: dict
    ) -> BaseDataProvider:
        """Inicijaliziraj odgovarajući data provider."""
        if config.data.provider == DataProvider.ALPACA:
            alpaca_secrets = secrets.get("alpaca", {})
            api_key = alpaca_secrets.get("api_key", "")
            secret_key = alpaca_secrets.get("secret_key", "")

            if not api_key or api_key == "YOUR_ALPACA_API_KEY":
                logger.warning(
                    "Alpaca API ključevi nisu konfigurirani. "
                    "Koristim Yahoo Finance kao fallback."
                )
                return YahooDataProvider()

            paper = config.broker.paper_trading
            return AlpacaDataProvider(
                api_key=api_key,
                secret_key=secret_key,
                paper=paper,
            )

        return YahooDataProvider()

    # ──────────────────────────────────────────────────────
    # Javno API sučelje
    # ──────────────────────────────────────────────────────

    def initialize_history(self) -> Dict[str, int]:
        """
        Dohvati i spremi historijske podatke za sve simbole.
        Poziva se jednom pri prvom pokretanju.
        Vraća mapu: simbol → broj dohvaćenih barova.
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=self.config.data.history_days)
        results = {}

        for symbol in self.symbols:
            existing = self.db.count_bars(symbol)
            if existing > 0:
                logger.info(
                    "%s: već postoji %d barova, preskačem inicijalizaciju",
                    symbol, existing
                )
                results[symbol] = existing
                continue

            bars = self.provider.get_historical_bars(symbol, start, end)
            if bars:
                inserted = self.db.insert_price_bars(bars)
                results[symbol] = inserted
                logger.info("%s: umetnuto %d barova", symbol, inserted)
            else:
                results[symbol] = 0
                logger.warning("%s: nije dohvaćen nijedan bar", symbol)

        return results

    def update_prices(self) -> Dict[str, Optional[PriceBar]]:
        """
        Dohvati i spremi najnovije cijene za sve simbole.
        Poziva se periodički (npr. svaku minutu kad je tržište otvoreno).
        """
        bars = self.provider.get_latest_bars_multi(self.symbols)

        for symbol, bar in bars.items():
            self.db.insert_price_bars([bar])

        # Logirati simbole za koje nema podataka
        missing = set(self.symbols) - set(bars.keys())
        if missing:
            logger.warning("Nema podataka za: %s", ", ".join(missing))

        return bars

    def get_snapshot(self) -> Optional[PortfolioSnapshot]:
        """
        Dohvati i spremi snapshot portfelja.
        Vraća PortfolioSnapshot ili None ako provider ne podržava.
        """
        snapshot = self.provider.get_portfolio_snapshot()
        if snapshot:
            self.db.insert_snapshot(snapshot)
            self.db.log_event(
                level="INFO",
                component="data_engine",
                message=f"Snapshot portfelja: equity=${snapshot.equity:,.2f}, "
                        f"cash=${snapshot.cash:,.2f}, "
                        f"positions={len(snapshot.positions)}",
            )
        return snapshot

    def is_market_open(self) -> bool:
        """Provjeri je li tržište otvoreno."""
        return self.provider.is_market_open()

    def get_price(self, symbol: str) -> Optional[float]:
        """Dohvati zadnju poznatu cijenu za simbol (close)."""
        bar = self.db.get_latest_price(symbol)
        return bar.close if bar else None

    def get_all_prices(self) -> Dict[str, float]:
        """Dohvati zadnje poznate cijene za sve simbole."""
        prices = {}
        for symbol in self.symbols:
            price = self.get_price(symbol)
            if price is not None:
                prices[symbol] = price
        return prices

    def get_history(
        self,
        symbol: str,
        days: int = 30,
    ) -> List[PriceBar]:
        """Dohvati historiju cijena za simbol iz baze."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        return self.db.get_price_bars(symbol, start=start, end=end)

    def health_check(self) -> Dict[str, any]:
        """Provjeri zdravlje Data Engine-a."""
        stats = self.db.get_db_stats()
        return {
            "status": "ok" if stats["price_bars"] > 0 else "empty",
            "provider": self.config.data.provider.value,
            "symbols_tracked": len(self.symbols),
            "total_bars": stats["price_bars"],
            "total_snapshots": stats["snapshots"],
            "db_size_kb": stats["db_size_kb"],
            "market_open": self.is_market_open(),
        }