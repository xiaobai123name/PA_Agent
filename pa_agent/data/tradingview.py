"""TradingView data source using tvdatafeed."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Mapping

from pa_agent.data.base import (
    DataSource,
    DataSourceCancelledError,
    DataSourceEmptyError,
    DataSourceInvalidSymbolError,
    DataSourceTransientError,
    KlineBar,
    VolumeMeta,
    normalize_kline_bar,
)
from pa_agent.data.datetime_ts import datetime_to_ts_ms, naive_local_to_utc
from pa_agent.data.market_defaults import (
    is_tv_exchange_auto,
    normalize_binance_display_symbol,
    resolve_tv_fetch_pair,
    tv_auto_probe_plan,
)
from pa_agent.data.tradingview_errors import format_tradingview_fetch_error
from pa_agent.data.tradingview_process import (
    TradingViewFetchRequest,
    TradingViewFetchSupervisor,
    TradingViewProcessCancelled,
)
from pa_agent.data.tv_symbol_lookup import TvSymbolNotFoundError

logger = logging.getLogger(__name__)

# The parent kills the child if tvDatafeed ignores its WebSocket timeout.
_TV_WS_TIMEOUT_S = 10.0
_TV_HARD_TIMEOUT_S = 12.0
_TRADED_VOLUME_EXCHANGES = frozenset(
    {"BINANCE", "OKX", "SSE", "SZSE", "HKEX", "NYSE", "NASDAQ", "CBOT", "CME_MINI"}
)

# Map our timeframe strings to tvDatafeed Interval enum names
_TF_MAP: dict[str, str] = {
    "1m":  "in_1_minute",
    "3m":  "in_3_minute",
    "5m":  "in_5_minute",
    "15m": "in_15_minute",
    "30m": "in_30_minute",
    "45m": "in_45_minute",
    "1h":  "in_1_hour",
    "2h":  "in_2_hour",
    "3h":  "in_3_hour",
    "4h":  "in_4_hour",
    "1d":  "in_daily",
    "1w":  "in_weekly",
    "1M":  "in_monthly",
}

# Forex / spot gold and China A-share (tvDatafeed exchange ids)
TV_EXCHANGE_PRESETS: tuple[str, ...] = (
    "OANDA",
    "PEPPERSTONE",
    "FOREXCOM",
    "FX",
    "BINANCE",
    "OKX",
    "TVC",
    "CAPITALCOM",
    "SSE",
    "SZSE",
    "HKEX",
    "SP",
    "NYSE",
    "NASDAQ",
    "CBOT",
    "CME_MINI",
    "",
)

TV_SYMBOL_PRESETS_BY_EXCHANGE: dict[str, tuple[str, ...]] = {
    "OANDA": (
        "XAUUSD", "XAGUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
        "USDCAD", "USDCHF", "NZDUSD", "EURJPY", "GBPJPY",
    ),
    "PEPPERSTONE": (
        "XAUUSD", "XAGUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
        "USDCAD", "USDCHF", "NZDUSD", "EURJPY", "GBPJPY",
    ),
    "FOREXCOM": (
        "XAUUSD", "XAGUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
        "USDCAD", "USDCHF", "NZDUSD", "EURJPY", "GBPJPY",
    ),
    "FX": (
        "XAUUSD", "XAGUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
        "USDCAD", "USDCHF", "NZDUSD", "EURJPY", "GBPJPY",
    ),
    "BINANCE": (
        "BTCUSDT", "ETHUSDT", "XAUUSDT", "QQQUSDT",
    ),
    "OKX": (
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "OKBUSDT", "XRPUSDT", "DOGEUSDT",
        "ADAUSDT", "LINKUSDT", "AVAXUSDT", "LTCUSDT", "BCHUSDT", "DOTUSDT",
    ),
    "TVC": (
        "GOLD", "SILVER", "DXY", "SPX", "NDX", "DJI", "USOIL", "UKOIL", "US10Y",
    ),
    "CAPITALCOM": (
        "GOLD", "SILVER", "US100", "US500", "US30", "OIL_CRUDE", "OIL_BRENT",
        "EURUSD", "GBPUSD",
    ),
    "SSE": (
        "600519", "601318", "600036", "601398", "688981", "601899", "000300",
        "000001",
    ),
    "SZSE": (
        "000001", "300750", "002594", "000858", "002475", "300059", "399001",
        "399006",
    ),
    "HKEX": (
        "1810", "700", "3690", "9988", "1211", "941", "939", "1398", "5",
        "1299", "1024", "9618", "9999", "9888", "2331", "9626", "2015",
        "9866", "9868",
    ),
    "SP": ("SPX", "SPY", "ES1!", "VIX"),
    "NYSE": ("BABA", "NIO", "XPEV", "LI", "KO", "JPM", "DIS", "BA", "XOM"),
    "NASDAQ": (
        "AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "META", "AMZN", "BILI", "QQQ",
        "NDX",
    ),
    "CBOT": ("ZS1!", "ZC1!", "ZW1!", "ZB1!", "ZN1!"),
    "CME_MINI": ("ES1!", "NQ1!", "RTY1!", "YM1!", "GC1!", "CL1!"),
    "": (
        "XAUUSD", "GOLD", "EURUSD", "GBPUSD", "600519", "000001", "1810", "700",
        "AAPL", "NVDA", "TSLA", "BTCUSDT", "ETHUSDT",
    ),
}


class TradingViewSource(DataSource):
    """Live K-line data from TradingView via tvdatafeed."""

    def __init__(self, username: str = "", password: str = "") -> None:
        self._username = username
        self._password = password
        self._supervisor: TradingViewFetchSupervisor | None = None
        self._connected: bool = False
        self._symbol: str = ""
        self._timeframe: str = ""
        self._exchange: str = ""
        self._resolved_exchange: str = ""
        # Callback for status updates during auto-probe: fn(symbol, exchange, label)
        self.on_probe_status = None

    @property
    def exchange(self) -> str:
        return self._exchange

    @property
    def volume_meta(self) -> VolumeMeta:
        exchange = self._resolved_exchange or self._exchange
        if exchange in _TRADED_VOLUME_EXCHANGES:
            return VolumeMeta(
                kind="traded",
                source=f"TradingView:{exchange}",
                unit="provider_reported",
            )
        return VolumeMeta(
            kind="unknown",
            source=f"TradingView:{exchange or 'AUTO'}",
            unit="provider_reported",
        )

    def set_exchange(self, exchange: str) -> None:
        """Set TradingView exchange id (e.g. ``BINANCE``); empty = auto-detect."""
        self._exchange = (exchange or "").strip().upper()
        self._resolved_exchange = ""

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        try:
            import tvDatafeed  # type: ignore[import]  # noqa: F401

            self._supervisor = TradingViewFetchSupervisor(
                username=self._username,
                password=self._password,
                ws_timeout_s=_TV_WS_TIMEOUT_S,
            )
            self._supervisor.start()
            self._connected = True
            logger.info("TradingViewSource connected (anonymous=%s)", not self._username)
        except Exception as exc:
            self._connected = False
            raise DataSourceTransientError(
                f"TradingView 连接失败：{exc}（若未安装请执行 "
                "pip install git+https://github.com/rongardF/tvdatafeed.git）"
            ) from exc

    def disconnect(self) -> None:
        supervisor = self._supervisor
        self._supervisor = None
        if supervisor is not None:
            supervisor.stop()
        self._connected = False
        logger.info("TradingViewSource disconnected")

    def cancel_pending(self) -> None:
        supervisor = self._supervisor
        if supervisor is not None:
            supervisor.cancel_inflight()

    def _close_tv_socket(self) -> None:
        """Compatibility alias for callers that used to close a shared socket."""
        self.cancel_pending()

    # ── Discovery ─────────────────────────────────────────────────────────────

    def list_symbols(self) -> list[str]:
        symbols = TV_SYMBOL_PRESETS_BY_EXCHANGE.get(
            self._exchange, TV_SYMBOL_PRESETS_BY_EXCHANGE[""]
        )
        return list(symbols)

    def supported_timeframes(self) -> list[str]:
        return list(_TF_MAP.keys())

    # ── Subscription ──────────────────────────────────────────────────────────

    def subscribe(self, symbol: str, timeframe: str) -> None:
        if timeframe not in _TF_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe!r}. Use one of {list(_TF_MAP)}")
        clean_symbol = symbol.strip()
        if self._exchange == "BINANCE":
            try:
                resolve_tv_fetch_pair(self._exchange, symbol)
            except TvSymbolNotFoundError as exc:
                raise DataSourceInvalidSymbolError(str(exc)) from exc
            clean_symbol = normalize_binance_display_symbol(symbol)
        self._timeframe = timeframe
        self._symbol = clean_symbol
        self._resolved_exchange = ""
        self.cancel_pending()
        logger.info(
            "TradingViewSource subscribed: %s %s exchange=%s",
            self._symbol,
            timeframe,
            self._exchange or "(auto)",
        )

    def unsubscribe(self) -> None:
        self._symbol = ""
        self._timeframe = ""
        self._resolved_exchange = ""
        logger.info("TradingViewSource unsubscribed")

    # ── Data fetch ────────────────────────────────────────────────────────────

    def _fetch_hist_with_retry(
        self,
        *,
        symbol: str,
        exchange: str,
        interval_name: str,
        n_bars: int,
        cancel_token: object | None,
        timeout_s: float,
    ) -> list[dict[str, Any]]:
        """Fetch once through the isolated worker; RefreshLoop owns retries."""
        supervisor = self._supervisor
        if supervisor is None:
            raise DataSourceTransientError("TradingView 未连接，请先选择数据来源 TradingView")
        logger.debug(
            "TradingView get_hist: display=%s feed=%s:%s interval=%s n_bars=%d",
            self._symbol,
            exchange,
            symbol,
            interval_name,
            n_bars,
        )
        return supervisor.fetch(
            TradingViewFetchRequest(
                exchange=exchange,
                symbol=symbol,
                interval_name=interval_name,
                n_bars=n_bars,
            ),
            cancel_token=cancel_token,
            timeout_s=timeout_s,
        )

    def _fetch_tv_auto_probe(
        self,
        *,
        symbol: str,
        plan: list[tuple[str, str]],
        interval_name: str,
        n_bars: int,
        cancel_token: object | None,
        timeout_s: float,
    ) -> tuple[list[dict[str, Any]], str]:
        """Try each (exchange, symbol) in *plan* until one returns bars."""
        if not plan:
            raise DataSourceTransientError(
                f"TradingView 无法识别品种「{symbol}」；"
                "请用 A 股 6 位代码、港股代码（如 1810）、"
                "指数代码（如 SPX、NDX、VIX）、"
                "外汇/黄金代码或已支持的股票名称"
            )
        last_exc: BaseException | None = None
        tried: list[str] = []
        for exchange, code in plan:
            label = f"{exchange}:{code}"
            tried.append(label)
            # Notify GUI about current probe attempt
            if self.on_probe_status is not None:
                try:
                    self.on_probe_status(symbol, exchange, label)
                except Exception:  # noqa: BLE001
                    pass
            try:
                rows = self._fetch_hist_with_retry(
                    symbol=code,
                    exchange=exchange,
                    interval_name=interval_name,
                    n_bars=n_bars,
                    cancel_token=cancel_token,
                    timeout_s=timeout_s,
                )
            except Exception as exc:
                last_exc = exc
                logger.info("TradingView auto probe %s failed: %s", label, exc)
                continue
            if rows:
                logger.info(
                    "TradingView auto probe picked %s (tried %s)",
                    label,
                    ", ".join(tried),
                )
                return rows, exchange
        if last_exc is not None:
            raise last_exc
        raise DataSourceTransientError(
            f"TradingView 自动探测失败（{symbol}）：已尝试 {', '.join(tried)} 均无 K 线"
        )

    def latest_snapshot(
        self,
        n: int,
        *,
        cancel_token: object | None = None,
        timeout_s: float | None = None,
    ) -> list[KlineBar]:
        """Return *n* bars newest-first through the isolated fetch process."""
        if not self._connected or self._supervisor is None:
            raise DataSourceTransientError("TradingView 未连接，请先选择数据来源 TradingView")
        if not self._symbol or not self._timeframe:
            raise DataSourceTransientError("TradingView 未订阅品种/周期")

        user_symbol = self._symbol
        req_exchange = self._exchange
        exchange = req_exchange or ""
        fetch_symbol = user_symbol
        auto_probe = is_tv_exchange_auto(req_exchange)
        probe_plan = tv_auto_probe_plan(user_symbol) if auto_probe else []
        hard_timeout = _TV_HARD_TIMEOUT_S if timeout_s is None else float(timeout_s)
        try:
            interval_name = _TF_MAP[self._timeframe]
            if auto_probe and probe_plan:
                rows, exchange = self._fetch_tv_auto_probe(
                    symbol=user_symbol,
                    plan=probe_plan,
                    interval_name=interval_name,
                    n_bars=n + 1,
                    cancel_token=cancel_token,
                    timeout_s=hard_timeout,
                )
            else:
                try:
                    exchange, fetch_symbol = resolve_tv_fetch_pair(
                        req_exchange, user_symbol
                    )
                except TvSymbolNotFoundError as exc:
                    raise DataSourceInvalidSymbolError(str(exc)) from exc
                rows = self._fetch_hist_with_retry(
                    symbol=fetch_symbol,
                    exchange=exchange,
                    interval_name=interval_name,
                    n_bars=n + 1,
                    cancel_token=cancel_token,
                    timeout_s=hard_timeout,
                )
        except (DataSourceInvalidSymbolError, DataSourceTransientError):
            raise
        except TradingViewProcessCancelled as exc:
            raise DataSourceCancelledError(str(exc)) from exc
        except Exception as exc:
            msg = format_tradingview_fetch_error(
                user_symbol, exchange or req_exchange or "自动", cause=exc,
            )
            logger.warning("TradingView fetch failed: %s", exc)
            raise DataSourceTransientError(msg) from exc

        if not rows:
            if req_exchange == "BINANCE":
                msg = (
                    f"Binance {user_symbol} 当前无可用 K 线，"
                    "可能休市或 TradingView 暂未返回数据"
                )
            else:
                msg = format_tradingview_fetch_error(
                    user_symbol, exchange or req_exchange or "自动", empty_data=True,
                )
            logger.debug(
                "TradingView empty data for %s exchange=%s",
                user_symbol,
                exchange or req_exchange or "(auto)",
            )
            raise DataSourceEmptyError(msg)

        self._resolved_exchange = exchange

        return _tv_rows_to_bars(rows, self._timeframe, n)

    def fetch_frame_once(
        self,
        symbol: str,
        timeframe: str,
        n: int,
        *,
        cancel_token: object | None = None,
        timeout_s: float | None = None,
    ) -> list[KlineBar]:
        """One-shot fetch reusing the live subscription's resolved exchange."""
        if not self._connected or self._supervisor is None:
            return []
        if timeframe not in _TF_MAP or not symbol:
            return []
        req_exchange = self._resolved_exchange or self._exchange
        if is_tv_exchange_auto(req_exchange):
            return []
        try:
            exchange, fetch_symbol = resolve_tv_fetch_pair(req_exchange, symbol)
            rows = self._fetch_hist_with_retry(
                symbol=fetch_symbol,
                exchange=exchange,
                interval_name=_TF_MAP[timeframe],
                n_bars=n + 1,
                cancel_token=cancel_token,
                timeout_s=_TV_HARD_TIMEOUT_S if timeout_s is None else float(timeout_s),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "TradingView fetch_frame_once failed for %s %s: %s",
                symbol, timeframe, exc,
            )
            return []
        if not rows:
            return []
        return _tv_rows_to_bars(rows, timeframe, n)


def _tv_rows_to_bars(rows: Any, timeframe: str, n: int) -> list[KlineBar]:
    """Convert tvDatafeed rows (oldest-first) to newest-first KlineBar list."""
    bars: list[KlineBar] = []
    for i, row in enumerate(reversed(rows)):
        ts_ms = _row_ts_ms(row)
        bar = KlineBar(
            seq=i + 1,
            ts_open=ts_ms,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 0.0)),
            closed=True,
        )
        if i == 0:
            # seconds_until_bar_closes only looks at the timestamp, so it is
            # robust to constant broker-time offsets (see latest_snapshot note).
            from pa_agent.data.bar_close_wait import seconds_until_bar_closes

            secs_left = seconds_until_bar_closes(ts_ms, timeframe, now_ms=None)
            still_forming = secs_left is not None and secs_left > 0
            bar = KlineBar(
                seq=bar.seq,
                ts_open=bar.ts_open,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                closed=not still_forming,
            )
        bars.append(normalize_kline_bar(bar))
        if len(bars) >= n:
            break

    return bars


def _row_ts_ms(row: object) -> int:
    """Extract bar open time in milliseconds from a tvDatafeed DataFrame row."""
    if isinstance(row, Mapping):
        dt = row.get("datetime")
    else:
        dt = getattr(row, "datetime", None)
    # tvDatafeed builds its DataFrame with:
    #     datetime.datetime.fromtimestamp(raw_epoch_seconds)
    # so the value is a *host-local naive* datetime, not a UTC-naive datetime.
    # Convert it back through local time first; otherwise non-UTC hosts shift
    # TradingView candles by the local UTC offset (UTC+8 => +8h).
    if isinstance(dt, datetime) and dt.tzinfo is None:
        return datetime_to_ts_ms(naive_local_to_utc(dt))
    return datetime_to_ts_ms(dt)
