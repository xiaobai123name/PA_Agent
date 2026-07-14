"""Download Binance Futures history and freeze it into immutable SQLite datasets."""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pa_agent.backtest.models import ContractMetadata, FrozenDataset
from pa_agent.config.paths import BACKTEST_DATA_DIR
from pa_agent.data.base import KlineBar


BINANCE_BACKTEST_SYMBOLS = ("BTCUSDT", "ETHUSDT", "XAUUSDT", "QQQUSDT")
BINANCE_TIMEFRAMES = ("1m", "5m", "15m", "1h", "4h", "1d")
_INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}
_API_ROOT = "https://fapi.binance.com"


class HistoricalDataError(RuntimeError):
    """Historical data cannot produce a valid frozen dataset."""


class BinanceFuturesHttpClient:
    """Small injectable JSON client for the public Binance Futures endpoints."""

    def __init__(self, *, base_url: str = _API_ROOT, timeout_s: float = 20.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = float(timeout_s)

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        query = urlencode(params or {})
        url = f"{self._base_url}{path}" + (f"?{query}" if query else "")
        request = Request(url, headers={"User-Agent": "PA-Agent/0.1"})
        with urlopen(request, timeout=self._timeout_s) as response:  # noqa: S310
            raw = response.read().decode("utf-8")
        return json.loads(raw)


class HistoricalDataRepository:
    """Create and load content-addressed Binance history datasets."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        client: BinanceFuturesHttpClient | Any | None = None,
    ) -> None:
        self._root = root or BACKTEST_DATA_DIR
        self._client = client or BinanceFuturesHttpClient()

    def download_and_freeze(
        self,
        symbol: str,
        analysis_timeframe: str,
        start_ms: int,
        end_ms: int,
        *,
        analysis_bar_count: int = 100,
        cancel_token: object | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> FrozenDataset:
        symbol = symbol.strip().upper()
        timeframe = analysis_timeframe.strip().lower()
        if symbol not in BINANCE_BACKTEST_SYMBOLS:
            raise HistoricalDataError(f"回测不支持品种：{symbol}")
        if timeframe not in BINANCE_TIMEFRAMES:
            raise HistoricalDataError(f"回测不支持周期：{timeframe}")
        if analysis_bar_count < 2:
            raise HistoricalDataError("analysis_bar_count 必须至少为 2")
        if int(start_ms) >= int(end_ms):
            raise HistoricalDataError("回测开始时间必须早于结束时间")
        interval_ms = _INTERVAL_MS[timeframe]
        if int(start_ms) % interval_ms != 0 or int(end_ms) % interval_ms != 0:
            raise HistoricalDataError(
                f"开始和结束时间必须对齐 {timeframe} 的 UTC K线边界"
            )

        exchange_info = self._client.get_json("/fapi/v1/exchangeInfo")
        metadata = self._parse_metadata(exchange_info, symbol)
        time_payload = self._client.get_json("/fapi/v1/time")
        if not isinstance(time_payload, dict) or not time_payload.get("serverTime"):
            raise HistoricalDataError("Binance 服务器时间响应无效")
        server_time = int(time_payload["serverTime"])
        if start_ms < metadata.onboard_date_ms:
            raise HistoricalDataError(
                f"请求开始时间早于 {symbol} 上市时间，禁止自动截断区间"
            )
        if end_ms > server_time:
            raise HistoricalDataError("请求结束时间晚于 Binance 服务器当前时间")

        warmup_count = analysis_bar_count + 50
        self._check_cancel(cancel_token)
        if on_progress:
            on_progress("下载分析周期预热数据", 0, 4)
        warmup = self._fetch_before(
            symbol,
            timeframe,
            end_ms=start_ms - 1,
            count=warmup_count,
            cancel_token=cancel_token,
        )
        if len(warmup) < analysis_bar_count:
            raise HistoricalDataError(
                f"开始时间之前只有 {len(warmup)} 根 {timeframe} K线，"
                f"不足分析窗口 {analysis_bar_count} 根"
            )

        if on_progress:
            on_progress("下载分析周期数据", 1, 4)
        target_analysis = self._fetch_range(
            symbol,
            timeframe,
            start_ms,
            end_ms,
            cancel_token=cancel_token,
        )
        target_analysis = [
            row for row in target_analysis if row[0] + interval_ms <= end_ms
        ]
        if not target_analysis:
            raise HistoricalDataError("请求区间没有完整收盘的分析周期 K线")

        if on_progress:
            on_progress("下载1分钟执行数据", 2, 4)
        execution = self._fetch_range(
            symbol,
            "1m",
            start_ms,
            end_ms,
            cancel_token=cancel_token,
        )
        execution = [row for row in execution if row[0] + 60_000 <= end_ms]
        if not execution:
            raise HistoricalDataError("请求区间没有可用的1分钟执行 K线")

        analysis_rows = self._merge_rows(warmup, target_analysis)
        allow_exec_gaps = metadata.contract_type == "TRADIFI_PERPETUAL"
        self._validate_rows(analysis_rows, timeframe, allow_gaps=allow_exec_gaps)
        gap_count = self._validate_rows(execution, "1m", allow_gaps=allow_exec_gaps)
        target_start = target_analysis[0][0]
        target_end = target_analysis[-1][0] + interval_ms

        canonical = {
            "symbol": symbol,
            "analysis_timeframe": timeframe,
            "requested_start_ms": int(start_ms),
            "requested_end_ms": int(end_ms),
            "target_start_ms": target_start,
            "target_end_ms": target_end,
            "analysis_bar_count": int(analysis_bar_count),
            "metadata": asdict(metadata),
            "analysis_rows": analysis_rows,
            "execution_rows": execution,
            "execution_gap_count": gap_count,
        }
        payload = json.dumps(
            canonical,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        dataset_hash = hashlib.sha256(payload).hexdigest()
        dataset_id = dataset_hash[:16]
        self._root.mkdir(parents=True, exist_ok=True)
        path = self._root / f"{dataset_id}.sqlite"
        if on_progress:
            on_progress("冻结本地数据集", 3, 4)
        if not path.exists():
            self._write_dataset(path, canonical, dataset_hash, dataset_id)
        frozen = FrozenDataset(
            dataset_id=dataset_id,
            dataset_hash=dataset_hash,
            path=path,
            symbol=symbol,
            analysis_timeframe=timeframe,
            requested_start_ms=int(start_ms),
            requested_end_ms=int(end_ms),
            target_start_ms=target_start,
            target_end_ms=target_end,
            analysis_bar_count=int(analysis_bar_count),
            metadata=metadata,
        )
        if on_progress:
            on_progress("数据集已冻结", 4, 4)
        return frozen

    def load_bars(self, dataset: FrozenDataset, timeframe: str) -> list[KlineBar]:
        with sqlite3.connect(dataset.path) as conn:
            rows = conn.execute(
                "SELECT open_time, open, high, low, close, volume "
                "FROM bars WHERE timeframe=? ORDER BY open_time ASC",
                (timeframe,),
            ).fetchall()
        return [
            KlineBar(
                seq=i + 1,
                ts_open=int(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                closed=True,
            )
            for i, row in enumerate(rows)
        ]

    def _fetch_range(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        *,
        cancel_token: object | None,
    ) -> list[list[Any]]:
        rows: list[list[Any]] = []
        cursor = int(start_ms)
        while cursor < end_ms:
            self._check_cancel(cancel_token)
            batch = self._client.get_json(
                "/fapi/v1/klines",
                {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": end_ms - 1,
                    "limit": 1500,
                },
            )
            if not isinstance(batch, list):
                raise HistoricalDataError(f"Binance {interval} K线响应不是数组")
            if not batch:
                break
            normalized = [self._normalize_row(row) for row in batch]
            rows.extend(normalized)
            last_open = int(normalized[-1][0])
            if last_open < cursor:
                raise HistoricalDataError("Binance K线分页游标倒退")
            cursor = last_open + _INTERVAL_MS[interval]
            if len(batch) < 1500:
                break
        return [row for row in self._merge_rows([], rows) if start_ms <= row[0] < end_ms]

    def _fetch_before(
        self,
        symbol: str,
        interval: str,
        *,
        end_ms: int,
        count: int,
        cancel_token: object | None,
    ) -> list[list[Any]]:
        collected: list[list[Any]] = []
        cursor_end = int(end_ms)
        while len(collected) < count:
            self._check_cancel(cancel_token)
            limit = min(1500, count - len(collected))
            batch = self._client.get_json(
                "/fapi/v1/klines",
                {
                    "symbol": symbol,
                    "interval": interval,
                    "endTime": cursor_end,
                    "limit": limit,
                },
            )
            if not isinstance(batch, list):
                raise HistoricalDataError(f"Binance {interval} 预热响应不是数组")
            if not batch:
                break
            normalized = [self._normalize_row(row) for row in batch]
            collected = normalized + collected
            first_open = int(normalized[0][0])
            if first_open >= cursor_end:
                raise HistoricalDataError("Binance 预热分页游标未前进")
            cursor_end = first_open - 1
            if len(batch) < limit:
                break
        return self._merge_rows([], collected)[-count:]

    @staticmethod
    def _parse_metadata(exchange_info: dict[str, Any], symbol: str) -> ContractMetadata:
        symbols = exchange_info.get("symbols")
        if not isinstance(symbols, list):
            raise HistoricalDataError("Binance exchangeInfo 缺少 symbols")
        item = next((row for row in symbols if row.get("symbol") == symbol), None)
        if not isinstance(item, dict):
            raise HistoricalDataError(f"Binance Futures 不存在品种：{symbol}")
        if item.get("status") != "TRADING":
            raise HistoricalDataError(f"{symbol} 当前状态不是 TRADING")
        filters = {
            str(row.get("filterType")): row
            for row in item.get("filters") or []
            if isinstance(row, dict)
        }
        try:
            tick_size = float(filters["PRICE_FILTER"]["tickSize"])
            lot = filters["LOT_SIZE"]
            step_size = float(lot["stepSize"])
            min_qty = float(lot["minQty"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HistoricalDataError(f"{symbol} 交易精度元数据不完整") from exc
        notional_filter = filters.get("MIN_NOTIONAL") or filters.get("NOTIONAL") or {}
        min_notional = float(
            notional_filter.get("notional")
            or notional_filter.get("minNotional")
            or 0.0
        )
        metadata = ContractMetadata(
            symbol=symbol,
            status=str(item.get("status")),
            contract_type=str(item.get("contractType")),
            onboard_date_ms=int(item.get("onboardDate") or 0),
            tick_size=tick_size,
            step_size=step_size,
            min_qty=min_qty,
            min_notional=min_notional,
        )
        if (
            metadata.onboard_date_ms <= 0
            or metadata.tick_size <= 0
            or metadata.step_size <= 0
            or metadata.min_qty <= 0
        ):
            raise HistoricalDataError(f"{symbol} 元数据包含无效数值")
        return metadata

    @staticmethod
    def _normalize_row(row: Any) -> list[Any]:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            raise HistoricalDataError("Binance K线行格式错误")
        values = [
            int(row[0]),
            float(row[1]),
            float(row[2]),
            float(row[3]),
            float(row[4]),
            float(row[5]),
        ]
        if not all(math.isfinite(float(value)) for value in values[1:]):
            raise HistoricalDataError("Binance K线包含非有限数值")
        if values[2] < values[3] or not (values[3] <= values[4] <= values[2]):
            raise HistoricalDataError("Binance K线 OHLC 关系无效")
        return values

    @staticmethod
    def _merge_rows(left: Iterable[list[Any]], right: Iterable[list[Any]]) -> list[list[Any]]:
        merged: dict[int, list[Any]] = {}
        for row in [*left, *right]:
            ts = int(row[0])
            existing = merged.get(ts)
            if existing is not None and existing != row:
                raise HistoricalDataError(f"同一时间 {ts} 存在冲突 K线")
            merged[ts] = list(row)
        return [merged[ts] for ts in sorted(merged)]

    @staticmethod
    def _validate_rows(rows: list[list[Any]], interval: str, *, allow_gaps: bool) -> int:
        if not rows:
            raise HistoricalDataError(f"{interval} 数据为空")
        expected = _INTERVAL_MS[interval]
        gaps = 0
        previous = None
        for row in rows:
            ts = int(row[0])
            if previous is not None:
                delta = ts - previous
                if delta <= 0:
                    raise HistoricalDataError(f"{interval} K线时间未严格递增")
                if delta != expected:
                    gaps += 1
                    if not allow_gaps:
                        raise HistoricalDataError(
                            f"普通永续合约 {interval} 数据存在缺口：{previous} -> {ts}"
                        )
            previous = ts
        return gaps

    @staticmethod
    def _write_dataset(
        path: Path,
        canonical: dict[str, Any],
        dataset_hash: str,
        dataset_id: str,
    ) -> None:
        tmp = path.with_suffix(".tmp")
        if tmp.exists():
            tmp.unlink()
        conn = sqlite3.connect(tmp)
        try:
            conn.executescript(
                """
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE bars (
                    timeframe TEXT NOT NULL,
                    open_time INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    PRIMARY KEY (timeframe, open_time)
                );
                """
            )
            meta = dict(canonical)
            analysis_rows = meta.pop("analysis_rows")
            execution_rows = meta.pop("execution_rows")
            meta["dataset_hash"] = dataset_hash
            meta["dataset_id"] = dataset_id
            for key, value in meta.items():
                conn.execute(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)",
                    (key, json.dumps(value, ensure_ascii=False, sort_keys=True)),
                )
            analysis_tf = str(canonical["analysis_timeframe"])
            conn.executemany(
                "INSERT INTO bars VALUES (?, ?, ?, ?, ?, ?, ?)",
                ((analysis_tf, *row) for row in analysis_rows),
            )
            if analysis_tf != "1m":
                conn.executemany(
                    "INSERT INTO bars VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (("1m", *row) for row in execution_rows),
                )
            conn.commit()
        finally:
            conn.close()
        tmp.replace(path)

    @staticmethod
    def _check_cancel(cancel_token: object | None) -> None:
        checker = getattr(cancel_token, "is_set", None)
        if callable(checker) and checker():
            raise InterruptedError("回测数据下载已取消")


def timeframe_ms(timeframe: str) -> int:
    try:
        return _INTERVAL_MS[timeframe]
    except KeyError as exc:
        raise HistoricalDataError(f"不支持周期：{timeframe}") from exc
