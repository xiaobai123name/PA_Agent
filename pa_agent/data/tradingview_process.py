"""Isolated tvDatafeed worker process with hard cancellation and timeout."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import multiprocessing
from multiprocessing.connection import Connection
import threading
import time
import uuid
from typing import Any, Callable

logger = logging.getLogger(__name__)


class TradingViewProcessCancelled(Exception):
    """The parent cancelled an in-flight fetch."""


@dataclass(frozen=True)
class TradingViewFetchRequest:
    exchange: str
    symbol: str
    interval_name: str
    n_bars: int


def _close_worker_socket(tv: Any) -> None:
    ws = getattr(tv, "ws", None)
    if ws is None:
        return
    try:
        ws.close()
    except Exception:  # noqa: BLE001
        pass
    finally:
        try:
            tv.ws = None
        except Exception:  # noqa: BLE001
            pass


def tradingview_worker_main(
    conn: Connection,
    username: str,
    password: str,
    ws_timeout_s: float,
) -> None:
    """Own TvDatafeed and serve one fetch command at a time."""
    tv = None
    try:
        from tvDatafeed import Interval, TvDatafeed  # type: ignore[import]

        tv = TvDatafeed(username, password) if username and password else TvDatafeed()
        try:
            setattr(tv, "_TvDatafeed__ws_timeout", ws_timeout_s)
        except Exception:  # noqa: BLE001
            pass

        while True:
            try:
                command = conn.recv()
            except (EOFError, OSError):
                break
            if command.get("op") == "stop":
                break
            if command.get("op") != "fetch":
                continue

            request_id = command["request_id"]
            try:
                interval = getattr(Interval, command["interval_name"])
                df = tv.get_hist(
                    symbol=command["symbol"],
                    exchange=command["exchange"],
                    interval=interval,
                    n_bars=int(command["n_bars"]),
                )
                rows: list[dict[str, Any]] = []
                if df is not None and not df.empty:
                    frame = df.reset_index()
                    for row in frame.itertuples(index=False):
                        rows.append(
                            {
                                "datetime": getattr(row, "datetime", None),
                                "open": float(row.open),
                                "high": float(row.high),
                                "low": float(row.low),
                                "close": float(row.close),
                                "volume": float(getattr(row, "volume", 0.0)),
                            }
                        )
                conn.send({"request_id": request_id, "ok": True, "rows": rows})
            except Exception as exc:  # noqa: BLE001
                try:
                    conn.send(
                        {
                            "request_id": request_id,
                            "ok": False,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                except (BrokenPipeError, EOFError, OSError):
                    break
            finally:
                if tv is not None:
                    _close_worker_socket(tv)
    finally:
        if tv is not None:
            _close_worker_socket(tv)
        try:
            conn.close()
        except OSError:
            pass


class TradingViewFetchSupervisor:
    """Supervise one long-lived worker and kill it on timeout or cancellation."""

    _POLL_S = 0.05
    _TERMINATE_JOIN_S = 1.0

    def __init__(
        self,
        *,
        username: str = "",
        password: str = "",
        ws_timeout_s: float = 10.0,
        context: Any = None,
        worker_target: Callable[..., None] = tradingview_worker_main,
    ) -> None:
        self._username = username
        self._password = password
        self._ws_timeout_s = ws_timeout_s
        self._context = context or multiprocessing.get_context("spawn")
        self._worker_target = worker_target
        self._state_lock = threading.Lock()
        self._fetch_lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._process: Any = None
        self._conn: Connection | None = None

    @property
    def is_running(self) -> bool:
        with self._state_lock:
            return bool(self._process is not None and self._process.is_alive())

    def start(self) -> None:
        self._stop_requested.clear()
        with self._state_lock:
            if self._process is not None and self._process.is_alive():
                return
            parent_conn, child_conn = self._context.Pipe(duplex=True)
            process = self._context.Process(
                target=self._worker_target,
                args=(
                    child_conn,
                    self._username,
                    self._password,
                    self._ws_timeout_s,
                ),
                name="pa-agent-tvdatafeed",
                daemon=True,
            )
            process.start()
            child_conn.close()
            self._process = process
            self._conn = parent_conn

    def stop(self) -> None:
        self._stop_requested.set()
        process, conn = self._detach_child()
        if conn is not None:
            try:
                conn.send({"op": "stop"})
            except (BrokenPipeError, EOFError, OSError):
                pass
        self._terminate(process, conn, graceful_s=0.2)

    def cancel_inflight(self) -> None:
        """Hard-stop the worker so a blocked recv cannot outlive cancellation."""
        process, conn = self._detach_child()
        self._terminate(process, conn, graceful_s=0.0)

    def fetch(
        self,
        request: TradingViewFetchRequest,
        *,
        cancel_token: object | None = None,
        timeout_s: float = 12.0,
    ) -> list[dict[str, Any]]:
        while not self._fetch_lock.acquire(timeout=self._POLL_S):
            if self._is_cancelled(cancel_token) or self._stop_requested.is_set():
                raise TradingViewProcessCancelled("TradingView request cancelled")
        try:
            if self._is_cancelled(cancel_token) or self._stop_requested.is_set():
                raise TradingViewProcessCancelled("TradingView request cancelled")
            self.start()
            with self._state_lock:
                process = self._process
                conn = self._conn
            if process is None or conn is None:
                raise RuntimeError("TradingView worker did not start")

            request_id = uuid.uuid4().hex
            try:
                conn.send(
                    {
                        "op": "fetch",
                        "request_id": request_id,
                        "exchange": request.exchange,
                        "symbol": request.symbol,
                        "interval_name": request.interval_name,
                        "n_bars": request.n_bars,
                    }
                )
            except (BrokenPipeError, EOFError, OSError) as exc:
                self.cancel_inflight()
                raise RuntimeError("TradingView worker connection failed") from exc

            deadline = time.monotonic() + max(0.1, float(timeout_s))
            while True:
                if self._is_cancelled(cancel_token) or self._stop_requested.is_set():
                    self.cancel_inflight()
                    raise TradingViewProcessCancelled("TradingView request cancelled")
                if time.monotonic() >= deadline:
                    self.cancel_inflight()
                    raise TimeoutError(
                        f"TradingView fetch timed out after {float(timeout_s):.1f}s"
                    )
                if not process.is_alive():
                    self.cancel_inflight()
                    raise RuntimeError("TradingView worker exited unexpectedly")
                try:
                    ready = conn.poll(self._POLL_S)
                except (EOFError, OSError) as exc:
                    self.cancel_inflight()
                    raise RuntimeError("TradingView worker connection closed") from exc
                if not ready:
                    continue
                try:
                    result = conn.recv()
                except (EOFError, OSError) as exc:
                    self.cancel_inflight()
                    raise RuntimeError("TradingView worker response failed") from exc
                if result.get("request_id") != request_id:
                    continue
                if not result.get("ok"):
                    error_type = result.get("error_type") or "Error"
                    message = result.get("error") or "unknown error"
                    raise RuntimeError(f"{error_type}: {message}")
                return list(result.get("rows") or [])
        finally:
            self._fetch_lock.release()

    def _detach_child(self) -> tuple[Any, Connection | None]:
        with self._state_lock:
            process = self._process
            conn = self._conn
            self._process = None
            self._conn = None
        return process, conn

    def _terminate(
        self,
        process: Any,
        conn: Connection | None,
        *,
        graceful_s: float,
    ) -> None:
        if process is not None:
            if graceful_s > 0:
                process.join(graceful_s)
            if process.is_alive():
                process.terminate()
                process.join(self._TERMINATE_JOIN_S)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(self._TERMINATE_JOIN_S)
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass

    @staticmethod
    def _is_cancelled(cancel_token: object | None) -> bool:
        checker = getattr(cancel_token, "is_set", None)
        return bool(callable(checker) and checker())
