"""Cancellable K-line refresh state machine running on one QThread."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
import threading
import time
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from pa_agent.data.base import (
    DataSource,
    DataSourceCancelledError,
    DataSourceEmptyError,
    DataSourceInvalidSymbolError,
    DataSourceTransientError,
)
from pa_agent.data.snapshot import INDICATOR_WARMUP_BARS

if TYPE_CHECKING:
    from pa_agent.util.threading import CancelToken

logger = logging.getLogger(__name__)


class RefreshPhase(str, Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    LIVE = "live"
    RETRY_WAIT = "retry_wait"
    STOPPING = "stopping"
    CIRCUIT_OPEN = "circuit_open"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class RefreshStatus:
    phase: RefreshPhase
    message: str
    attempt: int = 0
    max_attempts: int = 3
    retry_in_s: int = 0
    error_code: str = ""


class RefreshLoop(QThread):
    """Perform the initial fetch, then continue refreshing with one owner."""

    frame_ready = pyqtSignal(list)
    status_changed = pyqtSignal(str)
    state_changed = pyqtSignal(object)

    _RETRY_DELAYS_S = (1, 2, 4)
    _MAX_FAILURES = 3
    _FETCH_TIMEOUT_S = 12.0

    def __init__(
        self,
        data_source: DataSource,
        n_bars: int,
        interval_ms: int = 1000,
        cancel_token: "CancelToken | None" = None,
        parent: "QObject | None" = None,
    ) -> None:
        super().__init__(parent)
        self._source = data_source
        self._n_bars = n_bars
        self._interval_ms = interval_ms
        self._cancel_token = cancel_token or threading.Event()
        self._consecutive_failures = 0
        self._initial_fetch_complete = False

    def request_stop(self) -> None:
        if self._cancel_token is not None:
            self._cancel_token.set()
        cancel_pending = getattr(self._source, "cancel_pending", None)
        if callable(cancel_pending):
            cancel_pending()

    def run(self) -> None:
        while not self._cancelled():
            attempt = self._consecutive_failures + 1
            phase_message = (
                f"首次拉取 {attempt}/{self._MAX_FAILURES}"
                if not self._initial_fetch_complete
                else "正在刷新 K 线"
            )
            self._emit_state(
                RefreshStatus(
                    phase=RefreshPhase.CONNECTING,
                    message=phase_message,
                    attempt=attempt,
                    max_attempts=self._MAX_FAILURES,
                )
            )
            started = time.monotonic()
            try:
                bars = self._source.latest_snapshot(
                    self._n_bars + INDICATOR_WARMUP_BARS + 5,
                    cancel_token=self._cancel_token,
                    timeout_s=self._FETCH_TIMEOUT_S,
                )
                if not bars:
                    raise DataSourceEmptyError("数据源未返回 K 线")
            except DataSourceCancelledError:
                break
            except DataSourceInvalidSymbolError as exc:
                self._emit_circuit(str(exc), "invalid_symbol")
                return
            except DataSourceEmptyError as exc:
                if self._handle_retryable_failure(exc, "empty_data"):
                    return
                continue
            except DataSourceTransientError as exc:
                if self._handle_retryable_failure(exc, self._classify_error(exc)):
                    return
                continue
            except Exception as exc:  # noqa: BLE001
                logger.warning("RefreshLoop fetch failed: %s", exc)
                if self._handle_retryable_failure(exc, self._classify_error(exc)):
                    return
                continue

            self._consecutive_failures = 0
            self._initial_fetch_complete = True
            self.frame_ready.emit(bars)
            self._emit_state(
                RefreshStatus(
                    phase=RefreshPhase.LIVE,
                    message="数据已就绪",
                    max_attempts=self._MAX_FAILURES,
                )
            )
            elapsed_ms = (time.monotonic() - started) * 1000
            wait_s = max(0.0, (self._interval_ms - elapsed_ms) / 1000.0)
            if self._wait_cancel(wait_s):
                break

        self._emit_state(
            RefreshStatus(
                phase=RefreshPhase.CANCELLED,
                message="数据刷新已停止",
                max_attempts=self._MAX_FAILURES,
            )
        )

    def _handle_retryable_failure(self, exc: BaseException, error_code: str) -> bool:
        self._consecutive_failures += 1
        message = str(exc).strip() or "数据拉取失败"
        logger.info(
            "RefreshLoop failure %d/%d [%s]: %s",
            self._consecutive_failures,
            self._MAX_FAILURES,
            error_code,
            message,
        )
        if self._consecutive_failures >= self._MAX_FAILURES:
            self._emit_circuit(message, error_code)
            return True

        delay = self._RETRY_DELAYS_S[self._consecutive_failures - 1]
        for remaining in range(delay, 0, -1):
            self._emit_state(
                RefreshStatus(
                    phase=RefreshPhase.RETRY_WAIT,
                    message=f"{message}；{remaining}s 后重试",
                    attempt=self._consecutive_failures + 1,
                    max_attempts=self._MAX_FAILURES,
                    retry_in_s=remaining,
                    error_code=error_code,
                )
            )
            if self._wait_cancel(1.0):
                return True
        return False

    def _emit_circuit(self, message: str, error_code: str) -> None:
        self._emit_state(
            RefreshStatus(
                phase=RefreshPhase.CIRCUIT_OPEN,
                message=message,
                attempt=self._consecutive_failures,
                max_attempts=self._MAX_FAILURES,
                error_code=error_code,
            )
        )

    def _emit_state(self, status: RefreshStatus) -> None:
        self.state_changed.emit(status)
        self.status_changed.emit(status.message)

    def _cancelled(self) -> bool:
        return bool(self._cancel_token is not None and self._cancel_token.is_set())

    def _wait_cancel(self, seconds: float) -> bool:
        if seconds <= 0:
            return self._cancelled()
        if self._cancel_token is not None:
            return self._cancel_token.wait(seconds)
        time.sleep(seconds)
        return False

    @staticmethod
    def _classify_error(exc: BaseException) -> str:
        text = str(exc).lower()
        if "timed out" in text or "timeout" in text or "超时" in text:
            return "timeout"
        if "empty" in text or "no data" in text or "无可用" in text:
            return "empty_data"
        return "network"
