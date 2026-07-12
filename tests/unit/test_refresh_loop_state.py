"""RefreshLoop state transitions and retry circuit behaviour."""
from __future__ import annotations

from unittest.mock import MagicMock

from pa_agent.data.base import (
    DataSourceInvalidSymbolError,
    DataSourceTransientError,
    KlineBar,
)
from pa_agent.data.refresh_loop import RefreshLoop, RefreshPhase
from pa_agent.util.threading import CancelToken


def _bars() -> list[KlineBar]:
    return [
        KlineBar(
            seq=1,
            ts_open=1_700_000_000_000,
            open=1.0,
            high=2.0,
            low=0.5,
            close=1.5,
            volume=10.0,
        )
    ]


def _run_direct(loop: RefreshLoop) -> list[object]:
    states: list[object] = []
    loop.state_changed.connect(states.append)
    loop.run()
    return states


def test_first_success_emits_live_and_reuses_same_loop() -> None:
    source = MagicMock()
    token = CancelToken()

    def fetch(*args, **kwargs):
        token.set()
        return _bars()

    source.latest_snapshot.side_effect = fetch
    loop = RefreshLoop(source, n_bars=10, interval_ms=0, cancel_token=token)
    states = _run_direct(loop)

    assert source.latest_snapshot.call_count == 1
    assert [state.phase for state in states] == [
        RefreshPhase.CONNECTING,
        RefreshPhase.LIVE,
        RefreshPhase.CANCELLED,
    ]
    call = source.latest_snapshot.call_args
    assert call.args == (10 + 50 + 5,)
    assert call.kwargs["cancel_token"] is token
    assert call.kwargs["timeout_s"] == 12.0


def test_three_transient_failures_open_circuit() -> None:
    source = MagicMock()
    source.latest_snapshot.side_effect = DataSourceTransientError("network timeout")
    loop = RefreshLoop(source, n_bars=10, cancel_token=CancelToken())
    loop._wait_cancel = lambda _seconds: False

    states = _run_direct(loop)

    assert source.latest_snapshot.call_count == 3
    circuit = [state for state in states if state.phase == RefreshPhase.CIRCUIT_OPEN]
    assert len(circuit) == 1
    assert circuit[0].attempt == 3
    assert circuit[0].error_code == "timeout"


def test_invalid_symbol_opens_circuit_without_retry() -> None:
    source = MagicMock()
    source.latest_snapshot.side_effect = DataSourceInvalidSymbolError("unsupported")
    loop = RefreshLoop(source, n_bars=10, cancel_token=CancelToken())

    states = _run_direct(loop)

    assert source.latest_snapshot.call_count == 1
    assert [state.phase for state in states] == [RefreshPhase.CONNECTING, RefreshPhase.CIRCUIT_OPEN]
