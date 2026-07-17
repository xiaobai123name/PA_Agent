"""Unit tests for external HTF context (mapping, rendering, fetch orchestration)."""
from __future__ import annotations

from types import SimpleNamespace

from pa_agent.ai.htf_context import (
    HTF_BAR_COUNT,
    build_htf_text,
    fetch_htf_text,
    resolve_htf_timeframe,
)
from pa_agent.data.base import IndicatorBundle, KlineBar, KlineFrame, VolumeMeta

_BASE_TS = 1_600_000_000_000
_STEP_4H = 4 * 3600 * 1000


def _bars(n: int) -> list[KlineBar]:
    out = []
    for i in range(n):  # i=0 newest
        px = 100.0 + (n - i) * 0.5
        out.append(
            KlineBar(
                seq=i + 1,
                ts_open=float(_BASE_TS - i * _STEP_4H),
                open=px - 0.3,
                high=px + 0.6,
                low=px - 0.8,
                close=px,
                volume=10.0,
                closed=True,
            )
        )
    return out


def _frame(n: int = 60) -> KlineFrame:
    bars = tuple(_bars(n))
    return KlineFrame(
        symbol="BTCUSDT",
        timeframe="4h",
        volume_meta=VolumeMeta(kind="unknown", source="test", unit="lot"),
        bars=bars,
        indicators=IndicatorBundle(
            ema20=tuple(float(b.close) - 1.0 for b in bars),
            atr14=tuple(2.0 for _ in bars),
        ),
        snapshot_ts_local_ms=_BASE_TS,
    )


def test_default_map_skips_one_level() -> None:
    assert resolve_htf_timeframe("15m") == "4h"
    assert resolve_htf_timeframe("1h") == "1d"
    assert resolve_htf_timeframe("1w") is None
    assert resolve_htf_timeframe("") is None


def test_override_map_wins_and_same_tf_rejected() -> None:
    assert resolve_htf_timeframe("15m", {"15m": "1h"}) == "1h"
    assert resolve_htf_timeframe("15m", {"15m": "15m"}) is None


def test_supported_filter_falls_back_to_daily() -> None:
    assert resolve_htf_timeframe("15m", None, ["15m", "1h", "1d"]) == "1d"
    assert resolve_htf_timeframe("15m", None, ["15m", "4h"]) == "4h"
    assert resolve_htf_timeframe("15m", None, ["15m"]) is None


def test_build_htf_text_renders_summary() -> None:
    text = build_htf_text(_frame(), "15m")
    assert "外部高周期背景（4h" in text
    assert "上级背景" in text and "15m" in text
    assert "EMA20" in text
    assert "禁止" in text


def test_build_htf_text_needs_enough_bars() -> None:
    assert build_htf_text(_frame(5), "15m") == ""


class _FakeSource:
    def __init__(self, bars: list[KlineBar] | None):
        self._bars = bars
        self.calls: list[tuple[str, str, int]] = []

    def supported_timeframes(self) -> list[str]:
        return ["15m", "1h", "4h", "1d"]

    @property
    def volume_meta(self) -> VolumeMeta:
        return VolumeMeta(kind="unknown", source="test", unit="lot")

    def fetch_frame_once(
        self,
        symbol: str,
        timeframe: str,
        n: int,
        *,
        cancel_token: object | None = None,
        timeout_s: float | None = None,
    ) -> list[KlineBar]:
        self.calls.append((symbol, timeframe, n))
        if self._bars is None:
            raise RuntimeError("boom")
        return list(self._bars[:n])


def _general(**over: object) -> SimpleNamespace:
    base: dict[str, object] = {"htf_context_enabled": True, "htf_timeframe_map": {}}
    base.update(over)
    return SimpleNamespace(**base)


def test_fetch_htf_text_end_to_end() -> None:
    src = _FakeSource(_bars(130))
    text = fetch_htf_text(src, "BTCUSDT", "15m", _general())
    assert "外部高周期背景（4h" in text
    assert src.calls and src.calls[0][:2] == ("BTCUSDT", "4h")
    assert src.calls[0][2] > HTF_BAR_COUNT


def test_fetch_htf_text_disabled_skips_fetch() -> None:
    src = _FakeSource(_bars(130))
    result = fetch_htf_text(src, "BTCUSDT", "15m", _general(htf_context_enabled=False))
    assert result == ""
    assert src.calls == []


def test_fetch_htf_text_empty_bars_degrade() -> None:
    src = _FakeSource([])
    assert fetch_htf_text(src, "BTCUSDT", "15m", _general()) == ""


def test_fetch_htf_text_never_raises() -> None:
    src = _FakeSource(None)
    assert fetch_htf_text(src, "BTCUSDT", "15m", _general()) == ""


def test_record_carries_htf_text() -> None:
    from pa_agent.orchestrator.two_stage import _build_empty_record

    record = _build_empty_record(_frame(), None, htf_text="HTF-BLOCK")
    assert record.htf_text == "HTF-BLOCK"
