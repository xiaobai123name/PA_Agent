from __future__ import annotations

import math

import pytest

from pa_agent.ai.volume_price_features import compute_volume_price_features
from pa_agent.data.base import IndicatorBundle, KlineBar, KlineFrame, VolumeMeta


def test_volume_meta_rejects_unknown_contract_values() -> None:
    with pytest.raises(ValueError, match="Unsupported volume kind"):
        VolumeMeta("guessed", "test", "base_asset")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="source"):
        VolumeMeta("traded", "", "base_asset")
    with pytest.raises(ValueError, match="unit"):
        VolumeMeta("traded", "test", "")


def _frame(
    *,
    kind: str = "traded",
    latest_volume: float = 200.0,
    latest_bullish: bool = True,
    invalid_index: int | None = None,
    bar_count: int = 25,
    nan_index: int | None = None,
) -> KlineFrame:
    chronological: list[KlineBar] = []
    for i in range(bar_count):
        is_latest = i == bar_count - 1
        volume = latest_volume if is_latest else 100.0
        if invalid_index == i:
            volume = 0.0
        if nan_index == i:
            volume = math.nan
        open_price = 100.0
        close = 102.8 if is_latest and latest_bullish else 97.2 if is_latest else 100.2
        chronological.append(
            KlineBar(
                seq=bar_count - i,
                ts_open=i * 60_000,
                open=open_price,
                high=max(open_price, close) + 0.2,
                low=min(open_price, close) - 0.2,
                close=close,
                volume=volume,
                closed=True,
            )
        )
    bars = tuple(reversed(chronological))
    return KlineFrame(
        symbol="BTCUSDT",
        timeframe="15m",
        volume_meta=VolumeMeta(kind, "test", "base_asset"),
        bars=bars,
        indicators=IndicatorBundle(
            ema20=tuple(100.0 for _ in bars),
            atr14=tuple(2.0 for _ in bars),
        ),
        snapshot_ts_local_ms=2_000_000,
    )


def test_traded_volume_expansion_signal() -> None:
    features = compute_volume_price_features(_frame())
    assert features["status"] == "available"
    assert features["latest_rvol"] == 2.0
    assert features["regime"] == "expanding"
    assert any(x["kind"] == "volume_expansion" for x in features["signals"])


def test_tick_volume_is_explicit_proxy() -> None:
    features = compute_volume_price_features(_frame(kind="tick"))
    assert features["status"] == "available"
    assert features["confidence"] == "tick_proxy"


def test_unknown_and_zero_volume_do_not_emit_signals() -> None:
    unknown = compute_volume_price_features(_frame(kind="unknown"))
    zero = compute_volume_price_features(_frame(invalid_index=20))
    nan = compute_volume_price_features(_frame(nan_index=3))
    short = compute_volume_price_features(_frame(bar_count=20))
    assert unknown["status"] == "unavailable" and unknown["signals"] == []
    assert zero["status"] == "unavailable" and zero["signals"] == []
    assert nan["status"] == "unavailable" and nan["signals"] == []
    assert short["status"] == "unavailable" and short["signals"] == []


def test_low_volume_countertrend_bar_is_pullback_signal() -> None:
    features = compute_volume_price_features(
        _frame(latest_volume=50.0, latest_bullish=False),
        smc_features={"structure_bias": "bullish", "pivots": []},
    )
    assert features["regime"] == "contracting"
    assert any(x["kind"] == "low_volume_pullback" for x in features["signals"])
