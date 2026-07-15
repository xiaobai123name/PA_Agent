from __future__ import annotations

from pa_agent.ai.smc_features import compute_smc_features
from pa_agent.data.base import IndicatorBundle, KlineBar, KlineFrame, VolumeMeta


def _rows() -> list[tuple[float, float, float, float]]:
    rows = [(100.0, 101.0, 99.0, 100.0) for _ in range(25)]
    rows[13] = (100.0, 105.0, 99.0, 101.0)  # confirmed swing high
    rows[16] = (100.0, 101.0, 95.0, 99.0)   # confirmed swing low
    rows[18] = (103.0, 103.0, 99.0, 100.0)  # bullish OB origin
    rows[19] = (104.0, 108.0, 104.0, 107.0) # bullish displacement/BOS + FVG
    rows[20] = (106.0, 109.0, 105.0, 108.0)
    rows[21] = (108.0, 109.0, 100.0, 102.0) # mitigates bullish zones
    rows[22] = (98.0, 99.0, 92.0, 93.0)    # bearish CHoCH + invalidation
    return rows


def _frame(rows: list[tuple[float, float, float, float]]) -> KlineFrame:
    n = len(rows)
    chronological = [
        KlineBar(
            seq=n - i,
            ts_open=i * 60_000,
            open=o,
            high=h,
            low=lo,
            close=c,
            volume=100.0,
            closed=True,
        )
        for i, (o, h, lo, c) in enumerate(rows)
    ]
    bars = tuple(reversed(chronological))
    return KlineFrame(
        symbol="BTCUSDT",
        timeframe="15m",
        volume_meta=VolumeMeta("traded", "test", "base_asset"),
        bars=bars,
        indicators=IndicatorBundle(
            ema20=tuple(100.0 for _ in bars),
            atr14=tuple(2.0 for _ in bars),
        ),
        snapshot_ts_local_ms=2_000_000,
    )


def test_bos_choch_order_block_fvg_and_dealing_range() -> None:
    features = compute_smc_features(_frame(_rows()))
    assert features["status"] == "available"
    assert any(e["kind"] == "bos" and e["direction"] == "bullish" for e in features["events"])
    assert any(e["kind"] == "choch" and e["direction"] == "bearish" for e in features["events"])
    assert any(z["direction"] == "bullish" for z in features["fvgs"])
    assert any(z["direction"] == "bullish" for z in features["order_blocks"])
    assert features["dealing_range"]["zone"] in {"premium", "discount", "equilibrium"}
    assert features["dealing_range"]["id"].startswith("dealing_range:")
    assert all("id" in pivot and "ts_open" in pivot for pivot in features["pivots"])


def test_no_break_event_before_trigger_bar() -> None:
    before = compute_smc_features(_frame(_rows()[:19]))
    after = compute_smc_features(_frame(_rows()[:20]))
    assert not any(
        e["direction"] == "bullish" and e["kind"] in {"bos", "choch"}
        for e in before["events"]
    )
    assert any(e["direction"] == "bullish" and e["kind"] == "bos" for e in after["events"])


def test_liquidity_sweep_requires_close_back_inside() -> None:
    rows = _rows()[:20]
    rows[19] = (104.0, 108.0, 100.0, 104.0)
    features = compute_smc_features(_frame(rows))
    assert any(
        e["kind"] == "liquidity_sweep" and e["direction"] == "bullish"
        for e in features["events"]
    )
    assert not any(e["kind"] == "bos" and e["direction"] == "bullish" for e in features["events"])


def test_sweep_does_not_hide_a_later_close_break() -> None:
    rows = _rows()[:21]
    rows[19] = (104.0, 108.0, 100.0, 104.0)
    rows[20] = (104.0, 109.0, 103.0, 108.0)
    features = compute_smc_features(_frame(rows))
    assert any(e["kind"] == "liquidity_sweep" for e in features["events"])
    assert any(
        e["kind"] == "bos" and e["direction"] == "bullish"
        for e in features["events"]
    )


def test_mirrored_prices_detect_bearish_bos_and_bullish_choch() -> None:
    mirrored = [
        (200.0 - o, 200.0 - low, 200.0 - high, 200.0 - close)
        for o, high, low, close in _rows()
    ]
    features = compute_smc_features(_frame(mirrored))
    assert any(
        e["kind"] == "bos" and e["direction"] == "bearish"
        for e in features["events"]
    )
    assert any(
        e["kind"] == "choch" and e["direction"] == "bullish"
        for e in features["events"]
    )


def test_active_zones_become_invalidated_only_after_later_close() -> None:
    active = compute_smc_features(_frame(_rows()[:21]))
    mitigated = compute_smc_features(_frame(_rows()[:22]))
    final = compute_smc_features(_frame(_rows()[:23]))
    assert any(z["status"] == "active" for z in active["fvgs"] if z["direction"] == "bullish")
    assert any(z["status"] == "mitigated" for z in mitigated["fvgs"] if z["direction"] == "bullish")
    assert any(z["status"] == "mitigated" for z in mitigated["order_blocks"])
    assert any(z["status"] == "invalidated" for z in final["fvgs"] if z["direction"] == "bullish")
    assert any(z["status"] == "invalidated" for z in final["order_blocks"])


def test_premium_and_discount_use_confirmed_dealing_range() -> None:
    premium = compute_smc_features(_frame(_rows()[:20]))
    discount = compute_smc_features(_frame(_rows()[:23]))
    assert premium["dealing_range"]["zone"] == "premium"
    assert discount["dealing_range"]["zone"] == "discount"


def test_insufficient_data_is_explicitly_unavailable() -> None:
    features = compute_smc_features(_frame(_rows()[:14]))
    assert features["status"] == "unavailable"
    assert "至少需要" in features["reason"]
