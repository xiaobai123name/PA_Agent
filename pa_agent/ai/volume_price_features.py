"""Deterministic relative-volume and price/volume relationship features."""
from __future__ import annotations

import math
import statistics
from typing import Any

from pa_agent.data.base import KlineFrame

VOLUME_FEATURE_VERSION = "volume-price-v1"
_BASELINE_BARS = 20


def _unavailable(frame: KlineFrame, reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": reason,
        "kind": frame.volume_meta.kind,
        "source": frame.volume_meta.source,
        "unit": frame.volume_meta.unit,
        "version": VOLUME_FEATURE_VERSION,
        "signals": [],
    }


def compute_volume_price_features(
    frame: KlineFrame,
    *,
    smc_features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if any(not bar.closed for bar in frame.bars):
        raise ValueError("Volume-price features require a closed-only KlineFrame")
    if frame.volume_meta.kind in ("unknown", "unavailable"):
        return _unavailable(frame, "成交量语义未确认")
    if len(frame.bars) < _BASELINE_BARS + 1:
        return _unavailable(frame, f"至少需要 {_BASELINE_BARS + 1} 根已收盘K线")
    if len(frame.indicators.atr14) != len(frame.bars):
        raise ValueError("ATR14 must align one-to-one with KlineFrame.bars")
    atr_values = list(reversed(frame.indicators.atr14))
    bars = list(reversed(frame.bars))
    volumes = [float(bar.volume) for bar in bars]
    if any(not math.isfinite(v) or v <= 0 for v in volumes):
        return _unavailable(frame, "分析窗口存在零值或无效成交量")

    rvol_by_ts: dict[int, float] = {}
    signals: list[dict[str, Any]] = []
    structure_bias = str((smc_features or {}).get("structure_bias", "neutral"))
    for i in range(_BASELINE_BARS, len(bars)):
        baseline = volumes[i - _BASELINE_BARS : i]
        median = statistics.median(baseline)
        if median <= 0:
            raise ValueError("RVOL baseline median must be positive")
        rvol = volumes[i] / median
        bar = bars[i]
        rvol_by_ts[int(bar.ts_open)] = rvol
        atr = float(atr_values[i]) if i < len(atr_values) else float("nan")
        if not math.isfinite(atr) or atr <= 0:
            continue
        range_atr = (bar.high - bar.low) / atr
        direction = (
            "bullish"
            if bar.close > bar.open
            else "bearish"
            if bar.close < bar.open
            else "neutral"
        )
        if rvol >= 1.5 and range_atr >= 1.2 and direction != "neutral":
            signals.append(
                {
                    "id": f"volume_expansion:{direction}:{int(bar.ts_open)}",
                    "kind": "volume_expansion",
                    "direction": direction,
                    "bar": f"K{bar.seq}",
                    "ts_open": int(bar.ts_open),
                    "rvol": round(rvol, 3),
                    "range_atr": round(range_atr, 3),
                }
            )
        if rvol >= 1.5 and range_atr <= 0.8:
            signals.append(
                {
                    "id": f"effort_result_divergence:{int(bar.ts_open)}",
                    "kind": "high_effort_low_result",
                    "direction": direction,
                    "bar": f"K{bar.seq}",
                    "ts_open": int(bar.ts_open),
                    "rvol": round(rvol, 3),
                    "range_atr": round(range_atr, 3),
                }
            )
        opposing_pullback = (
            structure_bias == "bullish" and direction == "bearish"
        ) or (structure_bias == "bearish" and direction == "bullish")
        if rvol <= 0.7 and opposing_pullback:
            signals.append(
                {
                    "id": f"low_volume_pullback:{direction}:{int(bar.ts_open)}",
                    "kind": "low_volume_pullback",
                    "direction": direction,
                    "bar": f"K{bar.seq}",
                    "ts_open": int(bar.ts_open),
                    "rvol": round(rvol, 3),
                    "range_atr": round(range_atr, 3),
                }
            )

    pivots = (smc_features or {}).get("pivots", [])
    for kind in ("high", "low"):
        same_side = [p for p in pivots if p.get("kind") == kind]
        for previous, current in zip(same_side, same_side[1:], strict=False):
            prev_rvol = rvol_by_ts.get(int(previous["ts_open"]))
            cur_rvol = rvol_by_ts.get(int(current["ts_open"]))
            if prev_rvol is None or cur_rvol is None or cur_rvol > prev_rvol * 0.8:
                continue
            new_extreme = (
                float(current["price"]) > float(previous["price"])
                if kind == "high"
                else float(current["price"]) < float(previous["price"])
            )
            if not new_extreme:
                continue
            direction = "bullish" if kind == "high" else "bearish"
            signals.append(
                {
                    "id": f"swing_volume_divergence:{direction}:{int(current['ts_open'])}",
                    "kind": "swing_volume_divergence",
                    "direction": direction,
                    "bar": current["bar"],
                    "ts_open": int(current["ts_open"]),
                    "rvol": round(cur_rvol, 3),
                    "previous_rvol": round(prev_rvol, 3),
                }
            )

    latest_ts = int(frame.bars[0].ts_open)
    latest_rvol = rvol_by_ts.get(latest_ts)
    if latest_rvol is None:
        return _unavailable(frame, "最新K线缺少完整20棒RVOL基线")
    regime = (
        "expanding"
        if latest_rvol >= 1.5
        else "contracting"
        if latest_rvol <= 0.7
        else "normal"
    )
    return {
        "status": "available",
        "kind": frame.volume_meta.kind,
        "source": frame.volume_meta.source,
        "unit": frame.volume_meta.unit,
        "confidence": "primary" if frame.volume_meta.kind == "traded" else "tick_proxy",
        "version": VOLUME_FEATURE_VERSION,
        "baseline_bars": _BASELINE_BARS,
        "latest_rvol": round(latest_rvol, 3),
        "regime": regime,
        "signals": list(reversed(signals[-8:])),
    }


def render_volume_price_features(features: dict[str, Any]) -> str:
    lines = ["## 程序量价特征（相对量，仅作辅助证据）"]
    if features.get("status") != "available":
        lines.append(
            f"- 状态：不可用；kind={features.get('kind')}；原因：{features.get('reason', '未知')}"
        )
        return "\n".join(lines)
    lines.append(
        f"- kind={features['kind']}，confidence={features['confidence']}，"
        f"K1 RVOL20={features['latest_rvol']}，regime={features['regime']}"
    )
    for signal in features.get("signals", [])[:6]:
        lines.append(
            f"- {signal['kind']} {signal.get('direction', 'neutral')} @ {signal['bar']} "
            f"RVOL={signal['rvol']} id={signal['id']}"
        )
    lines.append("- 量价不得单独改变方向、通过闸门、否决交易或生成三价。")
    return "\n".join(lines)
