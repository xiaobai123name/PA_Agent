"""Deterministic single-timeframe Smart Money Concepts features."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from pa_agent.data.base import KlineBar, KlineFrame

SMC_FEATURE_VERSION = "smc-v1"
_PIVOT_SPAN = 2
_MIN_BARS = 15


@dataclass(frozen=True)
class _Pivot:
    index: int
    seq: int
    ts_open: int
    kind: str
    price: float


def _event_id(kind: str, direction: str, ts_open: int) -> str:
    return f"{kind}:{direction}:{ts_open}"


def _atr_series(frame: KlineFrame) -> list[float] | None:
    if len(frame.indicators.atr14) != len(frame.bars):
        raise ValueError("ATR14 must align one-to-one with KlineFrame.bars")
    if not frame.indicators.atr14:
        return None
    values = [float(value) for value in reversed(frame.indicators.atr14)]
    if any(not math.isfinite(value) or value <= 0 for value in values):
        return None
    return values


def _confirmed_pivots(bars: list[KlineBar]) -> list[_Pivot]:
    pivots: list[_Pivot] = []
    for i in range(_PIVOT_SPAN, len(bars) - _PIVOT_SPAN):
        bar = bars[i]
        neighbors = bars[i - _PIVOT_SPAN : i] + bars[i + 1 : i + _PIVOT_SPAN + 1]
        if all(bar.high > other.high for other in neighbors):
            pivots.append(_Pivot(i, int(bar.seq), int(bar.ts_open), "high", float(bar.high)))
        if all(bar.low < other.low for other in neighbors):
            pivots.append(_Pivot(i, int(bar.seq), int(bar.ts_open), "low", float(bar.low)))
    return pivots


def _zone_status(
    bars: list[KlineBar],
    *,
    created_index: int,
    direction: str,
    lower: float,
    upper: float,
) -> str:
    touched = False
    for bar in bars[created_index + 1 :]:
        if direction == "bullish" and bar.close < lower:
            return "invalidated"
        if direction == "bearish" and bar.close > upper:
            return "invalidated"
        if bar.high >= lower and bar.low <= upper:
            touched = True
    return "mitigated" if touched else "active"


def compute_smc_features(frame: KlineFrame) -> dict[str, Any]:
    """Return non-repainting SMC facts computed from closed bars only."""
    if any(not bar.closed for bar in frame.bars):
        raise ValueError("SMC features require a closed-only KlineFrame")
    if len(frame.bars) < _MIN_BARS:
        return {
            "status": "unavailable",
            "reason": f"至少需要 {_MIN_BARS} 根已收盘K线",
            "version": SMC_FEATURE_VERSION,
        }
    atr_values = _atr_series(frame)
    if atr_values is None:
        return {
            "status": "unavailable",
            "reason": "ATR14 不可用",
            "version": SMC_FEATURE_VERSION,
        }

    bars = list(reversed(frame.bars))
    pivots = _confirmed_pivots(bars)
    pivots_by_confirmation: dict[int, list[_Pivot]] = {}
    for pivot in pivots:
        pivots_by_confirmation.setdefault(pivot.index + _PIVOT_SPAN, []).append(pivot)

    latest_high: _Pivot | None = None
    latest_low: _Pivot | None = None
    consumed: set[tuple[str, int]] = set()
    bias = "neutral"
    events: list[dict[str, Any]] = []
    order_blocks: list[dict[str, Any]] = []

    for i, bar in enumerate(bars):
        for pivot in pivots_by_confirmation.get(i, []):
            if pivot.kind == "high":
                latest_high = pivot
            else:
                latest_low = pivot

        for direction, pivot in (("bullish", latest_high), ("bearish", latest_low)):
            if pivot is None or i <= pivot.index + _PIVOT_SPAN:
                continue
            key = (pivot.kind, pivot.ts_open)
            if key in consumed:
                continue
            atr = atr_values[i]
            epsilon = 0.05 * atr
            broke = (
                bar.close > pivot.price + epsilon
                if direction == "bullish"
                else bar.close < pivot.price - epsilon
            )
            swept = (
                bar.high > pivot.price + epsilon and bar.close <= pivot.price
                if direction == "bullish"
                else bar.low < pivot.price - epsilon and bar.close >= pivot.price
            )
            if swept:
                events.append(
                    {
                        "id": _event_id("sweep", direction, int(bar.ts_open)),
                        "kind": "liquidity_sweep",
                        "direction": direction,
                        "level": round(pivot.price, 8),
                        "bar": f"K{bar.seq}",
                        "ts_open": int(bar.ts_open),
                        "pivot_bar": f"K{pivot.seq}",
                    }
                )
                continue
            if not broke:
                continue

            kind = "choch" if bias not in ("neutral", direction) else "bos"
            events.append(
                {
                    "id": _event_id(kind, direction, int(bar.ts_open)),
                    "kind": kind,
                    "direction": direction,
                    "level": round(pivot.price, 8),
                    "bar": f"K{bar.seq}",
                    "ts_open": int(bar.ts_open),
                    "pivot_bar": f"K{pivot.seq}",
                }
            )
            consumed.add(key)
            bias = direction

            displacement = (bar.high - bar.low) / atr
            if kind != "bos" or displacement < 1.2:
                continue
            opposing_index: int | None = None
            for candidate_index in range(i - 1, -1, -1):
                candidate = bars[candidate_index]
                opposing = (
                    candidate.close < candidate.open
                    if direction == "bullish"
                    else candidate.close > candidate.open
                )
                if opposing:
                    opposing_index = candidate_index
                    break
            if opposing_index is None:
                continue
            ob_bar = bars[opposing_index]
            lower = float(ob_bar.low)
            upper = float(ob_bar.high)
            order_blocks.append(
                {
                    "id": (
                        f"ob:{direction}:{int(ob_bar.ts_open)}:"
                        f"{int(bar.ts_open)}"
                    ),
                    "direction": direction,
                    "lower": round(lower, 8),
                    "upper": round(upper, 8),
                    "origin_bar": f"K{ob_bar.seq}",
                    "origin_ts": int(ob_bar.ts_open),
                    "bos_event_id": _event_id(kind, direction, int(bar.ts_open)),
                    "displacement_atr": round(displacement, 3),
                    "status": _zone_status(
                        bars,
                        created_index=i,
                        direction=direction,
                        lower=lower,
                        upper=upper,
                    ),
                }
            )

    fvgs: list[dict[str, Any]] = []
    for i in range(2, len(bars)):
        older = bars[i - 2]
        current = bars[i]
        min_gap = 0.1 * atr_values[i]
        direction = ""
        lower = upper = 0.0
        if current.low - older.high >= min_gap:
            direction = "bullish"
            lower, upper = float(older.high), float(current.low)
        elif older.low - current.high >= min_gap:
            direction = "bearish"
            lower, upper = float(current.high), float(older.low)
        if not direction:
            continue
        status = _zone_status(
            bars,
            created_index=i,
            direction=direction,
            lower=lower,
            upper=upper,
        )
        fvgs.append(
            {
                "id": _event_id("fvg", direction, int(current.ts_open)),
                "direction": direction,
                "lower": round(lower, 8),
                "upper": round(upper, 8),
                "created_bar": f"K{current.seq}",
                "created_ts": int(current.ts_open),
                "status": status,
            }
        )

    latest_high_pivot = next((p for p in reversed(pivots) if p.kind == "high"), None)
    latest_low_pivot = next((p for p in reversed(pivots) if p.kind == "low"), None)
    dealing_range: dict[str, Any] | None = None
    if latest_high_pivot and latest_low_pivot and latest_low_pivot.price < latest_high_pivot.price:
        equilibrium = (latest_high_pivot.price + latest_low_pivot.price) / 2
        close = float(frame.bars[0].close)
        dealing_range = {
            "id": (
                f"dealing_range:{latest_low_pivot.ts_open}:"
                f"{latest_high_pivot.ts_open}"
            ),
            "low": round(latest_low_pivot.price, 8),
            "high": round(latest_high_pivot.price, 8),
            "equilibrium": round(equilibrium, 8),
            "zone": (
                "premium"
                if close > equilibrium
                else "discount"
                if close < equilibrium
                else "equilibrium"
            ),
            "high_pivot_bar": f"K{latest_high_pivot.seq}",
            "low_pivot_bar": f"K{latest_low_pivot.seq}",
            "high_pivot_ts": latest_high_pivot.ts_open,
            "low_pivot_ts": latest_low_pivot.ts_open,
        }

    return {
        "status": "available",
        "version": SMC_FEATURE_VERSION,
        "structure_bias": bias,
        "pivots": [
            {
                "id": _event_id("pivot", p.kind, p.ts_open),
                "bar": f"K{p.seq}",
                "ts_open": p.ts_open,
                "kind": p.kind,
                "price": round(p.price, 8),
            }
            for p in pivots[-12:]
        ],
        "events": list(reversed(events[-8:])),
        "fvgs": list(reversed(fvgs[-8:])),
        "order_blocks": list(reversed(order_blocks[-6:])),
        "dealing_range": dealing_range,
    }


def render_smc_features(features: dict[str, Any]) -> str:
    lines = ["## 程序 SMC 特征（闭合K线、无未来函数）"]
    if features.get("status") != "available":
        lines.append(f"- 状态：不可用；原因：{features.get('reason', '未知')}")
        return "\n".join(lines)
    lines.append(f"- 当前结构偏向：**{features['structure_bias']}**")
    for event in features.get("events", [])[:5]:
        lines.append(
            f"- {event['kind'].upper()} {event['direction']} @ {event['bar']} "
            f"level={event['level']} id={event['id']}"
        )
    active_fvgs = [x for x in features.get("fvgs", []) if x.get("status") != "invalidated"]
    active_obs = [x for x in features.get("order_blocks", []) if x.get("status") != "invalidated"]
    for zone in active_fvgs[:3]:
        lines.append(
            f"- FVG {zone['direction']} [{zone['lower']}, {zone['upper']}] "
            f"status={zone['status']} id={zone['id']}"
        )
    for zone in active_obs[:2]:
        lines.append(
            f"- OB {zone['direction']} [{zone['lower']}, {zone['upper']}] "
            f"status={zone['status']} id={zone['id']}"
        )
    dealing = features.get("dealing_range")
    if dealing:
        lines.append(
            f"- Dealing Range [{dealing['low']}, {dealing['high']}]，"
            f"EQ={dealing['equilibrium']}，当前={dealing['zone']}"
        )
    return "\n".join(lines)
