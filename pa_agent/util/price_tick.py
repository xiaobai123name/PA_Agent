"""Infer minimum price increment from K-line OHLC precision."""
from __future__ import annotations

import re
from typing import Any


def infer_price_tick_from_frame(kline_frame: Any) -> float | None:
    """Guess one tick from decimal places in the snapshot (e.g. XAU 0.01 or 0.001)."""
    bars = getattr(kline_frame, "bars", None) if kline_frame is not None else None
    if not bars:
        return None

    max_decimals = 0
    for bar in bars:
        for attr in ("open", "high", "low", "close"):
            try:
                value = float(getattr(bar, attr))
            except (TypeError, ValueError):
                continue
            text = f"{value:.12f}".rstrip("0")
            if "." in text:
                max_decimals = max(max_decimals, len(text.split(".")[1]))

    if max_decimals <= 0:
        return 1.0
    return 10 ** (-min(max_decimals, 6))


def round_to_tick(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    return round(round(price / tick) * tick, 10)


def parse_k_seq(value: object) -> int | None:
    if value is None:
        return None
    m = re.search(r"K\s*(\d+)", str(value), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def bar_by_seq(kline_frame: Any, seq: int) -> Any | None:
    for bar in getattr(kline_frame, "bars", ()) or ():
        if getattr(bar, "seq", None) == seq:
            return bar
    return None


def breakout_entry_target(
    *,
    direction: str,
    extreme: str,
    basis_high: float,
    basis_low: float,
    tick: float,
) -> float | None:
    """Return the minimum valid breakout entry (strictly outside the cited extreme)."""
    if direction == "做多" and extreme == "high":
        return round_to_tick(basis_high + tick, tick)
    if direction == "做空" and extreme == "low":
        return round_to_tick(basis_low - tick, tick)
    return None


def format_breakout_tick_hint(kline_frame: Any) -> str:
    """One-line Stage-2 user hint with inferred tick and formula."""
    tick = infer_price_tick_from_frame(kline_frame)
    if tick is None:
        return ""
    tick_s = f"{tick:g}"
    return (
        f"**突破单定价（程序推断最小跳动 ≈ {tick_s}）**：做多时 "
        f"`entry_price` 必须 **严格大于** `entry_basis_bar` 的 high，"
        f"推荐 `entry_price = 该 K 线 high + {tick_s}`（禁止等于 high）；"
        f"做空时 `entry_price` 必须 **严格低于** low，推荐 `low - {tick_s}`。"
        f"`entry_rule` 必须写明：`K{{n}} low/high = {{实际价格}}，entry = {{实际价格}} ± {tick_s}`，"
        f"勿重复 order_type/方向长句。"
        f"**程序只校验，不会修改 entry_price；数值与依据 K 线不一致时直接拒绝。**"
    )
