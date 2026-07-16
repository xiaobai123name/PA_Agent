"""Risk/reward and estimated win-rate helpers for trading decisions."""
from __future__ import annotations

import math
from typing import Any


def is_long_direction(direction: object) -> bool | None:
    """Return True for long, False for short, None if unknown."""
    text = str(direction or "").strip().lower()
    if not text:
        return None
    if "多" in text or text in ("long", "buy", "bull"):
        return True
    if "空" in text or text in ("short", "sell", "bear"):
        return False
    return None


def compute_risk_reward(
    entry: object,
    take_profit: object,
    stop_loss: object,
    direction: object,
) -> dict[str, float | str] | None:
    """Compute risk/reward distances and reward:risk ratio (盈亏比).

    Returns None when prices are invalid or risk is zero.
    """
    try:
        e = float(entry)
        tp = float(take_profit)
        sl = float(stop_loss)
    except (TypeError, ValueError):
        return None

    long = is_long_direction(direction)
    if long is True:
        risk = e - sl
        reward = tp - e
    elif long is False:
        risk = sl - e
        reward = e - tp
    else:
        if tp > e and sl < e:
            risk = e - sl
            reward = tp - e
        elif tp < e and sl > e:
            risk = sl - e
            reward = e - tp
        else:
            return None

    if risk <= 0 or reward <= 0:
        return None

    ratio = reward / risk
    return {
        "risk": risk,
        "reward": reward,
        "ratio": ratio,
        "ratio_text": f"{ratio:.2f} : 1",
    }


def format_estimated_win_rate(decision: dict[str, Any]) -> str | None:
    """Format model-provided estimated_win_rate (0–100) for display."""
    value = decision.get("estimated_win_rate")
    if value is None or value == "":
        return None
    try:
        pct = max(0, min(100, int(float(str(value).strip()))))
    except (ValueError, TypeError):
        return None
    return f"{pct}%"


def format_estimated_win_rate_reasoning(decision: dict[str, Any]) -> str:
    return str(decision.get("estimated_win_rate_reasoning", "") or "").strip()


# Lower floor: reward must be at least equal to risk (1:1) for any stance.
MIN_RISK_REWARD_RATIO = 1.0
# Review threshold only. It is not an acceptance cap and never changes a price.
MAX_TP1_RISK_REWARD_RATIO = 1.5
# Stop distance must clear single-bar noise: entry↔stop must be at least this
# fraction of the volatility baseline (ATR14; fallback median closed-bar range).
MIN_STOP_DISTANCE_ATR_FRACTION = 0.35
# Closed bars sampled for the median-range fallback baseline.
_STOP_NOISE_LOOKBACK = 20
_STOP_NOISE_MIN_BARS = 5

_HIGH_RR_REVIEW_FIELDS = (
    "stop_loss_basis",
    "tp1_basis",
    "win_rate_basis",
)


def min_risk_reward_ratio(decision_stance: str | None = None) -> float:
    """Minimum reward:risk ratio required to place an order (same for all stances)."""
    _ = decision_stance  # kept for call-site compatibility
    return MIN_RISK_REWARD_RATIO


def max_risk_reward_ratio() -> float | None:
    """Return the TP1 RR review threshold, not a maximum accepted RR."""
    return MAX_TP1_RISK_REWARD_RATIO


def high_rr_review_required(ratio: float) -> bool:
    """Return whether an order needs explicit structural review evidence."""
    return float(ratio) > MAX_TP1_RISK_REWARD_RATIO + 1e-9


def high_rr_review_is_approved(decision: dict[str, Any]) -> bool:
    """Return whether the decision contains a complete high-RR review."""
    review = decision.get("high_rr_review")
    if not isinstance(review, dict):
        return False
    if str(review.get("status", "")).strip() != "通过":
        return False
    return all(
        isinstance(review.get(field), str) and review[field].strip()
        for field in _HIGH_RR_REVIEW_FIELDS
    )


def validate_high_rr_review(
    decision: dict[str, Any],
    ratio: float,
) -> list[str]:
    """Require explicit stop/TP1/win-rate evidence when RR is unusually high."""
    if not high_rr_review_required(ratio):
        return []

    review = decision.get("high_rr_review")
    if not isinstance(review, dict):
        return [
            "high RR requires decision.high_rr_review with structural stop, TP1, "
            "and win-rate evidence; RR alone does not approve the order"
        ]

    errors: list[str] = []
    status = str(review.get("status", "")).strip()
    if status != "通过":
        errors.append(
            "high RR review is not approved; reject unless structural stop, TP1, "
            "and win-rate evidence are all confirmed"
        )

    labels = {
        "stop_loss_basis": "structural stop",
        "tp1_basis": "TP1 structural target",
        "win_rate_basis": "estimated win-rate",
    }
    for field in _HIGH_RR_REVIEW_FIELDS:
        value = review.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(
                f"high RR review missing {labels[field]} evidence: "
                f"decision.high_rr_review.{field}"
            )
    return errors


def passes_trader_equation(
    win_rate_pct: float,
    risk: float,
    reward: float,
) -> bool:
    """Brooks equation: win_rate × reward > (1 - win_rate) × risk."""
    if risk <= 0 or reward <= 0:
        return False
    p = max(0.0, min(100.0, float(win_rate_pct))) / 100.0
    return p * reward > (1.0 - p) * risk


def _parse_win_rate(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return max(0.0, min(100.0, float(str(value).strip())))
    except (TypeError, ValueError):
        return None


def _latest_closed_bar(kline_frame: Any) -> Any | None:
    """Return K1 (newest closed bar) from a snapshot frame."""
    bars = getattr(kline_frame, "bars", None) if kline_frame is not None else None
    if not bars:
        return None
    for bar in bars:
        if int(getattr(bar, "seq", 0) or 0) == 1 and bool(getattr(bar, "closed", True)):
            return bar
    for bar in bars:
        if bool(getattr(bar, "closed", True)):
            return bar
    return None


def validate_limit_order_k1_freshness(
    decision: dict[str, Any],
    kline_frame: Any,
    *,
    bar_analysis: dict[str, Any] | None = None,
) -> list[str]:
    """Reject stale limit orders that K1 has already traded through."""
    if decision.get("order_type") != "限价单":
        return []

    try:
        entry = float(decision.get("entry_price"))
        sl = float(decision.get("stop_loss_price"))
    except (TypeError, ValueError):
        return []

    bar = _latest_closed_bar(kline_frame)
    if bar is None:
        return []

    from pa_agent.util.price_tick import infer_price_tick_from_frame

    tick = infer_price_tick_from_frame(kline_frame) or 0.0
    k_high = float(bar.high)
    k_low = float(bar.low)
    k_close = float(bar.close)
    long = is_long_direction(decision.get("order_direction"))

    pending_planned = False
    if isinstance(bar_analysis, dict):
        entry_bar = bar_analysis.get("entry_bar")
        if isinstance(entry_bar, dict):
            freshness = str(entry_bar.get("freshness", "") or "").strip().lower()
            strength = str(entry_bar.get("strength", "") or "").strip().lower()
            pending_planned = (
                freshness == "pending"
                or strength == "not_triggered"
                or entry_bar.get("bar") is None
            )

    errors: list[str] = []
    if long is True:
        if pending_planned:
            # Planned buy limit: entry must stay below market close (waiting for dip).
            if k_close < entry - tick:
                errors.append(
                    f"limit long (planned): K1 close {k_close:.6g} is below entry {entry:.6g}; "
                    "reprice entry or 不下单"
                )
        else:
            if k_low <= entry + tick:
                errors.append(
                    f"limit long: K1 low {k_low:.6g} already touched/below entry {entry:.6g}; "
                    "pending buy limit is stale — use 市价单, reprice, or 不下单"
                )
            if k_close < entry - tick:
                errors.append(
                    f"limit long: K1 close {k_close:.6g} is below entry {entry:.6g}; "
                    "do not keep a buy limit above market without repricing"
                )
        if k_low <= sl + tick:
            errors.append(
                f"limit long: K1 low {k_low:.6g} already at/below stop {sl:.6g}; "
                "plan invalid — order_type=不下单"
            )
    elif long is False:
        if pending_planned:
            if k_close > entry + tick:
                errors.append(
                    f"limit short (planned): K1 close {k_close:.6g} is above entry {entry:.6g}; "
                    "reprice entry or 不下单"
                )
        else:
            if k_high >= entry - tick:
                errors.append(
                    f"limit short: K1 high {k_high:.6g} already reached/exceeded entry {entry:.6g}; "
                    "pending sell limit is stale — use 市价单, reprice, or 不下单"
                )
            if k_close > entry + tick:
                errors.append(
                    f"limit short: K1 close {k_close:.6g} is above entry {entry:.6g}; "
                    "do not keep a sell limit below market without repricing"
                )
        if k_high >= sl - tick:
            errors.append(
                f"limit short: K1 high {k_high:.6g} already at/above stop {sl:.6g}; "
                "plan invalid — order_type=不下单"
            )

    return errors


def _volatility_baseline(kline_frame: Any) -> float | None:
    """Recent volatility baseline: ATR14 when finite, else median closed-bar range."""
    indicators = getattr(kline_frame, "indicators", None) if kline_frame is not None else None
    atr14 = getattr(indicators, "atr14", None) if indicators is not None else None
    if atr14:
        for raw in atr14[:2]:
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(val) and val > 0:
                return val

    bars = getattr(kline_frame, "bars", None) if kline_frame is not None else None
    if not bars:
        return None
    ranges: list[float] = []
    for bar in bars:
        if not bool(getattr(bar, "closed", True)):
            continue
        try:
            span = float(bar.high) - float(bar.low)
        except (TypeError, ValueError, AttributeError):
            continue
        if span > 0:
            ranges.append(span)
        if len(ranges) >= _STOP_NOISE_LOOKBACK:
            break
    if len(ranges) < _STOP_NOISE_MIN_BARS:
        return None
    ranges.sort()
    mid = len(ranges) // 2
    if len(ranges) % 2:
        return ranges[mid]
    return (ranges[mid - 1] + ranges[mid]) / 2.0


def validate_stop_distance_floor(
    decision: dict[str, Any],
    kline_frame: Any,
) -> list[str]:
    """Reject noise stops: entry↔stop distance below a fraction of recent volatility.

    A structurally correct invalidation level still yields a noise stop when the
    entry is placed on top of it; the floor guards distance, not the level.
    """
    try:
        entry = float(decision.get("entry_price"))
        sl = float(decision.get("stop_loss_price"))
    except (TypeError, ValueError):
        return []

    distance = abs(entry - sl)
    if distance <= 0:
        return []
    baseline = _volatility_baseline(kline_frame)
    if baseline is None:
        return []
    floor = baseline * MIN_STOP_DISTANCE_ATR_FRACTION
    if distance >= floor:
        return []
    return [
        f"decision.stop_loss_price: stop distance {distance:.6g} is below the noise "
        f"floor {floor:.6g} ({MIN_STOP_DISTANCE_ATR_FRACTION:.0%} of recent volatility "
        f"{baseline:.6g}); entry sits too close to the invalidation level — wait for a "
        "deeper pullback entry, use a wider structural stop (swing extreme), or set "
        "order_type=不下单; prices are not auto-adjusted"
    ]


def validate_take_profit_2_geometry(
    decision: dict[str, Any],
) -> list[str]:
    """Ensure TP2 is beyond TP1 in the profit direction (no RR cap on TP2)."""
    entry = decision.get("entry_price")
    tp1 = decision.get("take_profit_price")
    tp2 = decision.get("take_profit_price_2")
    sl = decision.get("stop_loss_price")
    direction = decision.get("order_direction")

    try:
        e = float(entry)
        t1 = float(tp1)
        t2 = float(tp2)
        s = float(sl)
    except (TypeError, ValueError):
        return ["decision.take_profit_price_2: required finite number when placing an order"]

    long = is_long_direction(direction)
    if long is True:
        if not (s < e < t1 < t2):
            return [
                "decision.take_profit_price_2: long plan requires "
                "stop < entry < take_profit_price < take_profit_price_2"
            ]
    elif long is False:
        if not (t2 < t1 < e < s):
            return [
                "decision.take_profit_price_2: short plan requires "
                "take_profit_price_2 < take_profit_price < entry < stop"
            ]
    else:
        if t1 > e and t2 <= t1:
            return [
                "decision.take_profit_price_2: must be above take_profit_price for long geometry"
            ]
        if t1 < e and t2 >= t1:
            return [
                "decision.take_profit_price_2: must be below take_profit_price for short geometry"
            ]

    return []


def validate_order_trade_metrics(
    decision: dict[str, Any],
    *,
    decision_stance: str | None = None,
    kline_frame: Any = None,
    bar_analysis: dict[str, Any] | None = None,
) -> list[str]:
    """Validate trade geometry, RR floor, stop noise floor, review, and trader equation.

    The stop is a structural invalidation price supplied by the decision. RR never
    rewrites entry, TP1, TP2, or stop. High RR is allowed after explicit review.
    """
    order_type = decision.get("order_type")
    if order_type not in ("限价单", "突破单", "市价单"):
        return []

    entry = decision.get("entry_price")
    tp = decision.get("take_profit_price")
    sl = decision.get("stop_loss_price")
    direction = decision.get("order_direction")
    rr = compute_risk_reward(entry, tp, sl, direction)
    if rr is None:
        return [
            "decision prices: entry/stop/target must form a valid long (sl<entry<tp) "
            "or short (tp<entry<sl) trade with positive risk and reward"
        ]

    errors: list[str] = []
    ratio = float(rr["ratio"])
    risk = float(rr["risk"])
    reward = float(rr["reward"])
    min_rr = min_risk_reward_ratio(decision_stance)

    if ratio < min_rr:
        errors.append(
            f"decision prices: risk_reward {rr['ratio_text']} is below minimum "
            f"{min_rr:.2f}:1 for this stance; re-review the structural setup or set "
            "order_type=不下单 with 10.3=否; stop_loss_price is not auto-adjusted"
        )

    errors.extend(validate_high_rr_review(decision, ratio))

    win_rate = _parse_win_rate(decision.get("estimated_win_rate"))
    if win_rate is None:
        errors.append(
            "decision.estimated_win_rate: required integer 0–100 when placing an order"
        )
    elif not passes_trader_equation(win_rate, risk, reward):
        ev = win_rate / 100.0 * reward - (1.0 - win_rate / 100.0) * risk
        errors.append(
            f"decision prices: trader equation fails at {win_rate:.0f}% win rate "
            f"(risk={risk:.4g}, reward={reward:.4g}, expectancy≈{ev:.4g}); "
            "10.3 must be 否 and order_type=不下单 unless prices are fixed"
        )

    if kline_frame is not None:
        errors.extend(
            validate_limit_order_k1_freshness(
                decision, kline_frame, bar_analysis=bar_analysis
            )
        )
        errors.extend(validate_stop_distance_floor(decision, kline_frame))

    errors.extend(validate_take_profit_2_geometry(decision))

    return errors
