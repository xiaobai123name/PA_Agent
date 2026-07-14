"""Resolve an AI trade intent into one deterministic execution method."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import time
from typing import Any

from pa_agent.data.live_quote import LiveQuote
from pa_agent.util.price_tick import infer_price_tick_from_frame


_INTENT_TO_ORDER_TYPE = {
    "pullback": "限价单",
    "breakout": "突破单",
    "immediate": "市价单",
    "none": "不下单",
}

_ORDER_TYPE_TO_NODE = {
    "市价单": "11.1",
    "突破单": "11.2",
    "限价单": "11.3",
}

_PRICE_FIELDS = (
    "order_direction",
    "entry_price",
    "take_profit_price",
    "take_profit_price_2",
    "stop_loss_price",
    "entry_basis_bar",
    "entry_basis_extreme",
    "entry_rule",
)

_PROPOSED_STRUCTURE_FIELDS = (
    "order_direction",
    "entry_price",
    "take_profit_price",
    "take_profit_price_2",
    "stop_loss_price",
    "entry_basis_bar",
    "entry_basis_extreme",
    "entry_rule",
)


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    quote_max_age_ms: int = 3000
    immediate_max_slippage_atr: float = 0.10
    immediate_max_slippage_ticks: int = 3


def execution_policy_from_settings(settings: Any) -> ExecutionPolicy:
    general = getattr(settings, "general", None)
    return ExecutionPolicy(
        quote_max_age_ms=int(getattr(general, "execution_quote_max_age_ms", 3000)),
        immediate_max_slippage_atr=float(
            getattr(general, "execution_max_slippage_atr", 0.10)
        ),
        immediate_max_slippage_ticks=int(
            getattr(general, "execution_max_slippage_ticks", 3)
        ),
    )


def resolve_stage2_execution(
    stage2: dict[str, Any],
    *,
    frame: Any,
    quote: LiveQuote | None,
    policy: ExecutionPolicy | None = None,
    now_ms: int | None = None,
) -> dict[str, Any]:
    """Return a resolved copy; invalid execution conditions become explicit rejects."""
    out = copy.deepcopy(stage2)
    decision = out.get("decision")
    if not isinstance(decision, dict):
        raise ValueError("stage2.decision must be an object")

    policy = policy or ExecutionPolicy()
    intent = str(decision.get("entry_intent") or "").strip().lower()
    if intent not in _INTENT_TO_ORDER_TYPE:
        raise ValueError(
            "decision.entry_intent must be one of pullback, breakout, immediate, none"
        )

    proposed_type = str(decision.get("order_type") or "").strip()
    expected_type = _INTENT_TO_ORDER_TYPE[intent]
    proposed_entry = decision.get("entry_price")

    if intent == "none":
        if proposed_type != "不下单":
            return _reject(
                out,
                intent=intent,
                proposed_type=proposed_type,
                proposed_entry=proposed_entry,
                quote=quote,
                reason_code="declared_order_type_mismatch",
                reason="entry_intent=none 时 order_type 必须为不下单",
                now_ms=now_ms,
            )
        decision["execution_review"] = {
            "status": "not_applicable",
            "reason_code": "no_trade_intent",
            "reason": "阶段二没有交易意图",
            "proposed_entry_intent": intent,
            "proposed_order_type": proposed_type,
            "proposed_entry_price": None,
            "resolved_order_type": "不下单",
            "market_price": _quote_price_or_none(quote),
            "quote_timestamp_ms": quote.received_at_ms if quote is not None else None,
            "quote_age_ms": _quote_age_ms(quote, now_ms),
            "max_slippage": None,
        }
        _remove_execution_nodes(out)
        return out

    if proposed_type != expected_type:
        return _reject(
            out,
            intent=intent,
            proposed_type=proposed_type,
            proposed_entry=proposed_entry,
            quote=quote,
            reason_code="declared_order_type_mismatch",
            reason=f"entry_intent={intent} 必须声明为 {expected_type}，实际为 {proposed_type or '空'}",
            now_ms=now_ms,
        )

    if quote is None:
        return _reject(
            out,
            intent=intent,
            proposed_type=proposed_type,
            proposed_entry=proposed_entry,
            quote=None,
            reason_code="live_quote_unavailable",
            reason="没有当前品种和周期的形成中 K 线报价",
            now_ms=now_ms,
        )

    if quote.symbol.strip().upper() != str(frame.symbol).strip().upper() or (
        quote.timeframe.strip().lower() != str(frame.timeframe).strip().lower()
    ):
        return _reject(
            out,
            intent=intent,
            proposed_type=proposed_type,
            proposed_entry=proposed_entry,
            quote=quote,
            reason_code="live_quote_identity_mismatch",
            reason=(
                "实时报价与分析标的不一致："
                f"quote={quote.symbol} {quote.timeframe}, "
                f"frame={frame.symbol} {frame.timeframe}"
            ),
            now_ms=now_ms,
        )

    age_ms = _quote_age_ms(quote, now_ms)
    if age_ms is None or age_ms > policy.quote_max_age_ms:
        return _reject(
            out,
            intent=intent,
            proposed_type=proposed_type,
            proposed_entry=proposed_entry,
            quote=quote,
            reason_code="stale_live_quote",
            reason=f"实时报价已过期：{age_ms}ms > {policy.quote_max_age_ms}ms",
            now_ms=now_ms,
        )

    try:
        entry = float(proposed_entry)
    except (TypeError, ValueError):
        return _reject(
            out,
            intent=intent,
            proposed_type=proposed_type,
            proposed_entry=proposed_entry,
            quote=quote,
            reason_code="invalid_entry_price",
            reason="交易意图缺少有效 entry_price",
            now_ms=now_ms,
        )
    if not math.isfinite(entry) or entry <= 0:
        return _reject(
            out,
            intent=intent,
            proposed_type=proposed_type,
            proposed_entry=proposed_entry,
            quote=quote,
            reason_code="invalid_entry_price",
            reason=f"entry_price 不是有效正数：{proposed_entry!r}",
            now_ms=now_ms,
        )

    tick = infer_price_tick_from_frame(frame)
    if tick is None or not math.isfinite(float(tick)) or float(tick) <= 0:
        return _reject(
            out,
            intent=intent,
            proposed_type=proposed_type,
            proposed_entry=entry,
            quote=quote,
            reason_code="tick_unavailable",
            reason="无法从 K 线数据确定最小价格跳动",
            now_ms=now_ms,
        )
    tick = float(tick)

    direction = str(decision.get("order_direction") or "").strip()
    if direction not in ("做多", "做空"):
        return _reject(
            out,
            intent=intent,
            proposed_type=proposed_type,
            proposed_entry=entry,
            quote=quote,
            reason_code="invalid_order_direction",
            reason="交易意图缺少做多或做空方向",
            now_ms=now_ms,
        )

    try:
        market = float(quote.last_price)
    except (TypeError, ValueError):
        return _reject(
            out,
            intent=intent,
            proposed_type=proposed_type,
            proposed_entry=entry,
            quote=quote,
            reason_code="invalid_live_quote",
            reason=f"实时报价不是数值：{quote.last_price!r}",
            now_ms=now_ms,
        )
    if not math.isfinite(market) or market <= 0:
        return _reject(
            out,
            intent=intent,
            proposed_type=proposed_type,
            proposed_entry=entry,
            quote=quote,
            reason_code="invalid_live_quote",
            reason=f"实时报价不是有效正数：{quote.last_price!r}",
            now_ms=now_ms,
        )
    max_slippage: float | None = None
    relation_error = _validate_price_relation(
        intent=intent,
        direction=direction,
        entry=entry,
        market=market,
        tick=tick,
    )
    if relation_error is not None:
        code, reason = relation_error
        return _reject(
            out,
            intent=intent,
            proposed_type=proposed_type,
            proposed_entry=entry,
            quote=quote,
            reason_code=code,
            reason=reason,
            now_ms=now_ms,
        )

    if intent == "immediate":
        atr = _newest_atr(frame)
        atr_limit = (
            atr * policy.immediate_max_slippage_atr
            if atr is not None
            else 0.0
        )
        max_slippage = max(
            tick * policy.immediate_max_slippage_ticks,
            atr_limit,
        )
        distance = abs(market - entry)
        if distance > max_slippage:
            return _reject(
                out,
                intent=intent,
                proposed_type=proposed_type,
                proposed_entry=entry,
                quote=quote,
                reason_code="immediate_entry_missed",
                reason=(
                    f"即时入场已错过：当前价与计划入场相差 {distance:.8g}，"
                    f"允许偏差 {max_slippage:.8g}"
                ),
                now_ms=now_ms,
                max_slippage=max_slippage,
            )

    decision["execution_review"] = {
        "status": "resolved",
        "reason_code": "execution_method_resolved",
        "reason": _resolved_reason(intent, direction, entry, market),
        "proposed_entry_intent": intent,
        "proposed_order_type": proposed_type,
        "proposed_entry_price": entry,
        "resolved_order_type": expected_type,
        "market_price": market,
        "quote_timestamp_ms": quote.received_at_ms,
        "quote_age_ms": age_ms,
        "max_slippage": max_slippage,
    }
    _set_execution_node(out, expected_type, decision["execution_review"]["reason"], True)
    terminal = out.setdefault("terminal", {})
    terminal["node_id"] = _ORDER_TYPE_TO_NODE[expected_type]
    terminal["outcome"] = "trade"
    terminal["label"] = f"{expected_type}执行条件已确认"
    return out


def _validate_price_relation(
    *,
    intent: str,
    direction: str,
    entry: float,
    market: float,
    tick: float,
) -> tuple[str, str] | None:
    if intent == "pullback":
        valid = entry <= market - tick if direction == "做多" else entry >= market + tick
        if not valid:
            return (
                "pullback_entry_not_pending",
                f"回撤入场价不在当前价的有利一侧：entry={entry:.8g}, market={market:.8g}",
            )
    elif intent == "breakout":
        valid = entry >= market + tick if direction == "做多" else entry <= market - tick
        if not valid:
            return (
                "breakout_trigger_already_crossed",
                f"突破触发位已经越过或位于错误一侧：entry={entry:.8g}, market={market:.8g}",
            )
    return None


def _newest_atr(frame: Any) -> float | None:
    values = getattr(getattr(frame, "indicators", None), "atr14", ()) or ()
    if not values:
        return None
    try:
        value = float(values[0])
    except (TypeError, ValueError, IndexError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _quote_age_ms(quote: LiveQuote | None, now_ms: int | None) -> int | None:
    if quote is None:
        return None
    current = int(time.time() * 1000 if now_ms is None else now_ms)
    return max(0, current - int(quote.received_at_ms))


def _quote_price_or_none(quote: LiveQuote | None) -> float | None:
    if quote is None:
        return None
    try:
        price = float(quote.last_price)
    except (TypeError, ValueError):
        return None
    return price if math.isfinite(price) and price > 0 else None


def _resolved_reason(intent: str, direction: str, entry: float, market: float) -> str:
    labels = {
        "pullback": "结构回撤尚未到价，使用限价单等待",
        "breakout": "结构突破尚未触发，使用突破单等待确认",
        "immediate": "已收盘确认且当前报价仍在允许偏差内，允许立即入场",
    }
    return f"{labels[intent]}；{direction} entry={entry:.8g}, market={market:.8g}"


def _reject(
    out: dict[str, Any],
    *,
    intent: str,
    proposed_type: str,
    proposed_entry: Any,
    quote: LiveQuote | None,
    reason_code: str,
    reason: str,
    now_ms: int | None,
    max_slippage: float | None = None,
) -> dict[str, Any]:
    decision = out["decision"]
    proposed_structure = {
        field: copy.deepcopy(decision.get(field))
        for field in _PROPOSED_STRUCTURE_FIELDS
    }
    decision["execution_review"] = {
        "status": "rejected",
        "reason_code": reason_code,
        "reason": reason,
        "proposed_entry_intent": intent,
        "proposed_order_type": proposed_type or None,
        "proposed_entry_price": proposed_entry,
        "resolved_order_type": "不下单",
        "market_price": _quote_price_or_none(quote),
        "quote_timestamp_ms": quote.received_at_ms if quote is not None else None,
        "quote_age_ms": _quote_age_ms(quote, now_ms),
        "max_slippage": max_slippage,
        "proposed_structure": proposed_structure,
    }
    decision["order_type"] = "不下单"
    for field in _PRICE_FIELDS:
        decision[field] = None
    decision["estimated_win_rate"] = None
    decision["estimated_win_rate_reasoning"] = None
    decision["high_rr_review"] = None
    decision["reasoning"] = f"执行解析拒绝：{reason}"
    node_type = _INTENT_TO_ORDER_TYPE.get(intent)
    _set_execution_node(out, node_type, reason, False)
    terminal = out.setdefault("terminal", {})
    terminal["node_id"] = _ORDER_TYPE_TO_NODE.get(node_type, "11.execution")
    terminal["outcome"] = "reject"
    terminal["label"] = f"执行条件拒绝：{reason_code}"
    return out


def _remove_execution_nodes(out: dict[str, Any]) -> None:
    trace = out.get("decision_trace")
    if isinstance(trace, list):
        out["decision_trace"] = [
            item
            for item in trace
            if not (
                isinstance(item, dict)
                and str(item.get("node_id") or "").startswith("11.")
            )
        ]


def _set_execution_node(
    out: dict[str, Any],
    order_type: str | None,
    reason: str,
    accepted: bool,
) -> None:
    _remove_execution_nodes(out)
    if order_type not in _ORDER_TYPE_TO_NODE:
        return
    node = {
        "node_id": _ORDER_TYPE_TO_NODE[order_type],
        "section": "执行方式",
        "question": f"{order_type}的实时执行条件是否成立？",
        "answer": "是" if accepted else "否",
        "reason": reason,
        "skipped": False,
        "bar_range": "K1",
    }
    trace = out.setdefault("decision_trace", [])
    insert_at = len(trace)
    for index, item in enumerate(trace):
        if isinstance(item, dict) and str(item.get("node_id") or "").startswith("14"):
            insert_at = index
            break
    trace.insert(insert_at, node)
