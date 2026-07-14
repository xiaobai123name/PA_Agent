"""Presentation models for the backtest decision audit UI."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo


ACTION_LABELS = {
    "none": "无操作",
    "place": "新建挂单",
    "keep": "继续等待",
    "replace": "替换挂单",
    "cancel": "取消挂单",
}
INTENT_LABELS = {
    "none": "无入场",
    "pullback": "回撤限价",
    "breakout": "突破触发",
    "immediate": "立即入场",
}
STATUS_LABELS = {
    "resolved": "执行通过",
    "rejected": "执行拒绝",
    "not_applicable": "无需执行",
    "failed": "决策失败",
}
DIRECTION_LABELS = {
    "做多": "做多",
    "做空": "做空",
    "bullish": "看多",
    "bearish": "看空",
    "neutral": "中性",
}
FILTER_LABELS = (
    ("关键决策", "key"),
    ("全部决策", "all"),
    ("交易方案", "plans"),
    ("决策失败", "failed"),
    ("执行拒绝", "rejected"),
    ("无操作", "none"),
)
LOCAL_TIMEZONE = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True, slots=True)
class DecisionAuditEntry:
    decision_id: str
    decision_time_ms: int
    cache_hit: bool
    payload: dict[str, Any]
    action: str
    action_label: str
    intent: str
    intent_label: str
    direction: str
    method_label: str
    entry_price: float | None
    status: str
    status_label: str
    reason: str
    trade_confidence: int | None
    validation_attempts: tuple[dict[str, Any], ...]
    is_key: bool
    search_text: str


def build_audit_entry(
    decision_id: str,
    decision_time_ms: int,
    cache_hit: bool,
    payload: dict[str, Any],
) -> DecisionAuditEntry:
    stage2 = payload.get("stage2_decision") or {}
    decision = stage2.get("decision") or {}
    review = decision.get("execution_review") or {}
    exception = payload.get("exception") or {}
    action = str(decision.get("order_action") or "")
    intent = str(decision.get("entry_intent") or "")
    direction = str(decision.get("order_direction") or "")
    status = str(review.get("status") or ("failed" if exception else ""))
    reason = str(
        review.get("reason")
        or exception.get("message")
        or decision.get("reasoning")
        or ("缓存命中" if cache_hit else "")
    )
    action_label = ACTION_LABELS.get(action, "决策失败" if exception else action or "未知")
    intent_label = INTENT_LABELS.get(intent, intent or "—")
    direction_label = DIRECTION_LABELS.get(direction, direction or "—")
    method_label = (
        f"{direction_label} / {intent_label}" if direction else intent_label
    )
    entry_price = _number_or_none(decision.get("entry_price"))
    confidence = _int_or_none(decision.get("trade_confidence"))
    validation_attempts = tuple(payload.get("validation_attempts") or [])
    is_key = bool(
        exception
        or action in {"place", "keep", "replace", "cancel"}
        or status == "rejected"
    )
    search_values = [
        decision_id,
        action,
        action_label,
        intent,
        intent_label,
        direction,
        method_label,
        status,
        STATUS_LABELS.get(status, status),
        reason,
        str(decision.get("reasoning") or ""),
        " ".join(
            str(attempt.get("message") or "")
            for attempt in validation_attempts
            if isinstance(attempt, dict)
        ),
        " ".join(str(x) for x in decision.get("key_factors") or []),
        " ".join(str(x) for x in decision.get("watch_points") or []),
    ]
    return DecisionAuditEntry(
        decision_id=decision_id,
        decision_time_ms=int(decision_time_ms),
        cache_hit=bool(cache_hit),
        payload=payload,
        action=action,
        action_label=action_label,
        intent=intent,
        intent_label=intent_label,
        direction=direction,
        method_label=method_label,
        entry_price=entry_price,
        status=status,
        status_label=STATUS_LABELS.get(status, status or "—"),
        reason=reason,
        trade_confidence=confidence,
        validation_attempts=validation_attempts,
        is_key=is_key,
        search_text="\n".join(search_values).lower(),
    )


def matches_filter(entry: DecisionAuditEntry, filter_key: str) -> bool:
    if filter_key == "key":
        return entry.is_key
    if filter_key == "all":
        return True
    if filter_key == "plans":
        return entry.action in {"place", "keep", "replace", "cancel"}
    if filter_key == "failed":
        return entry.status == "failed"
    if filter_key == "rejected":
        return entry.status == "rejected"
    if filter_key == "none":
        return entry.action == "none"
    raise ValueError(f"未知决策筛选器: {filter_key}")


def format_local_time(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(
        int(timestamp_ms) / 1000,
        tz=LOCAL_TIMEZONE,
    ).strftime("%Y-%m-%d %H:%M")


def summary_fields(entry: DecisionAuditEntry) -> dict[str, str]:
    payload = entry.payload
    stage1 = payload.get("stage1_diagnosis") or {}
    stage2 = payload.get("stage2_decision") or {}
    decision = stage2.get("decision") or {}
    review = decision.get("execution_review") or {}
    exception = payload.get("exception") or {}
    return {
        "市场周期": str(stage1.get("cycle_position") or "—"),
        "市场方向": DIRECTION_LABELS.get(
            str(stage1.get("direction") or ""), str(stage1.get("direction") or "—")
        ),
        "诊断置信度": _percent(stage1.get("diagnosis_confidence")),
        "交易置信度": _percent(decision.get("trade_confidence")),
        "决策动作": entry.action_label,
        "方向/方式": entry.method_label,
        "入场价": _price(decision.get("entry_price")),
        "结构止损": _price(decision.get("stop_loss_price")),
        "TP1": _price(decision.get("take_profit_price")),
        "TP2": _price(decision.get("take_profit_price_2")),
        "挂单有效期": _bars(decision.get("order_valid_bars")),
        "执行状态": entry.status_label,
        "模拟报价": _price(review.get("market_price")),
        "缓存": "命中" if entry.cache_hit else "未命中",
        "校验重试": str(len(entry.validation_attempts)),
        "失败阶段": str(exception.get("stage") or "—"),
        "失败类型": str(exception.get("type") or "—"),
    }


def ai_basis_text(entry: DecisionAuditEntry) -> str:
    stage1 = entry.payload.get("stage1_diagnosis") or {}
    stage2 = entry.payload.get("stage2_decision") or {}
    decision = stage2.get("decision") or {}
    if not decision:
        exception = entry.payload.get("exception") or {}
        return str(exception.get("raw_text") or exception.get("message") or "无阶段二决策")
    parts = [
        f"阶段一诊断\n{stage1.get('entry_setup') or stage1.get('risk_warning') or '—'}",
        _list_block("阶段一关键信号", stage1.get("key_signals")),
        f"决策理由\n{decision.get('reasoning') or '—'}",
    ]
    parts.append(_list_block("关键因素", decision.get("key_factors")))
    parts.append(_list_block("观察点", decision.get("watch_points")))
    parts.append(f"风险评估\n{decision.get('risk_assessment') or '—'}")
    parts.append(f"结构失效条件\n{decision.get('invalidation_condition') or '—'}")
    return "\n\n".join(part for part in parts if part)


def execution_audit_text(entry: DecisionAuditEntry) -> str:
    payload = entry.payload
    decision = ((payload.get("stage2_decision") or {}).get("decision") or {})
    review = decision.get("execution_review") or {}
    exception = payload.get("exception") or {}
    lines = [
        f"动作: {entry.action_label}",
        f"方向/方式: {entry.method_label}",
        f"订单类型: {decision.get('order_type') or '—'}",
        f"执行状态: {entry.status_label}",
        f"执行代码: {review.get('reason_code') or '—'}",
        f"执行原因: {review.get('reason') or exception.get('message') or '—'}",
        f"模拟报价: {_price(review.get('market_price'))}",
        f"报价年龄: {_milliseconds(review.get('quote_age_ms'))}",
        f"允许滑点: {_price(review.get('max_slippage'))}",
        f"挂单有效期: {_bars(decision.get('order_valid_bars'))}",
        f"缓存: {'命中' if entry.cache_hit else '未命中'}",
    ]
    proposed = review.get("proposed_structure")
    if isinstance(proposed, dict) and proposed:
        lines.extend(
            [
                "",
                "被拒绝的原始结构",
                f"方向: {proposed.get('order_direction') or '—'}",
                f"入场: {_price(proposed.get('entry_price'))}",
                f"止损: {_price(proposed.get('stop_loss_price'))}",
                f"TP1: {_price(proposed.get('take_profit_price'))}",
                f"TP2: {_price(proposed.get('take_profit_price_2'))}",
            ]
        )
    high_rr_review = decision.get("high_rr_review")
    if isinstance(high_rr_review, dict) and high_rr_review:
        lines.extend(
            [
                "",
                "高盈亏比复核",
                f"状态: {high_rr_review.get('status') or '—'}",
                f"止损依据: {high_rr_review.get('stop_loss_basis') or '—'}",
                f"TP1依据: {high_rr_review.get('tp1_basis') or '—'}",
                f"胜率依据: {high_rr_review.get('win_rate_basis') or '—'}",
            ]
        )
    return "\n".join(lines)


def validation_attempts_text(entry: DecisionAuditEntry) -> str:
    if not entry.validation_attempts:
        return "本次决策未发生校验重试。"
    blocks: list[str] = []
    for item in entry.validation_attempts:
        stage = str(item.get("stage") or "unknown")
        attempt = item.get("attempt")
        category = str(item.get("category") or "")
        invalid = item.get("invalid_fields") or []
        missing = item.get("missing_fields") or []
        lines = [
            f"{stage} 第 {attempt} 次输出未通过",
            f"类别: {category}",
            f"说明: {item.get('message') or '—'}",
            f"无效字段: {', '.join(str(value) for value in invalid) or '—'}",
            f"缺少字段: {', '.join(str(value) for value in missing) or '—'}",
        ]
        feedback = str(item.get("feedback") or "")
        if feedback:
            lines.extend(["", "定向反馈", feedback])
        lines.extend(["", "原始响应", str(item.get("raw_text") or "")])
        blocks.append("\n".join(lines))
    final_status = "最终校验失败" if entry.payload.get("exception") else "最终校验通过"
    return f"{final_status}\n\n" + "\n\n".join(blocks)


def _list_block(title: str, values: Any) -> str:
    if not isinstance(values, list) or not values:
        return f"{title}\n—"
    return title + "\n" + "\n".join(f"• {value}" for value in values)


def _number_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _price(value: Any) -> str:
    number = _number_or_none(value)
    return "—" if number is None else f"{number:g}"


def _percent(value: Any) -> str:
    number = _int_or_none(value)
    return "—" if number is None else f"{number}%"


def _bars(value: Any) -> str:
    number = _int_or_none(value)
    return "—" if number is None else f"{number} 根"


def _milliseconds(value: Any) -> str:
    number = _int_or_none(value)
    return "—" if number is None else f"{number} ms"
