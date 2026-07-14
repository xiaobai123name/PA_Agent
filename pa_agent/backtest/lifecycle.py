"""Strict AI-owned pending-order lifecycle validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pa_agent.backtest.models import OrderAction


class BacktestDecisionError(ValueError):
    """The AI decision does not match the explicit backtest lifecycle contract."""


@dataclass(frozen=True, slots=True)
class LifecycleDecision:
    action: OrderAction
    valid_bars: int | None
    execution_status: str
    decision: dict[str, Any]


def validate_lifecycle_decision(
    stage2: dict[str, Any],
    *,
    pending_order: dict[str, Any] | None,
) -> LifecycleDecision:
    decision = stage2.get("decision")
    if not isinstance(decision, dict):
        raise BacktestDecisionError("stage2.decision 必须是对象")

    raw_action = decision.get("order_action")
    try:
        action = OrderAction(raw_action)
    except (TypeError, ValueError) as exc:
        raise BacktestDecisionError(
            "decision.order_action 必须是 place/keep/replace/cancel/none"
        ) from exc

    valid_raw = decision.get("order_valid_bars")
    valid_bars: int | None
    if valid_raw is None:
        valid_bars = None
    elif isinstance(valid_raw, bool) or not isinstance(valid_raw, int):
        raise BacktestDecisionError("decision.order_valid_bars 必须是整数或 null")
    else:
        valid_bars = int(valid_raw)

    if action in (OrderAction.PLACE, OrderAction.REPLACE):
        if valid_bars is None or not 1 <= valid_bars <= 100:
            raise BacktestDecisionError(
                f"order_action={action.value} 时 order_valid_bars 必须为 1..100"
            )
    elif valid_bars is not None:
        raise BacktestDecisionError(
            f"order_action={action.value} 时 order_valid_bars 必须为 null"
        )

    has_pending_order = pending_order is not None
    if action == OrderAction.PLACE and has_pending_order:
        raise BacktestDecisionError("已有挂单时不能使用 order_action=place")
    if action == OrderAction.REPLACE and not has_pending_order:
        raise BacktestDecisionError("没有挂单时不能使用 order_action=replace")
    if action in (OrderAction.KEEP, OrderAction.CANCEL) and not has_pending_order:
        raise BacktestDecisionError(
            f"没有挂单时不能使用 order_action={action.value}"
        )
    if action == OrderAction.NONE and has_pending_order:
        raise BacktestDecisionError("已有挂单时不能使用 order_action=none")

    intent = str(decision.get("entry_intent") or "").strip().lower()
    order_type = str(decision.get("order_type") or "").strip()
    if action in (OrderAction.CANCEL, OrderAction.NONE):
        if intent != "none" or order_type != "不下单":
            raise BacktestDecisionError(
                f"order_action={action.value} 时必须 entry_intent=none 且 order_type=不下单"
            )
        for field in (
            "entry_price",
            "stop_loss_price",
            "take_profit_price",
            "take_profit_price_2",
            "order_direction",
        ):
            if decision.get(field) is not None:
                raise BacktestDecisionError(
                    f"order_action={action.value} 时 decision.{field} 必须为 null"
                )
    elif intent not in ("pullback", "breakout", "immediate"):
        raise BacktestDecisionError(
            f"order_action={action.value} 时必须声明交易 entry_intent"
        )

    if action == OrderAction.KEEP:
        assert pending_order is not None
        expected = {
            "entry_intent": pending_order.get("entry_intent"),
            "order_type": pending_order.get("order_type"),
            "order_direction": pending_order.get("direction"),
            "entry_price": pending_order.get("entry_price"),
            "stop_loss_price": pending_order.get("stop_price"),
            "take_profit_price": pending_order.get("tp1_price"),
            "take_profit_price_2": pending_order.get("tp2_price"),
        }
        for field, expected_value in expected.items():
            actual = decision.get(field)
            if isinstance(expected_value, (int, float)):
                try:
                    matches = float(actual) == float(expected_value)
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = actual == expected_value
            if not matches:
                raise BacktestDecisionError(
                    f"order_action=keep 时 decision.{field} 必须原样重复当前挂单；"
                    f"expected={expected_value!r}, actual={actual!r}"
                )

    review = decision.get("execution_review")
    if not isinstance(review, dict):
        raise BacktestDecisionError("decision.execution_review 缺失")
    status = str(review.get("status") or "")
    allowed_statuses = (
        {"resolved", "rejected"}
        if action in (OrderAction.PLACE, OrderAction.REPLACE)
        else ({"resolved"} if action == OrderAction.KEEP else {"not_applicable"})
    )
    if status not in allowed_statuses:
        raise BacktestDecisionError(
            f"order_action={action.value} 与 execution_review.status={status!r} 不一致"
        )
    return LifecycleDecision(
        action=action,
        valid_bars=valid_bars,
        execution_status=status,
        decision=decision,
    )


def render_lifecycle_prompt(pending: dict[str, Any] | None) -> str:
    """Return strict backtest-only Stage 2 instructions."""
    if pending is None:
        pending_block = "当前没有未成交挂单。"
        allowed = (
            "- 新建方案：order_action=\"place\"，并输出完整交易字段与 "
            "order_valid_bars=1..100。\n"
            "- 本轮不交易：order_action=\"none\"，entry_intent=\"none\"，"
            "order_type=\"不下单\"，order_valid_bars=null。"
        )
        field_rule = (
            "`keep` / `cancel` / `none` 时所有价格、方向、入场依据和胜率字段必须为 null。"
        )
    else:
        pending_block = (
            "当前存在一个未成交挂单：\n"
            f"```json\n{_json_dumps(pending)}\n```"
        )
        allowed = (
            "- 原方案仍成立：order_action=\"keep\"，必须把当前挂单的 entry_intent、"
            "order_type、order_direction、entry、stop、TP1、TP2 原样重复，"
            "order_valid_bars=null；程序严格逐字段比较，原剩余有效期不重置。\n"
            "- 用新结构替换：order_action=\"replace\"，输出完整新方案与 "
            "order_valid_bars=1..100。\n"
            "- 主动取消：order_action=\"cancel\"，entry_intent=\"none\"，"
            "order_type=\"不下单\"，order_valid_bars=null。"
        )
        field_rule = (
            "`cancel` / `none` 时所有价格、方向、入场依据和胜率字段必须为 null；"
            "`keep` 不是新方案，但必须完整重复现有挂单数值以通过标准交易结构校验。"
        )
    return (
        "## 回测挂单生命周期（必填，严格校验）\n\n"
        f"{pending_block}\n\n{allowed}\n\n"
        "在 decision 对象中必须额外输出 `order_action` 和 `order_valid_bars`。\n"
        f"{field_rule}\n"
        "禁止省略字段、禁止写近义词、禁止让程序猜测你的意图。"
    )


def _json_dumps(value: dict[str, Any]) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
