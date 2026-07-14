from __future__ import annotations

import pytest

from pa_agent.gui.backtest_audit import (
    ai_basis_text,
    build_audit_entry,
    execution_audit_text,
    format_local_time,
    matches_filter,
    summary_fields,
    validation_attempts_text,
)


def _payload(action: str = "place", status: str = "resolved") -> dict:
    return {
        "stage1_diagnosis": {
            "cycle_position": "trending_tr",
            "direction": "bearish",
            "diagnosis_confidence": 75,
            "entry_setup": "等待反弹至阻力后做空",
            "key_signals": ["空头破位", "反弹测试"],
        },
        "stage2_decision": {
            "decision": {
                "order_action": action,
                "order_direction": "做空",
                "entry_intent": "pullback",
                "order_type": "限价单",
                "entry_price": 63030,
                "stop_loss_price": 63367,
                "take_profit_price": 62435,
                "take_profit_price_2": 62358,
                "order_valid_bars": 4,
                "trade_confidence": 48,
                "reasoning": "顺势等待阻力位做空",
                "key_factors": ["Always In Short"],
                "watch_points": ["阻力位拒绝"],
                "risk_assessment": "反弹突破阻力则失效",
                "invalidation_condition": "收盘升破 63367",
                "execution_review": {
                    "status": status,
                    "reason_code": "execution_method_resolved",
                    "reason": "结构回撤尚未到价",
                    "market_price": 62952.8,
                    "quote_age_ms": 0,
                },
            }
        },
        "exception": None,
    }


def test_build_entry_maps_chinese_labels_and_key_decision() -> None:
    entry = build_audit_entry("decision-1", 0, False, _payload())

    assert entry.action_label == "新建挂单"
    assert entry.method_label == "做空 / 回撤限价"
    assert entry.status_label == "执行通过"
    assert entry.is_key is True
    assert "顺势等待阻力位做空" in entry.search_text


def test_none_is_hidden_from_key_filter_but_available_explicitly() -> None:
    entry = build_audit_entry("decision-2", 0, True, _payload("none", "not_applicable"))

    assert entry.action_label == "无操作"
    assert entry.status_label == "无需执行"
    assert matches_filter(entry, "key") is False
    assert matches_filter(entry, "none") is True
    assert matches_filter(entry, "all") is True


def test_rejected_decision_is_key_and_matches_rejected_filter() -> None:
    entry = build_audit_entry("decision-3", 0, False, _payload("none", "rejected"))

    assert entry.is_key is True
    assert matches_filter(entry, "rejected") is True
    assert entry.status_label == "执行拒绝"


def test_structured_detail_text_uses_existing_fields() -> None:
    entry = build_audit_entry("decision-4", 0, False, _payload())

    summary = summary_fields(entry)
    assert summary["诊断置信度"] == "75%"
    assert summary["交易置信度"] == "48%"
    assert summary["结构止损"] == "63367"
    assert "阶段一关键信号" in ai_basis_text(entry)
    assert "空头破位" in ai_basis_text(entry)
    assert "执行代码: execution_method_resolved" in execution_audit_text(entry)


def test_local_time_is_asia_shanghai() -> None:
    assert format_local_time(0) == "1970-01-01 08:00"


def test_failed_record_remains_visible_and_exposes_raw_failure() -> None:
    payload = {
        "stage1_diagnosis": {},
        "stage2_decision": None,
        "exception": {
            "message": "stage2 schema error",
            "raw_text": "{invalid response}",
        },
    }
    entry = build_audit_entry("failed-decision", 0, False, payload)

    assert entry.action_label == "决策失败"
    assert entry.status_label == "决策失败"
    assert entry.is_key is True
    assert matches_filter(entry, "failed") is True
    assert "{invalid response}" in ai_basis_text(entry)
    assert "stage2 schema error" in execution_audit_text(entry)
    assert summary_fields(entry)["失败类型"] == "—"


def test_validation_retry_audit_shows_first_failure_and_feedback() -> None:
    payload = _payload()
    payload["validation_attempts"] = [
        {
            "stage": "stage2",
            "attempt": 1,
            "category": "c",
            "message": "非法 Stage 2 节点",
            "missing_fields": [],
            "invalid_fields": ["decision_trace[1].node_id='1.2'"],
            "raw_text": '{"node_id":"1.2"}',
            "feedback": "只输出 §3–§10、§14",
        }
    ]
    entry = build_audit_entry("retried", 0, False, payload)

    assert summary_fields(entry)["校验重试"] == "1"
    text = validation_attempts_text(entry)
    assert "最终校验通过" in text
    assert "非法 Stage 2 节点" in text
    assert "只输出 §3–§10、§14" in text
    assert '{"node_id":"1.2"}' in text


def test_unknown_filter_is_not_silently_accepted() -> None:
    entry = build_audit_entry("decision-5", 0, False, _payload())

    with pytest.raises(ValueError, match="未知决策筛选器"):
        matches_filter(entry, "typo")
