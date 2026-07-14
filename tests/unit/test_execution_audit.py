from __future__ import annotations

import csv
import json

import pytest

from pa_agent.records import trade_logger


def _decision(status: str = "rejected") -> dict:
    return {
        "entry_intent": "breakout",
        "order_type": "不下单" if status == "rejected" else "突破单",
        "execution_review": {
            "status": status,
            "reason_code": "breakout_trigger_already_crossed",
            "reason": "trigger crossed",
            "proposed_order_type": "突破单",
            "proposed_entry_price": 101.0,
            "proposed_structure": {
                "order_direction": "做多",
                "entry_price": 101.0,
                "stop_loss_price": 98.0,
            },
            "resolved_order_type": "不下单" if status == "rejected" else "突破单",
            "market_price": 102.0,
            "quote_timestamp_ms": 10_000,
            "quote_age_ms": 100,
            "max_slippage": None,
        },
    }


def test_execution_audit_preserves_rejected_proposal(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(trade_logger, "_TRADE_RECORDS_DIR", tmp_path)
    decision = _decision()

    path = trade_logger.save_execution_audit(
        decision_inner=decision,
        stage2_full={
            "decision": decision,
            "terminal": {"node_id": "11.2", "outcome": "reject"},
        },
        meta_symbol="BTCUSDT",
        meta_timeframe="15m",
        decision_stance="balanced",
        model_name="test-model",
    )

    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["execution_status"] == "rejected"
    assert rows[0]["execution_reason_code"] == "breakout_trigger_already_crossed"
    assert json.loads(rows[0]["proposed_structure"])["entry_price"] == 101.0
    assert rows[0]["terminal_outcome"] == "reject"


def test_execution_audit_rejects_non_attempt_status(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(trade_logger, "_TRADE_RECORDS_DIR", tmp_path)
    decision = _decision(status="not_applicable")

    with pytest.raises(ValueError, match="only accepts resolved or rejected"):
        trade_logger.save_execution_audit(
            decision_inner=decision,
            stage2_full={"decision": decision},
            meta_symbol="BTCUSDT",
            meta_timeframe="15m",
            decision_stance="balanced",
            model_name="test-model",
        )

