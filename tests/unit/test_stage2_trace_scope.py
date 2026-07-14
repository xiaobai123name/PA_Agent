from __future__ import annotations

import copy
import json

from pa_agent.ai.json_validator import Ok, ValidationError
from tests.fixtures.validators import schema_test_validator
from tests.integration.conftest import VALID_STAGE2


def _validate(payload: dict):
    return schema_test_validator().validate(
        "stage2",
        json.dumps(payload, ensure_ascii=False),
    )


def test_stage2_rejects_repeated_stage1_node_even_with_valid_answer() -> None:
    payload = copy.deepcopy(VALID_STAGE2)
    payload["decision_trace"].append(
        {
            "node_id": "1.2",
            "question": "是否能识别市场周期？",
            "answer": "是",
            "branch": "trading_range",
            "reason": "识别为交易区间",
            "bar_range": "K20-K1",
        }
    )

    result = _validate(payload)

    assert isinstance(result, ValidationError)
    assert result.retryable_format is True
    assert any(
        "decision_trace" in field and "node_id='1.2'" in field
        for field in result.invalid_fields
    )


def test_stage2_real_failure_reports_enum_and_scope_errors() -> None:
    payload = copy.deepcopy(VALID_STAGE2)
    payload["decision_trace"].append(
        {
            "node_id": "1.2",
            "question": "是否能识别市场周期？",
            "answer": "极速",
            "branch": "trading_range",
            "reason": "波段高低点交错",
            "bar_range": "K20-K1",
        }
    )

    result = _validate(payload)

    assert isinstance(result, ValidationError)
    assert result.retryable_format is True
    assert any("answer" in field for field in result.invalid_fields)
    assert any("超出 Stage 2 范围" in field for field in result.invalid_fields)


def test_stage2_allows_direction_reassessment_node_23() -> None:
    payload = copy.deepcopy(VALID_STAGE2)
    payload["decision_trace"].append(
        {
            "node_id": "2.3",
            "question": "阶段二是否重新判定市场方向？",
            "answer": "是",
            "branch": "bullish",
            "reason": "K1 强势突破",
            "bar_range": "K2-K1",
        }
    )

    assert isinstance(_validate(payload), Ok)


def test_stage2_rejects_unknown_and_non_applicable_chapters() -> None:
    for node_id in ("0.3", "2.4", "11.1", "12.1", "13.1", "15.1", "custom"):
        payload = copy.deepcopy(VALID_STAGE2)
        payload["decision_trace"].append(
            {
                "node_id": node_id,
                "question": "非法节点",
                "answer": "否",
                "reason": "测试",
                "bar_range": "K1",
            }
        )
        result = _validate(payload)
        assert isinstance(result, ValidationError), node_id
        assert any("超出 Stage 2 范围" in field for field in result.invalid_fields)
