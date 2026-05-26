"""JSON validator for Stage 1 and Stage 2 AI outputs.

Categories:
  a — syntax error (invalid JSON)
  b — missing required field
  c — illegal value (enum violation, type mismatch, 不下单 price non-null, etc.)
  d — plain text (no JSON structure at all)
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Tokens in §9 / decision reasoning that justify trading on weak|invalid signal_bar.
_EXPLICIT_S9_TRADABLE_TOKENS = (
    "弱",
    "瑕疵",
    "激进",
    "仍可",
    "例外",
    "次优",
    "等待信号",
    "无信号",
    "挂单",
    "计划型",
    "接受",
    "限价",
)

# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class Ok:
    """Successful validation result."""
    obj: dict[str, Any]


@dataclass
class ValidationError:
    """Failed validation result."""
    category: Literal["a", "b", "c", "d", "e"]
    stage: str                          # "stage1" or "stage2"
    raw_text: str
    parse_position: str | None = None   # "line:col" if available
    missing_fields: list[str] = field(default_factory=list)
    invalid_fields: list[str] = field(default_factory=list)
    allowed_values: dict[str, list] = field(default_factory=dict)
    message: str = ""


Result = Ok | ValidationError

# ── Markdown fence stripper ───────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_TRAILING_FENCE_RE = re.compile(r"\n?```\s*$")
_LEADING_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?", re.IGNORECASE)


def _extract_outer_json_object(text: str) -> str:
    """Return the first top-level `{...}` object, ignoring trailing prose/fences."""
    start = text.find("{")
    if start < 0:
        return text.strip()

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:].strip()


def _strip_fences(text: str) -> str:
    """Remove markdown fences and isolate the JSON object payload."""
    t = text.strip()
    if not t:
        return t

    # Fully fenced ```json ... ```
    if t.startswith("```"):
        m = _FENCE_RE.search(t)
        if m:
            t = m.group(1).strip()
        else:
            t = _LEADING_FENCE_RE.sub("", t, count=1).strip()

    # Common model mistake: raw JSON + trailing ``` only
    t = _TRAILING_FENCE_RE.sub("", t).strip()

    return _repair_unescaped_quotes(_extract_outer_json_object(t))


# ── Unescaped quote repair ────────────────────────────────────────────────────

_STRING_END_CHARS = frozenset(",:}]")


def _repair_unescaped_quotes(text: str) -> str:
    """Escape ``"`` inside JSON string values that were not backslash-escaped.

    Uses a peek-ahead heuristic: a quote ends the string only when the next
    non-whitespace character is structural (`,`, `:`, `}`, `]`, or EOF).
    """
    out: list[str] = []
    in_string = False
    escape = False
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if not in_string:
            if ch == '"':
                in_string = True
            out.append(ch)
            i += 1
            continue

        if escape:
            escape = False
            out.append(ch)
            i += 1
            continue
        if ch == "\\":
            escape = True
            out.append(ch)
            i += 1
            continue
        if ch == '"':
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j >= n or text[j] in _STRING_END_CHARS:
                in_string = False
                out.append(ch)
            else:
                out.append('\\"')
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


# ── Truncated JSON repair ───────────────────────────────────────────────────────

def _balance_json_brackets(text: str) -> str:
    """Close unclosed ``{`` / ``[`` outside JSON strings."""
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            stack.append("{")
        elif ch == "[":
            stack.append("[")
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()
    closers = "".join("]" if opener == "[" else "}" for opener in reversed(stack))
    return text + closers


def _inject_stage1_missing_tail(text: str) -> str:
    """Append minimal gate fields when the model stopped after bar_by_bar_summary."""
    lowered = text.lower()
    if '"gate_result"' in lowered or '"gate_trace"' in lowered:
        return text

    tail = text.rstrip()
    if not tail.endswith((",", "]", "}")):
        return text

    if not tail.endswith(","):
        tail += ","

    stub_trace = (
        '{"node_id":"AUTO","question":"输出是否在gate_trace前被截断？",'
        '"answer":"否","reason":"JSON在gate_trace前截断，程序已补全最小闸门记录",'
        '"bar_range":"K1"}'
    )
    tail += f'"gate_trace":[{stub_trace}],"gate_result":"unknown"'
    return _balance_json_brackets(tail)


def _try_repair_json_syntax(
    text: str,
    stage: Literal["stage1", "stage2"],
    *,
    allow_tail_inject: bool = False,
) -> str | None:
    """Return repaired JSON text when truncation caused a syntax error, else None."""
    if not text.strip().startswith("{"):
        return None

    candidate = text.rstrip()
    if stage == "stage1" and allow_tail_inject:
        candidate = _inject_stage1_missing_tail(candidate)
    candidate = _balance_json_brackets(candidate)
    if candidate == text.rstrip():
        return None
    try:
        json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return candidate


# ── JsonValidator ─────────────────────────────────────────────────────────────

class JsonValidator:
    """Validates raw AI text against Stage 1 or Stage 2 JSON schemas."""

    def __init__(self, validation: Any = None) -> None:
        from pa_agent.ai.prompts.schemas import STAGE1_SCHEMA, STAGE2_SCHEMA
        from pa_agent.config.settings import ValidationSettings

        if validation is None:
            self._validation = ValidationSettings()
        elif hasattr(validation, "validation"):
            self._validation = validation.validation
        else:
            self._validation = validation

        self._schemas = {
            "stage1": STAGE1_SCHEMA,
            "stage2": STAGE2_SCHEMA,
        }

    def validate(
        self,
        stage: Literal["stage1", "stage2"],
        raw_text: str,
        *,
        decision_stance: str | None = None,
        kline_frame: Any = None,
        stage1_json: dict[str, Any] | None = None,
        incremental_new_bar_count: int = 0,
        incremental_previous_stage1: dict[str, Any] | None = None,
    ) -> Result:
        """Validate *raw_text* against the schema for *stage*.

        Returns Ok(obj) on success, ValidationError on any failure.
        """
        schema = self._schemas[stage]

        # ── Category d: plain text (no JSON at all) ───────────────────────────
        stripped = _strip_fences(raw_text)
        if not stripped.startswith("{") and not stripped.startswith("["):
            return ValidationError(
                category="d",
                stage=stage,
                raw_text=raw_text,
                message="Response is plain text, not JSON",
            )

        # ── Category a: syntax error ──────────────────────────────────────────
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            # Stage 2: fail fast on syntax errors (no silent truncation repair).
            allow_inject = (
                stage == "stage1"
                and not getattr(self._validation, "disable_truncation_repair", True)
            )
            repaired = (
                _try_repair_json_syntax(stripped, stage, allow_tail_inject=allow_inject)
                if stage == "stage1"
                else None
            )
            if repaired is not None:
                try:
                    obj = json.loads(repaired)
                    logger.warning(
                        "Repaired truncated %s JSON (%d -> %d chars)",
                        stage,
                        len(stripped),
                        len(repaired),
                    )
                except json.JSONDecodeError:
                    repaired = None
            if repaired is None:
                pos = f"{exc.lineno}:{exc.colno}"
                return ValidationError(
                    category="a",
                    stage=stage,
                    raw_text=raw_text,
                    parse_position=pos,
                    message=f"JSON syntax error at {pos}: {exc.msg}",
                )

        if not isinstance(obj, dict):
            return ValidationError(
                category="a",
                stage=stage,
                raw_text=raw_text,
                message="Top-level JSON value is not an object",
            )

        norm_mode = getattr(self._validation, "normalization_mode", "strict")
        if stage == "stage1":
            from pa_agent.ai.stage1_normalizer import normalize_stage1

            obj = normalize_stage1(
                obj,
                normalization_mode=norm_mode,
                kline_frame=kline_frame,
                incremental_new_bar_count=int(incremental_new_bar_count or 0),
                incremental_previous_stage1=incremental_previous_stage1
                if incremental_new_bar_count > 0
                else None,
            )
        elif stage == "stage2":
            from pa_agent.ai.stage2_normalizer import normalize_stage2

            obj = normalize_stage2(
                obj, normalization_mode=norm_mode, kline_frame=kline_frame
            )

        # ── Schema validation (b and c) ───────────────────────────────────────
        try:
            import jsonschema  # type: ignore[import]
        except ImportError:
            logger.warning("jsonschema not installed; skipping schema validation")
            return Ok(obj=obj)

        errors = list(jsonschema.Draft7Validator(schema).iter_errors(obj))

        # Classify errors
        missing: list[str] = []
        invalid: list[str] = []
        allowed: dict[str, list] = {}

        for err in errors:
            path = ".".join(str(p) for p in err.absolute_path) or err.schema_path[-1]
            if err.validator == "required":
                # Extract the missing property name from the message
                missing.append(err.message.split("'")[1] if "'" in err.message else str(path))
            else:
                invalid.append(str(path) or err.message[:80])
                if "enum" in err.schema:
                    allowed[str(path)] = err.schema["enum"]

        # ── Explicit cross-field checks ───────────────────────────────────────
        if stage == "stage1":
            from pa_agent.ai.decision_tree import validate_gate_result_consistency
            from pa_agent.ai.coherence_checks import (
                validate_incremental_stage1_coherence,
                validate_stage1_coherence,
            )

            for msg in validate_gate_result_consistency(obj):
                invalid.append(f"gate:{msg}")
            for msg in validate_stage1_coherence(
                obj,
                kline_frame=kline_frame,
                strict_bar_features=getattr(
                    self._validation, "strict_bar_by_bar_features", True
                ),
            ):
                invalid.append(f"s1:{msg}")
            if incremental_new_bar_count > 0:
                for msg in validate_incremental_stage1_coherence(
                    obj,
                    new_bar_count=incremental_new_bar_count,
                    previous_stage1=incremental_previous_stage1,
                ):
                    invalid.append(f"s1:{msg}")
            if getattr(self._validation, "trace_semantic_checks", True):
                from pa_agent.ai.trace_semantic_checks import validate_trace_semantics

                gate_trace = obj.get("gate_trace")
                if isinstance(gate_trace, list):
                    for msg in validate_trace_semantics(
                        gate_trace,
                        path_prefix="gate_trace",
                        stage="stage1",
                        gate_result=str(obj.get("gate_result", "")),
                    ):
                        invalid.append(f"trace_semantic:{msg}")

        if stage == "stage2":
            no_order_err = self._check_no_order_invariant(obj)
            if no_order_err:
                invalid.extend(no_order_err["fields"])
                allowed.update(no_order_err["allowed"])

            breakout_err = self._check_breakout_order_basis(obj)
            if breakout_err:
                invalid.extend(breakout_err["fields"])
                allowed.update(breakout_err["allowed"])

            for msg in self._check_breakout_price_extreme(obj, kline_frame):
                invalid.append(f"breakout_price:{msg}")

            for msg in self._check_signal_chain(obj, kline_frame):
                invalid.append(f"signal_chain:{msg}")

            for msg in self._check_next_bar_prediction(obj):
                invalid.append(msg)

            for msg in self._check_trade_metrics(obj, decision_stance=decision_stance):
                invalid.append(f"metrics:{msg}")

            from pa_agent.ai.decision_tree import validate_stage2_trace_consistency
            from pa_agent.ai.coherence_checks import validate_stage2_coherence

            for msg in validate_stage2_trace_consistency(obj):
                invalid.append(f"trace:{msg}")
            if isinstance(stage1_json, dict):
                for msg in validate_stage2_coherence(
                    obj, stage1_json, kline_frame=kline_frame
                ):
                    invalid.append(f"s2:{msg}")
            if getattr(self._validation, "trace_semantic_checks", True):
                from pa_agent.ai.trace_semantic_checks import (
                    validate_stage2_order_trace_semantics,
                    validate_trace_semantics,
                )

                decision_trace = obj.get("decision_trace")
                if isinstance(decision_trace, list):
                    for msg in validate_trace_semantics(
                        decision_trace,
                        path_prefix="decision_trace",
                        stage="stage2",
                    ):
                        invalid.append(f"trace_semantic:{msg}")
                for msg in validate_stage2_order_trace_semantics(obj):
                    invalid.append(f"trace_semantic:{msg}")

        if not errors and not missing and not invalid:
            return Ok(obj=obj)

        # Determine category: b if only missing fields, c otherwise
        if invalid or (missing and errors[0].validator not in ("required",)):
            category: Literal["b", "c"] = "c"
        elif missing:
            category = "b"
        else:
            category = "c"

        first_message = errors[0].message[:120] if errors else (invalid[0] if invalid else "custom validation failed")
        return ValidationError(
            category=category,
            stage=stage,
            raw_text=raw_text,
            missing_fields=missing,
            invalid_fields=invalid,
            allowed_values=allowed,
            message=f"{len(errors)} schema error(s): {first_message}",
        )

    @staticmethod
    def _check_no_order_invariant(obj: dict) -> dict | None:
        """Explicitly enforce the 不下单 ↔ null iron law.

        Returns a dict with 'fields' and 'allowed' if violated, else None.
        """
        decision = obj.get("decision", {})
        if not isinstance(decision, dict):
            return None

        order_type = decision.get("order_type")
        price_fields = ["entry_price", "take_profit_price", "stop_loss_price", "order_direction"]

        if order_type == "不下单":
            violated = [f for f in price_fields if decision.get(f) is not None]
            if violated:
                return {
                    "fields": violated,
                    "allowed": {f: [None] for f in violated},
                }
        elif order_type in ("限价单", "突破单", "市价单"):
            violated = [f for f in price_fields if decision.get(f) is None]
            if violated:
                return {
                    "fields": violated,
                    "allowed": {
                        "entry_price": ["<finite number>"],
                        "take_profit_price": ["<finite number>"],
                        "stop_loss_price": ["<finite number>"],
                        "order_direction": ["做多", "做空"],
                    },
                }
        return None

    @staticmethod
    def _check_breakout_order_basis(obj: dict) -> dict | None:
        """Require breakout orders to be tied to a bar extreme, not a mid-bar price."""
        decision = obj.get("decision", {})
        if not isinstance(decision, dict) or decision.get("order_type") != "突破单":
            return None

        fields: list[str] = []
        allowed: dict[str, list] = {}
        direction = decision.get("order_direction")
        extreme = decision.get("entry_basis_extreme")

        if not decision.get("entry_basis_bar"):
            fields.append("decision.entry_basis_bar")
            allowed["decision.entry_basis_bar"] = ["K{n}"]
        if extreme not in ("high", "low"):
            fields.append("decision.entry_basis_extreme")
            allowed["decision.entry_basis_extreme"] = ["high", "low"]
        if not decision.get("entry_rule"):
            fields.append("decision.entry_rule")
            allowed["decision.entry_rule"] = [
                "做多突破单=依据K线高点上方1跳动",
                "做空突破单=依据K线低点下方1跳动",
            ]

        if direction == "做多" and extreme == "low":
            fields.append("decision.entry_basis_extreme")
            allowed["decision.entry_basis_extreme"] = ["做多突破单必须使用 high"]
        if direction == "做空" and extreme == "high":
            fields.append("decision.entry_basis_extreme")
            allowed["decision.entry_basis_extreme"] = ["做空突破单必须使用 low"]

        if fields:
            return {"fields": fields, "allowed": allowed}
        return None

    @staticmethod
    def _check_trade_metrics(
        obj: dict,
        *,
        decision_stance: str | None = None,
    ) -> list[str]:
        """Enforce RR and trader equation from entry/stop/target (not narrative distances)."""
        from pa_agent.util.trade_metrics import validate_order_trade_metrics

        decision = obj.get("decision", {})
        if not isinstance(decision, dict):
            return []
        return validate_order_trade_metrics(
            decision,
            decision_stance=decision_stance,
        )

    @staticmethod
    def _check_breakout_price_extreme(obj: dict, kline_frame: Any = None) -> list[str]:
        """Numerically verify breakout entry is outside the cited bar extreme."""
        if kline_frame is None:
            return []
        decision = obj.get("decision", {})
        if not isinstance(decision, dict) or decision.get("order_type") != "突破单":
            return []

        basis = _parse_k_seq(decision.get("entry_basis_bar"))
        if basis is None:
            return []
        bar = _bar_by_seq(kline_frame, basis)
        if bar is None:
            return [f"entry_basis_bar K{basis} not found in current K-line frame"]

        try:
            entry = float(decision.get("entry_price"))
        except (TypeError, ValueError):
            return []

        direction = decision.get("order_direction")
        extreme = decision.get("entry_basis_extreme")
        if direction == "做多" and extreme == "high" and entry <= float(bar.high):
            return [
                f"做多突破单 entry_price={entry:.6g} must be above "
                f"K{basis}.high={float(bar.high):.6g}"
            ]
        if direction == "做空" and extreme == "low" and entry >= float(bar.low):
            return [
                f"做空突破单 entry_price={entry:.6g} must be below "
                f"K{basis}.low={float(bar.low):.6g}"
            ]
        return []

    @staticmethod
    def _check_next_bar_prediction(obj: dict) -> list[str]:
        """Cross-field validation: sum constraint, direction=argmax, null consistency.

        Returns error message list; caller adds each to invalid_fields.
        """
        pred = obj.get("next_bar_prediction")
        if pred is None:
            return []  # Missing field is backward-compatible (R2.3, R7.3)
        if not isinstance(pred, dict):
            return ["next_bar_prediction: must be an object when present"]

        errors: list[str] = []
        unpredictable = bool(pred.get("unpredictable", False))

        if unpredictable:
            if pred.get("direction") is not None:
                errors.append("next_bar_prediction.direction: must be null when unpredictable=true")
            if pred.get("probabilities") is not None:
                errors.append("next_bar_prediction.probabilities: must be null when unpredictable=true")
            return errors

        # unpredictable=false path
        probs = pred.get("probabilities")
        if not isinstance(probs, dict):
            return ["next_bar_prediction.probabilities: must be an object when unpredictable=false"]

        for key in ("bullish", "bearish", "neutral"):
            value = probs.get(key)
            if not isinstance(value, int) or not (0 <= value <= 100):
                errors.append(f"next_bar_prediction.probabilities.{key}: must be int in [0, 100]")
        if errors:
            return errors

        # R3.2: sum in [99, 101]
        total = probs["bullish"] + probs["bearish"] + probs["neutral"]
        if not (99 <= total <= 101):
            errors.append(
                f"next_bar_prediction.probabilities: sum={total}, must satisfy 99 <= sum <= 101"
            )

        # R3.3: direction = argmax, break ties by literal order
        order = ("bullish", "bearish", "neutral")
        max_value = max(probs[k] for k in order)
        expected = next(k for k in order if probs[k] == max_value)
        direction = pred.get("direction")
        if direction != expected:
            errors.append(
                f"next_bar_prediction.direction: expected '{expected}' (argmax of probabilities), "
                f"got {direction!r}"
            )

        return errors

    @staticmethod
    def _check_signal_chain(obj: dict, kline_frame: Any = None) -> list[str]:
        """Require order decisions to ground §9 in signal/entry/follow-through facts."""
        decision = obj.get("decision", {})
        if not isinstance(decision, dict):
            return []
        if decision.get("order_type") not in ("限价单", "突破单", "市价单"):
            return []

        errors: list[str] = []
        bar_analysis = obj.get("bar_analysis")
        if not isinstance(bar_analysis, dict):
            return ["bar_analysis is required when placing an order"]

        signal_bar = bar_analysis.get("signal_bar")
        entry_bar = bar_analysis.get("entry_bar")
        if not isinstance(signal_bar, dict):
            errors.append("bar_analysis.signal_bar is required when placing an order")
        if not isinstance(entry_bar, dict):
            errors.append("bar_analysis.entry_bar is required when placing an order")
        if errors:
            return errors

        sig_seq = _parse_k_seq(signal_bar.get("bar"))
        entry_seq = _parse_k_seq(entry_bar.get("bar"))
        strength = str(entry_bar.get("strength", "") or "").strip().lower()
        freshness = str(entry_bar.get("freshness", "fresh")).strip().lower()
        quality = str(signal_bar.get("quality", "")).strip().lower()
        pattern = str(signal_bar.get("pattern", "") or "").strip().lower()
        pending_entry = (
            strength == "not_triggered"
            or freshness == "pending"
            or entry_bar.get("bar") is None
        )
        planned_without_signal = (
            pending_entry
            and decision.get("order_type") in ("限价单", "突破单")
            and quality == "invalid"
            and pattern in ("", "none", "not_triggered", "pending")
            and signal_bar.get("bar") is None
        )
        if sig_seq is None and not planned_without_signal:
            errors.append("bar_analysis.signal_bar.bar must be a K{n} reference")
        if entry_seq is None and not pending_entry:
            errors.append("bar_analysis.entry_bar.bar must be a K{n} reference")
        if pending_entry and decision.get("order_type") == "市价单":
            errors.append("market order requires a concrete entry_bar.bar")
        if sig_seq is not None and entry_seq is not None and sig_seq <= entry_seq:
            errors.append(
                "signal_bar must be older than entry_bar "
                f"(expected signal K seq > entry K seq, got K{sig_seq} and K{entry_seq})"
            )
        if kline_frame is not None:
            for label, seq in (("signal_bar", sig_seq), ("entry_bar", entry_seq)):
                if seq is not None and _bar_by_seq(kline_frame, seq) is None:
                    errors.append(f"bar_analysis.{label}.bar K{seq} not found in current K-line frame")

        if quality in ("weak", "invalid") and not planned_without_signal:
            reasons = _all_stage2_reasons(obj)
            if not any(token in reasons for token in _EXPLICIT_S9_TRADABLE_TOKENS):
                errors.append(
                    "weak/invalid signal_bar requires explicit §9 reasoning for why the setup remains tradable"
                )

        follow = entry_bar.get("follow_through")
        no_follow = follow is False or str(follow).strip().lower() in ("false", "no", "failed")
        trade_conf = decision.get("trade_confidence")
        try:
            trade_conf_num = int(trade_conf)
        except (TypeError, ValueError):
            trade_conf_num = 0
        if freshness in ("stale", "invalid"):
            errors.append("entry_bar.freshness stale/invalid cannot support a new order")
        if no_follow and not pending_entry and trade_conf_num >= 50:
            errors.append(
                "entry_bar.follow_through=false/failed cannot support trade_confidence >= 50"
            )
        return errors


def _parse_k_seq(value: object) -> int | None:
    if value is None:
        return None
    m = re.search(r"K\s*(\d+)", str(value), flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def _bar_by_seq(kline_frame: Any, seq: int) -> Any | None:
    for bar in getattr(kline_frame, "bars", ()) or ():
        if getattr(bar, "seq", None) == seq:
            return bar
    return None


def _all_stage2_reasons(obj: dict) -> str:
    parts: list[str] = []
    decision = obj.get("decision", {})
    if isinstance(decision, dict):
        for key in ("reasoning", "trade_confidence_reasoning", "risk_assessment"):
            parts.append(str(decision.get(key, "") or ""))
    for item in obj.get("decision_trace", []) or []:
        if isinstance(item, dict):
            parts.append(str(item.get("reason", "") or ""))
    return "\n".join(parts)
