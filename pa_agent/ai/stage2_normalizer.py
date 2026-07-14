"""Normalize common Stage 2 AI JSON variants before schema validation."""
from __future__ import annotations

import copy
import logging
from typing import Any

from pa_agent.ai.trace_normalize import normalize_stage2_traces
from pa_agent.util.price_tick import parse_k_seq

logger = logging.getLogger(__name__)

# Max length for decision.reasoning (stage-2 trade rationale paragraph).
DECISION_REASONING_MAX_LEN = 280

# ── Model alias mappings (Stage 1 normalizer has the same; keep in sync) ──

_SIGNAL_BAR_QUALITY_ALIASES: dict[str, str] = {
    "low": "weak",
    "high": "strong",
    "moderate": "medium",
    "poor": "weak",
    "good": "strong",
    "bad": "invalid",
    # 中文 synonyms
    "弱": "weak",
    "中": "medium",
    "强": "strong",
    "无效": "invalid",
}

_BAR_TYPE_ENUM = frozenset({
    "trend_bull", "trend_bear", "doji", "inside",
    "outside_bull", "outside_bear", "flat", "other",
})
_SIGNAL_BAR_QUALITY_ENUM = frozenset({"strong", "medium", "weak", "invalid"})


_TRADE_ORDER_TYPES = frozenset({"限价单", "突破单", "市价单"})

# Valid enum values for features_used in next_bar_prediction / next_cycle_prediction.
# Must stay in sync with schemas.py _NEXT_BAR_PREDICTION / _NEXT_CYCLE_PREDICTION.
_VALID_FEATURES_USED = frozenset({
    "stage1_diagnosis",
    "kline_features",
    "analysis_history",
    "experience_library",
    "stage2_decision",
    "previous_prediction_summary",
})


def _strip_enum_suffix(raw: str) -> str:
    """Drop trailing annotations models append to closed enums (e.g. ``invalid（…）``)."""
    text = raw.strip()
    for sep in ("（", "(", "【", "[", "—", "–", " - ", "：", ":"):
        if sep in text:
            head = text.split(sep, 1)[0].strip()
            if head:
                return head
    return text


def _normalize_closed_enum(
    raw: object,
    allowed: frozenset[str],
    *,
    aliases: dict[str, str] | None = None,
) -> str | None:
    """Map messy model enum text to a schema token, or None if unrecognized."""
    if not isinstance(raw, str):
        return None
    text = _strip_enum_suffix(raw)
    key = text.strip().lower().replace(" ", "_")
    if aliases:
        key = aliases.get(key, key)
    if key in allowed:
        return key
    for token in sorted(allowed, key=len, reverse=True):
        if key.startswith(token):
            return token
    return None


def _stage1_bar_analysis_bar_type(stage1_json: dict[str, Any] | None) -> str | None:
    if not isinstance(stage1_json, dict):
        return None
    bar_analysis = stage1_json.get("bar_analysis")
    if not isinstance(bar_analysis, dict):
        return None
    return _normalize_closed_enum(bar_analysis.get("bar_type"), _BAR_TYPE_ENUM)


def _normalize_stage2_bar_analysis_enums(
    out: dict[str, Any],
    *,
    stage1_json: dict[str, Any] | None = None,
) -> bool:
    """Strip enum annotations and sync bar_type from stage1 when available."""
    changed = False
    bar_analysis = out.get("bar_analysis")
    if not isinstance(bar_analysis, dict):
        return False

    stage1_bt = _stage1_bar_analysis_bar_type(stage1_json)
    raw_bt = bar_analysis.get("bar_type")
    norm_bt = stage1_bt or _normalize_closed_enum(raw_bt, _BAR_TYPE_ENUM)
    if norm_bt and norm_bt != raw_bt:
        bar_analysis["bar_type"] = norm_bt
        changed = True

    signal_bar = bar_analysis.get("signal_bar")
    if isinstance(signal_bar, dict):
        raw_q = signal_bar.get("quality")
        norm_q = _normalize_closed_enum(
            raw_q,
            _SIGNAL_BAR_QUALITY_ENUM,
            aliases=_SIGNAL_BAR_QUALITY_ALIASES,
        )
        if norm_q and norm_q != raw_q:
            signal_bar["quality"] = norm_q
            changed = True
        raw_pat = str(signal_bar.get("pattern", "") or "").strip().lower()
        if raw_pat in ("no_signal", "no-signal", "nosignal", "not_triggered"):
            signal_bar["pattern"] = "none"
            changed = True
        if not str(signal_bar.get("reason") or "").strip():
            signal_bar["reason"] = "无独立信号棒（quality=invalid 或计划型观望）"
            changed = True

    second_entry = bar_analysis.get("second_entry")
    if isinstance(second_entry, dict) and _normalize_second_entry(second_entry):
        changed = True

    return changed


def _normalize_second_entry(second_entry: dict[str, Any]) -> bool:
    """``type`` must be a string; models often emit null when ``is_second_entry`` is false."""
    raw_type = second_entry.get("type")
    if raw_type is not None and not (
        isinstance(raw_type, str) and not str(raw_type).strip()
    ):
        return False
    second_entry["type"] = "none"
    return True


def _normalize_always_in_value(
    raw: object,
    *,
    diagnosis_direction: str | None = None,
) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    key = text.lower().replace(" ", "")
    if key in ("long", "short", "neutral"):
        return key
    if "失效" in text or "invalid" in key or key in ("none", "n/a", "na"):
        return "neutral"
    if "ais" in key or "空头" in text:
        return "short"
    if "ail" in key or "多头" in text:
        return "long"
    if "bear" in key:
        return "short"
    if "bull" in key:
        return "long"
    if "中性" in text or key == "neutral":
        return "neutral"
    if diagnosis_direction == "bearish":
        return "short"
    if diagnosis_direction == "bullish":
        return "long"
    return None


def _normalize_stage2_enum_aliases(out: dict[str, Any]) -> bool:
    """Map common OpenClaw/Agent enum slips before schema validation."""
    changed = False
    diag = out.get("diagnosis_summary")
    diag_direction = (
        str(diag.get("direction", "")).strip()
        if isinstance(diag, dict)
        else ""
    ) or None

    bar_analysis = out.get("bar_analysis")
    if isinstance(bar_analysis, dict):
        raw_ai = bar_analysis.get("always_in")
        mapped_ai = _normalize_always_in_value(
            raw_ai, diagnosis_direction=diag_direction
        )
        if mapped_ai and mapped_ai != raw_ai:
            bar_analysis["always_in"] = mapped_ai
            logger.debug("always_in %r -> %r", raw_ai, mapped_ai)
            changed = True

    return changed


def _hoist_terminal_from_decision(out: dict[str, Any]) -> bool:
    """Move terminal nested under decision to the top level."""
    if isinstance(out.get("terminal"), dict):
        return False
    decision = out.get("decision")
    if not isinstance(decision, dict):
        return False
    nested = decision.pop("terminal", None)
    if not isinstance(nested, dict):
        return False
    out["terminal"] = nested
    logger.debug("Hoisted terminal from decision to top level")
    return True


def _ensure_decision_required_fields(
    out: dict[str, Any],
    *,
    stage1_json: dict[str, Any] | None = None,
) -> bool:
    """Fill missing decision sub-fields that commonly trigger schema retries."""
    decision = out.get("decision")
    if not isinstance(decision, dict):
        return False
    s1 = stage1_json or {}
    changed = False
    if not isinstance(decision.get("key_factors"), list):
        decision["key_factors"] = []
        changed = True
    if not isinstance(decision.get("watch_points"), list):
        decision["watch_points"] = []
        changed = True
    text_defaults = {
        "reasoning": "基于阶段一诊断与当前K线结构的阶段二决策说明",
        "diagnosis_confidence_reasoning": (
            str(s1.get("htf_context") or "").strip()[:500]
            or "依据阶段一诊断与闸门结论"
        ),
        "risk_assessment": "见 watch_points 与 invalidation_condition",
    }
    for key, default in text_defaults.items():
        if not isinstance(decision.get(key), str) or not str(decision.get(key)).strip():
            decision[key] = default
            changed = True
    if decision.get("diagnosis_confidence") is None:
        try:
            decision["diagnosis_confidence"] = int(s1.get("diagnosis_confidence") or 50)
        except (TypeError, ValueError):
            decision["diagnosis_confidence"] = 50
        changed = True
    if decision.get("trade_confidence") is None:
        decision["trade_confidence"] = (
            0 if decision.get("order_type") == "不下单" else 50
        )
        changed = True
    if (
        not isinstance(decision.get("trade_confidence_reasoning"), str)
        or not decision["trade_confidence_reasoning"].strip()
    ):
        decision["trade_confidence_reasoning"] = (
            "无入场计划，不存在交易信心"
            if decision.get("order_type") == "不下单"
            else "基于结构与入场方案的综合评估"
        )
        changed = True
    if decision.get("order_type") == "不下单":
        if "estimated_win_rate" not in decision:
            decision["estimated_win_rate"] = None
            changed = True
        if decision.get("estimated_win_rate_reasoning") is not None and not isinstance(
            decision.get("estimated_win_rate_reasoning"), str
        ):
            decision["estimated_win_rate_reasoning"] = None
            changed = True
        elif "estimated_win_rate_reasoning" not in decision:
            decision["estimated_win_rate_reasoning"] = None
            changed = True
    terminal = out.get("terminal")
    if isinstance(terminal, dict) and not str(terminal.get("label") or "").strip():
        outcome = str(terminal.get("outcome") or "wait")
        terminal["label"] = {
            "trade": "执行下单方案",
            "reject": "交易者方程未通过",
            "wait": "等待更好 setup",
            "proceed": "继续评估",
        }.get(outcome, "阶段二终局")
        changed = True
    return changed


def _truncate_decision_reasoning(decision: dict[str, Any]) -> bool:
    """Cap decision.reasoning length to avoid verbose JSON and schema failures."""
    reasoning = decision.get("reasoning")
    if not isinstance(reasoning, str):
        return False
    text = reasoning.strip()
    if len(text) <= DECISION_REASONING_MAX_LEN:
        if text != reasoning:
            decision["reasoning"] = text
            return True
        return False
    decision["reasoning"] = text[: DECISION_REASONING_MAX_LEN - 1] + "…"
    return True


def _normalize_next_cycle_prediction(
    prediction: dict[str, Any],
    *,
    stage1_json: dict[str, Any] | None = None,
) -> None:
    """In-place normalize next_cycle_prediction common model quirks. Idempotent."""
    from pa_agent.ai.cycle_enums import CYCLE_ORDER

    if not isinstance(prediction, dict):
        return

    # 0. Migrate primary/secondary shorthand → cycle + probabilities
    primary = prediction.pop("primary", None)
    prediction.pop("primary_probability", None)
    prediction.pop("secondary", None)
    prediction.pop("secondary_probability", None)
    if primary and not prediction.get("cycle"):
        prediction["cycle"] = str(primary).strip().lower()
    prediction.setdefault("unpredictable", False)

    # 1. unpredictable fallback
    unpredictable = bool(prediction.get("unpredictable", False))
    prediction["unpredictable"] = unpredictable

    # 2. features_used: ensure list, dedup, minimum set, filter invalid values
    feats = prediction.get("features_used")
    if not isinstance(feats, list):
        feats = []
    feats = [f for f in feats if isinstance(f, str)]
    # Filter out values not in the schema enum (e.g. "detected_patterns")
    invalid_feats = [f for f in feats if f not in _VALID_FEATURES_USED]
    if invalid_feats:
        logger.debug(
            "next_cycle_prediction.features_used dropped invalid values: %s",
            invalid_feats,
        )
    feats = [f for f in feats if f in _VALID_FEATURES_USED]
    if "stage1_diagnosis" not in feats:
        feats.insert(0, "stage1_diagnosis")
    seen: set[str] = set()
    deduped: list[str] = []
    for f in feats:
        if f not in seen:
            deduped.append(f)
            seen.add(f)
    prediction["features_used"] = deduped

    # 3. reasoning truncation
    reasoning = prediction.get("reasoning")
    if isinstance(reasoning, str) and len(reasoning) > 1500:
        prediction["reasoning"] = reasoning[:1499] + "…"
    elif not isinstance(reasoning, str):
        prediction["reasoning"] = ""

    if unpredictable:
        # unpredictable → force cycle / direction / probabilities = null
        prediction["cycle"] = None
        prediction["direction"] = None
        prediction["probabilities"] = None
        return

    # 4. probabilities integer rounding, clamping, and sum normalization
    probs = prediction.get("probabilities")
    if not unpredictable and not isinstance(probs, dict):
        cycle_guess = str(
            prediction.get("cycle")
            or (stage1_json or {}).get("cycle_position")
            or "trading_range"
        ).strip().lower()
        prediction["probabilities"] = _default_cycle_probs(cycle_guess)
        probs = prediction["probabilities"]
        logger.debug(
            "Synthesized next_cycle_prediction.probabilities from cycle=%r",
            cycle_guess,
        )
    if isinstance(probs, dict):
        normalized: dict[str, int] = {}
        for key in CYCLE_ORDER:
            raw = probs.get(key)
            try:
                value = int(round(float(raw))) if raw is not None else 0
            except (TypeError, ValueError):
                value = 0
            normalized[key] = max(0, min(100, value))

        # Auto-rescale if sum is outside [99, 101] (model arithmetic error)
        total = sum(normalized[k] for k in CYCLE_ORDER)
        if total > 0 and not (99 <= total <= 101):
            scale = 100.0 / total
            rescaled = {k: int(round(normalized[k] * scale)) for k in CYCLE_ORDER}
            # Fix rounding residual so sum == 100
            diff = 100 - sum(rescaled[k] for k in CYCLE_ORDER)
            if diff != 0:
                # Add/subtract from the largest bucket
                biggest = max(CYCLE_ORDER, key=lambda k: rescaled[k])
                rescaled[biggest] = max(0, rescaled[biggest] + diff)
            normalized = rescaled
            logger.debug(
                "next_cycle_prediction probabilities rescaled (sum was %d -> 100)", total
            )

        prediction["probabilities"] = normalized

        if not prediction.get("cycle"):
            max_value = max(normalized[k] for k in CYCLE_ORDER)
            prediction["cycle"] = next(k for k in CYCLE_ORDER if normalized[k] == max_value)

        # 5. cycle = argmax, tie-break by CYCLE_ORDER literal order
        max_value = max(normalized[k] for k in CYCLE_ORDER)
        # First winner in CYCLE_ORDER order
        argmax_cycle = next(k for k in CYCLE_ORDER if normalized[k] == max_value)

        model_cycle = str(prediction.get("cycle") or "").strip().lower()
        if model_cycle != argmax_cycle:
            logger.debug(
                "next_cycle_prediction cycle %r -> %r (argmax of %s)",
                model_cycle, argmax_cycle, normalized,
            )
            prediction["cycle"] = argmax_cycle

    # direction: keep model value; only type-coerce non-string to None
    direction = prediction.get("direction")
    if direction is not None and not isinstance(direction, str):
        prediction["direction"] = None


def _normalize_next_bar_prediction(prediction: dict[str, Any]) -> None:
    """In-place normalize next_bar_prediction common model quirks. Idempotent."""
    if not isinstance(prediction, dict):
        return

    # -1. Detect next_cycle_prediction content mistakenly placed here.
    #     If probabilities has cycle-position keys (spike/broad_channel/etc.) instead of
    #     direction keys (bullish/bearish/neutral), the model confused the two fields.
    #     Wipe probabilities so the synthesize-from-direction fallback kicks in.
    _CYCLE_KEYS = frozenset({
        "spike", "micro_channel", "tight_channel", "normal_channel",
        "broad_channel", "trending_tr", "trading_range", "extreme_tr",
    })
    _BAR_PROB_KEYS = frozenset({"bullish", "bearish", "neutral"})
    probs_raw = prediction.get("probabilities")
    if isinstance(probs_raw, dict):
        probs_keys = set(probs_raw.keys())
        if probs_keys & _CYCLE_KEYS and not (probs_keys & _BAR_PROB_KEYS):
            logger.debug(
                "next_bar_prediction.probabilities contains cycle keys (%s); "
                "replacing with direction-based default (model confused next_bar with next_cycle)",
                list(probs_keys & _CYCLE_KEYS)[:3],
            )
            # Replace with direction-based default right away
            raw_dir = str(prediction.get("direction") or "neutral").strip().lower()
            prediction["probabilities"] = _default_bar_probs(raw_dir)
            # Also remove cycle-specific fields that don't belong here
            prediction.pop("cycle", None)

    # 0. Extract probabilities from 'scenarios' dict if present and probabilities missing.
    #    Some models output: scenarios: {bullish: {probability: 30}, bearish: {probability: 35}, ...}
    if not isinstance(prediction.get("probabilities"), dict):
        scenarios = prediction.get("scenarios")
        if isinstance(scenarios, dict):
            extracted: dict[str, int] = {}
            for key in ("bullish", "bearish", "neutral"):
                s = scenarios.get(key)
                if isinstance(s, dict):
                    try:
                        extracted[key] = int(round(float(s.get("probability") or 0)))
                    except (TypeError, ValueError):
                        extracted[key] = 0
                else:
                    extracted[key] = 0
            if any(v > 0 for v in extracted.values()):
                prediction["probabilities"] = extracted
                logger.debug(
                    "next_bar_prediction: extracted probabilities from 'scenarios' dict"
                )

    # 1. unpredictable fallback
    unpredictable = bool(prediction.get("unpredictable", False))
    prediction["unpredictable"] = unpredictable

    # 2. features_used: ensure list, dedup, minimum set, filter invalid values
    feats = prediction.get("features_used")
    if not isinstance(feats, list):
        feats = []
    feats = [f for f in feats if isinstance(f, str)]
    # Filter out values not in the schema enum (e.g. "detected_patterns")
    invalid_feats = [f for f in feats if f not in _VALID_FEATURES_USED]
    if invalid_feats:
        logger.debug(
            "next_bar_prediction.features_used dropped invalid values: %s",
            invalid_feats,
        )
    feats = [f for f in feats if f in _VALID_FEATURES_USED]
    if "stage1_diagnosis" not in feats:
        feats.insert(0, "stage1_diagnosis")
    seen: set[str] = set()
    deduped: list[str] = []
    for f in feats:
        if f not in seen:
            deduped.append(f)
            seen.add(f)
    prediction["features_used"] = deduped

    # 3. reasoning truncation (R7.6)
    reasoning = prediction.get("reasoning")
    if isinstance(reasoning, str) and len(reasoning) > 1500:
        prediction["reasoning"] = reasoning[:1499] + "…"
    elif not isinstance(reasoning, str):
        prediction["reasoning"] = ""

    if unpredictable:
        # unpredictable → force direction / probabilities = null
        prediction["direction"] = None
        prediction["probabilities"] = None
        return

    # 3b. Legacy field migration: some models output separate probability keys
    #     instead of the required nested dict.
    #     e.g. bullish_probability/bearish_probability/neutral_probability + analysis
    if not isinstance(prediction.get("probabilities"), dict):
        bp = prediction.get("bullish_probability")
        berp = prediction.get("bearish_probability")
        np_ = prediction.get("neutral_probability")
        if any(v is not None for v in (bp, berp, np_)):
            try:
                prediction["probabilities"] = {
                    "bullish": int(round(float(bp or 0))),
                    "bearish": int(round(float(berp or 0))),
                    "neutral": int(round(float(np_ or 0))),
                }
                logger.debug(
                    "next_bar_prediction: migrated legacy flat probability fields -> probabilities dict"
                )
            except (TypeError, ValueError):
                pass  # leave for validator to catch

    # 3c. Legacy field migration: "analysis" -> "reasoning"
    if not isinstance(prediction.get("reasoning"), str) or not prediction["reasoning"]:
        analysis = prediction.get("analysis")
        if isinstance(analysis, str) and analysis:
            prediction["reasoning"] = analysis
            logger.debug(
                "next_bar_prediction: migrated 'analysis' field -> 'reasoning'"
            )

    # 4. probabilities integer rounding (R3.1)
    probs = prediction.get("probabilities")
    if isinstance(probs, dict):
        normalized: dict[str, int] = {}
        bar_order = ("bullish", "bearish", "neutral")
        for key in bar_order:
            raw = probs.get(key)
            try:
                value = int(round(float(raw))) if raw is not None else 0
            except (TypeError, ValueError):
                value = 0
            normalized[key] = max(0, min(100, value))

        # Auto-rescale if sum is outside [99, 101] (model arithmetic error)
        total = sum(normalized[k] for k in bar_order)
        if total > 0 and not (99 <= total <= 101):
            scale = 100.0 / total
            rescaled = {k: int(round(normalized[k] * scale)) for k in bar_order}
            diff = 100 - sum(rescaled[k] for k in bar_order)
            if diff != 0:
                biggest = max(bar_order, key=lambda k: rescaled[k])
                rescaled[biggest] = max(0, rescaled[biggest] + diff)
            normalized = rescaled
            logger.debug(
                "next_bar_prediction probabilities rescaled (sum was %d -> 100)", total
            )

        prediction["probabilities"] = normalized

        # 5. direction = argmax (R3.3) — respect model choice on ties
        order = ("bullish", "bearish", "neutral")
        max_value = max(normalized[k] for k in order)
        tied_winners = [k for k in order if normalized[k] == max_value]
        model_direction = str(prediction.get("direction") or "").strip().lower()

        if len(tied_winners) > 1:
            # Tie: preserve model's choice if it's one of the winners
            if model_direction in tied_winners:
                pass  # keep model's semantic choice
            else:
                # Model direction not in tied set — override with first winner
                logger.warning(
                    "next_bar_prediction direction=%r not in tied winners %s "
                    "(probs=%s); overriding to %r",
                    model_direction, tied_winners, normalized, tied_winners[0],
                )
                prediction["direction"] = tied_winners[0]
        else:
            # Clear winner
            expected = tied_winners[0]
            if model_direction != expected:
                logger.debug(
                    "next_bar_prediction direction %r -> %r (argmax of %s)",
                    model_direction, expected, normalized,
                )
                prediction["direction"] = expected
            # else: model direction matches argmax, no change needed
    # else: unparseable probabilities with unpredictable=False — leave for validator

    # 6. Strip extra keys not allowed by the schema (additionalProperties: false).
    #    This prevents schema validation failures caused by model adding creative fields
    #    like 'bar_type', 'key_levels', 'scenarios', 'confidence', 'analysis', etc.
    _ALLOWED_KEYS = frozenset({
        "direction", "probabilities", "reasoning", "unpredictable", "features_used",
    })
    extra_keys = [k for k in list(prediction.keys()) if k not in _ALLOWED_KEYS]
    if extra_keys:
        for k in extra_keys:
            del prediction[k]
        logger.debug(
            "next_bar_prediction: removed extra keys not allowed by schema: %s",
            extra_keys,
        )


_BAR_DIRECTION_ALIASES: dict[str, str] = {
    "up": "bullish",
    "long": "bullish",
    "bull": "bullish",
    "down": "bearish",
    "short": "bearish",
    "bear": "bearish",
    "sideways": "neutral",
    "flat": "neutral",
    "mixed": "neutral",
    "neutral_to_bullish": "bullish",
    "neutral_to_bearish": "bearish",
    "阴线": "bearish",
    "阳线": "bullish",
    "中性": "neutral",
    "阴": "bearish",
    "阳": "bullish",
    "看跌": "bearish",
    "看涨": "bullish",
}


def _alias_bar_direction(raw: Any) -> str | None:
    text = str(raw or "").strip()
    if not text:
        return None
    for key in (text, text.lower()):
        if key in _BAR_DIRECTION_ALIASES:
            return _BAR_DIRECTION_ALIASES[key]
    return None


def _probabilities_from_singular(direction: str, value: Any) -> dict[str, int] | None:
    """Build a probabilities dict from model shorthand ``probability: 60``."""
    try:
        p = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    p = max(0, min(100, p))
    dom = _alias_bar_direction(direction) or str(direction or "").strip().lower()
    if dom not in ("bullish", "bearish", "neutral"):
        return None
    probs = {"bullish": 0, "bearish": 0, "neutral": 0}
    probs[dom] = p
    rest = 100 - p
    others = [k for k in ("bullish", "bearish", "neutral") if k != dom]
    probs[others[0]] = rest // 2
    probs[others[1]] = rest - rest // 2
    return probs


def _repair_next_bar_prediction_shape(prediction: dict[str, Any]) -> bool:
    """Migrate common shorthand (阴线/阳线, singular probability) before alien discard."""
    if not isinstance(prediction, dict):
        return False
    changed = False
    aliased = _alias_bar_direction(prediction.get("direction"))
    if aliased and prediction.get("direction") != aliased:
        prediction["direction"] = aliased
        changed = True
    if not isinstance(prediction.get("probabilities"), dict):
        probs = _probabilities_from_singular(
            str(prediction.get("direction") or ""),
            prediction.get("probability"),
        )
        if probs is not None:
            prediction["probabilities"] = probs
            prediction.pop("probability", None)
            changed = True
            logger.debug(
                "next_bar_prediction: migrated singular probability -> probabilities dict"
            )
    return changed


def _default_bar_probs(direction: str) -> dict[str, int]:
    d = (direction or "neutral").strip().lower()
    if d == "bullish":
        return {"bullish": 45, "bearish": 30, "neutral": 25}
    if d == "bearish":
        return {"bearish": 45, "bullish": 30, "neutral": 25}
    return {"neutral": 40, "bearish": 30, "bullish": 30}


def _default_cycle_probs(cycle: str) -> dict[str, int]:
    from pa_agent.ai.cycle_enums import CYCLE_ORDER

    c = (cycle or "unknown").strip().lower()
    base = {k: 0 for k in CYCLE_ORDER}
    if c in base:
        base[c] = 55
        rest = 45 // max(len(CYCLE_ORDER) - 1, 1)
        for k in CYCLE_ORDER:
            if k != c:
                base[k] = rest
        # fix sum
        diff = 100 - sum(base.values())
        base[c] = max(0, base[c] + diff)
    else:
        base["broad_channel"] = 30
        base["trading_range"] = 25
        base["normal_channel"] = 20
        base["trending_tr"] = 15
        base["spike"] = 10
    return base


def ensure_stage2_predictions(
    out: dict[str, Any],
    *,
    stage1_json: dict[str, Any] | None = None,
    skip_next_bar: bool = False,
) -> bool:
    """Inject next_bar/next_cycle prediction stubs when the model omitted them.

    Parameters
    ----------
    skip_next_bar:
        When True, skip injecting ``next_bar_prediction`` (UI replay path when
        the user disabled the feature).  Schema validation always injects via
        ``skip_next_bar=False``; orchestrator strips the field before save when
        disabled.  ``next_cycle_prediction`` is always injected when missing.
    """
    changed = False
    diag = out.get("diagnosis_summary") if isinstance(out.get("diagnosis_summary"), dict) else {}
    s1 = stage1_json or {}
    direction = str(diag.get("direction") or s1.get("direction") or "neutral")
    cycle = str(diag.get("cycle_position") or s1.get("cycle_position") or "unknown")

    decision = out.get("decision") if isinstance(out.get("decision"), dict) else {}
    reasoning = str(decision.get("reasoning") or "").strip()
    synth_note = "（程序根据阶段二诊断摘要补全，原模型未输出预测字段）"

    if not skip_next_bar and not isinstance(out.get("next_bar_prediction"), dict):
        probs = _default_bar_probs(direction)
        dom = max(probs, key=probs.get)  # type: ignore[arg-type]
        out["next_bar_prediction"] = {
            "direction": dom,
            "probabilities": probs,
            "unpredictable": False,
            "reasoning": (
                (reasoning[:400] + "…") if len(reasoning) > 400 else reasoning
            ) or f"基于当前方向 {direction} 的参考预测{synth_note}",
            "features_used": ["stage1_diagnosis", "stage2_decision"],
        }
        changed = True

    if not isinstance(out.get("next_cycle_prediction"), dict):
        c_probs = _default_cycle_probs(cycle)
        dom_c = max(c_probs, key=c_probs.get)  # type: ignore[arg-type]
        out["next_cycle_prediction"] = {
            "cycle": dom_c,
            "direction": direction if direction in ("bullish", "bearish", "neutral") else "neutral",
            "probabilities": c_probs,
            "unpredictable": False,
            "reasoning": (
                f"当前周期 {cycle}，方向 {direction}。"
                f"下一周期概率为程序参考分布{synth_note}"
            ),
            "features_used": ["stage1_diagnosis", "stage2_decision"],
        }
        changed = True

    return changed


def _max_bar_seq_from_frame(kline_frame: Any) -> int | None:
    bars = getattr(kline_frame, "bars", None) if kline_frame is not None else None
    if not bars:
        return None
    seqs = [int(getattr(b, "seq", 0)) for b in bars if getattr(b, "seq", None)]
    return max(seqs) if seqs else None


def normalize_stage2(
    obj: dict[str, Any],
    *,
    normalization_mode: str = "strict",
    kline_frame: Any = None,
    decision_stance: str | None = None,
    stage1_json: dict[str, Any] | None = None,
    skip_next_bar: bool = False,
    previous_record: Any | None = None,
    structure_flip_cooldown_bars: int = 3,
    ignore_previous_context: bool = False,
) -> dict[str, Any]:
    """Return a copy of *obj* with decision_trace quirks corrected."""
    out = copy.deepcopy(obj)
    frame_max = _max_bar_seq_from_frame(kline_frame)
    _hoist_terminal_from_decision(out)
    _ensure_decision_required_fields(out, stage1_json=stage1_json)
    decision = out.get("decision")
    if isinstance(decision, dict):
        _truncate_decision_reasoning(decision)
    _normalize_stage2_enum_aliases(out)
    _normalize_stage2_bar_analysis_enums(out, stage1_json=stage1_json)
    # ── DecisionNodeEngine: fill §9.1/§9.2/§9.3/§9.5 ─────────────────────────
    if kline_frame is not None:
        try:
            from pa_agent.ai.decision_nodes import DecisionNodeEngine
            DecisionNodeEngine.apply_stage2(out, kline_frame, stage1_json)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DecisionNodeEngine.apply_stage2 failed: %s", exc)

    normalize_stage2_traces(
        out,
        normalization_mode=normalization_mode,
        default_max_seq=frame_max,
    )
    bar_analysis = out.get("bar_analysis")
    decision = out.get("decision")
    if isinstance(bar_analysis, dict):
        signal_bar = bar_analysis.get("signal_bar")
        if isinstance(signal_bar, dict):
            if not signal_bar.get("bar"):
                signal_bar["bar"] = None
                signal_bar.setdefault("quality", "invalid")
                signal_bar.setdefault("pattern", "none")

    # ── diagnosis_summary ────────────────────────────────────────────────
    # Schema requires diagnosis_summary; inject minimal default if missing.
    if not isinstance(out.get("diagnosis_summary"), dict):
        s1 = stage1_json or {}
        out["diagnosis_summary"] = {
            "cycle_position": s1.get("cycle_position", "unknown"),
            "direction": s1.get("direction", "neutral"),
            "key_signals": [],
        }
        logger.debug(
            "Injected missing diagnosis_summary from stage1 (cycle=%s, dir=%s)",
            out["diagnosis_summary"]["cycle_position"],
            out["diagnosis_summary"]["direction"],
        )

    ensure_stage2_predictions(out, stage1_json=stage1_json, skip_next_bar=skip_next_bar)

    pred = out.get("next_bar_prediction")
    if isinstance(pred, dict):
        # ── Step 1: shorthand repair (阴线/阳线, singular probability) ───────
        _repair_next_bar_prediction_shape(pred)

        # ── Step 2: migrate legacy flat probability fields ───────────────────
        # e.g. bullish_probability/bearish_probability/neutral_probability
        if not isinstance(pred.get("probabilities"), dict):
            bp = pred.get("bullish_probability")
            berp = pred.get("bearish_probability")
            np_ = pred.get("neutral_probability")
            if any(v is not None for v in (bp, berp, np_)):
                try:
                    pred["probabilities"] = {
                        "bullish": int(round(float(bp or 0))),
                        "bearish": int(round(float(berp or 0))),
                        "neutral": int(round(float(np_ or 0))),
                    }
                    logger.debug(
                        "next_bar_prediction: migrated legacy flat probability fields -> probabilities dict"
                    )
                except (TypeError, ValueError):
                    pass

        # ── Step 3: detect completely alien formats and discard ──────────────
        # A valid (or migratable) prediction must have at least one structural
        # key: probabilities (the canonical form), unpredictable, OR a valid
        # direction enum value.  If direction exists but is still non-standard
        # after aliasing, and probabilities is absent, treat as alien.
        _valid_directions = frozenset({"bullish", "bearish", "neutral"})
        has_probs = isinstance(pred.get("probabilities"), dict)
        has_unpredictable = "unpredictable" in pred
        direction_after_alias = str(pred.get("direction") or "").strip().lower()
        has_valid_direction = direction_after_alias in _valid_directions
        is_alien = not has_probs and not has_unpredictable and not has_valid_direction
        if is_alien:
            logger.debug(
                "next_bar_prediction has unrecognised schema (keys=%s); discarding and re-injecting",
                list(pred.keys()),
            )
            del out["next_bar_prediction"]
            ensure_stage2_predictions(
                out,
                stage1_json=stage1_json,
                skip_next_bar=False,
            )
            pred = out.get("next_bar_prediction")

        # ── Step 4: if direction is valid but probabilities still missing,
        #    synthesize probabilities from the direction value ─────────────────
        elif has_valid_direction and not has_probs and not has_unpredictable:
            pred["probabilities"] = _default_bar_probs(direction_after_alias)
            pred.setdefault("unpredictable", False)
            logger.debug(
                "next_bar_prediction: synthesized probabilities from direction=%r",
                direction_after_alias,
            )

    if isinstance(pred, dict):
        _normalize_next_bar_prediction(pred)

    pred_c = out.get("next_cycle_prediction")
    if isinstance(pred_c, dict):
        _normalize_next_cycle_prediction(pred_c, stage1_json=stage1_json)

    if kline_frame is not None and stage1_json and not ignore_previous_context:
        try:
            from pa_agent.ai.decision_continuity import (
                apply_continuity_guard,
                build_continuity_context,
            )

            ctx = build_continuity_context(
                frame=kline_frame,
                stage1_json=stage1_json,
                previous_record=previous_record,
                cooldown_bars=structure_flip_cooldown_bars,
            )
            out = apply_continuity_guard(out, ctx)
        except Exception as exc:  # noqa: BLE001
            logger.warning("apply_continuity_guard failed: %s", exc)

    return out
