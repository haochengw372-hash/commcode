"""Validate provider answers without inventing labels for missing outputs."""

import json

from .models import ValidationError, finite_number
from .requests import cells


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("Duplicate JSON answer key")
        result[key] = value
    return result


def load_json(text):
    return json.loads(
        text,
        object_pairs_hook=unique_object,
        parse_constant=lambda x: (_ for _ in ()).throw(ValidationError("Nonfinite JSON value")),
    )


def bounded(value, lo, hi, name):
    value = finite_number(value, name)
    if not lo <= value <= hi:
        raise ValidationError(f"{name} outside original range")
    return value


def distribution(value, labels):
    if not isinstance(value, dict) or set(value) != set(labels):
        raise ValidationError("Probabilities must cover exactly the requested labels")
    result = {k: bounded(v, 0, 1, "probability") for k, v in value.items()}
    if abs(sum(result.values()) - 1) > 0.025:
        raise ValidationError("Probabilities do not sum to one")
    return result


def parse_jev_response(response, packet, book, targets=None):
    expected = cells(packet, book, targets)
    answers = response.get("answers")
    if not isinstance(answers, dict) or set(answers) != {x[0] for x in expected}:
        raise ValidationError("Missing or extra answers require review")
    result = []
    for key, _, unit, dim in expected:
        raw = answers[key]
        if not isinstance(raw, dict) or raw.get("type") != dim.kind:
            raise ValidationError("Primitive does not match original question")
        row = {
            "unit_id": unit["unit_id"],
            "dimension_id": dim.id,
            "kind": dim.kind,
            "estimand": dim.estimand,
            "raw": raw,
        }
        if dim.kind == "choice":
            label = raw.get("choice")
            if not isinstance(label, str) or label not in dim.labels:
                raise ValidationError("Invalid original label")
            row.update(
                label=label, probabilities=distribution(raw.get("probabilities"), dim.labels)
            )
        elif dim.kind == "noul":
            row["value"] = bounded(raw.get("noul"), 0, 1, "noul")
            row["probabilities"] = {"true": row["value"], "false": 1 - row["value"]}
        else:
            k = len(dim.criteria)
            native = bounded(raw.get("score"), 0, k - 1, "score")
            probabilities = distribution(raw.get("probabilities"), [str(i) for i in range(k)])
            # API probabilities are rounded; permit only the corresponding rounding error.
            weighted = sum(int(i) * p for i, p in probabilities.items())
            if abs(native - weighted) > max(0.03, 0.015 * k):
                raise ValidationError("Score disagrees with its probability-weighted index")
            legend = raw.get("legend")
            if not isinstance(legend, dict) or legend != dict(enumerate_anchors(dim.criteria)):
                raise ValidationError("Score legend differs from original ordered anchors")
            lo, hi = dim.output_range
            row.update(
                value=lo + native / (k - 1) * (hi - lo),
                native_score=native,
                output_range=dim.output_range,
                probabilities=probabilities,
            )
        if "confidence" in raw:
            row["confidence"] = bounded(raw["confidence"], 0, 1, "confidence")
        result.append(row)
    return result


def enumerate_anchors(criteria):
    return ((str(i), x) for i, x in enumerate(criteria))


def parse_llm_response(response, packet, book, targets=None):
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValidationError("Missing LLM message requires review") from exc
    parsed = load_json(content) if isinstance(content, str) else content
    expected = cells(packet, book, targets)
    if not isinstance(parsed, dict) or set(parsed) != {"answers"}:
        raise ValidationError("LLM output must contain only answers")
    answers = parsed["answers"]
    if not isinstance(answers, dict) or set(answers) != {x[0] for x in expected}:
        raise ValidationError("Missing or extra LLM answers require review")
    result = []
    for key, _, unit, dim in expected:
        value = answers[key]
        row = {
            "unit_id": unit["unit_id"],
            "dimension_id": dim.id,
            "kind": dim.kind,
            "estimand": dim.estimand,
            "raw": value,
        }
        if dim.kind == "choice":
            if not isinstance(value, str) or value not in dim.labels:
                raise ValidationError("LLM label outside original categories")
            row["label"] = value
        else:
            lo, hi = (0, 1) if dim.kind == "noul" else dim.output_range
            row["value"] = bounded(value, lo, hi, dim.kind)
        result.append(row)
    return result
