"""Validate provider answers without inventing labels for missing outputs."""

import json

from .models import ValidationError, finite_number
from .requests import boolean_source, cells


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


def parse_jev_response(response, packet, book, targets=None, *, prompt_family=None):
    expected = cells(packet, book, targets)
    answers = response.get("answers")
    if not isinstance(answers, dict) or set(answers) != {x[0] for x in expected}:
        raise ValidationError("Missing or extra answers require review")
    result = []
    for key, _, unit, dim in expected:
        raw = answers[key]
        if not isinstance(raw, dict):
            raise ValidationError("Primitive answer must be an object")
        projection = (
            boolean_source(dim)
            if prompt_family == "native" and dim.kind == "choice" and raw.get("type") == "noul"
            else None
        )
        if raw.get("type") != dim.kind and projection is None:
            raise ValidationError("Primitive does not match original question")
        row = {
            "unit_id": unit["unit_id"],
            "dimension_id": dim.id,
            "kind": dim.kind,
            "estimand": dim.estimand,
            "raw": raw,
        }
        if projection is not None:
            positive = bounded(raw.get("noul"), 0, 1, "native noul")
            positive_label, negative_label = projection["true"], projection["false"]
            label = positive_label if positive > 0.5 else negative_label
            probabilities = {positive_label: positive, negative_label: 1 - positive}
            row.update(
                label=label,
                probabilities=probabilities,
                native_primitive="noul",
                primitive_actual="noul",
                native_value=positive,
                raw_noul=raw,
                native_projection={
                    "positive_label": positive_label,
                    "negative_label": negative_label,
                    "source": projection["source"],
                    "tie_policy": "negative_at_half",
                },
                raw_choice={"label": label, "probabilities": probabilities},
            )
        elif dim.kind == "choice":
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
        if "confidence" in raw and projection is None:
            row["confidence"] = bounded(raw["confidence"], 0, 1, "confidence")
        result.append(row)
    return result


def enumerate_anchors(criteria):
    return ((str(i), x) for i, x in enumerate(criteria))


LLM_NORMALIZATION_VERSION = "llm-normalize-1"


def normalize_choice(value, labels):
    if isinstance(value, str) and value in labels:
        return value, []
    if not isinstance(value, bool) and isinstance(value, (int, float)):
        numeric = finite_number(value, "numeric code")
        if numeric.is_integer():
            matching = []
            for label in labels:
                try:
                    if float(label) == numeric:
                        matching.append(label)
                except ValueError:
                    continue
            if len(matching) == 1 and matching[0] == str(int(numeric)):
                return matching[0], ["integer_code_to_original_string"]
    raise ValidationError("LLM label outside original categories or ambiguous numeric code")


def parse_llm_response(response, packet, book, targets=None):
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValidationError("Missing LLM message requires review") from exc
    parsed = load_json(content) if isinstance(content, str) else content
    expected = cells(packet, book, targets)
    expected_keys = {x[0] for x in expected}
    normalization = []
    if not isinstance(parsed, dict):
        raise ValidationError("LLM output must contain an answer object")
    if set(parsed) == {"answers"}:
        answers = parsed["answers"]
    elif set(parsed) == {"type", "answers"} and parsed["type"] == "json_object":
        answers = parsed["answers"]
        normalization.append("removed_json_object_type_wrapper")
    elif set(parsed) == expected_keys:
        answers = parsed
        normalization.append("wrapped_exact_top_level_answer_map")
    else:
        raise ValidationError("LLM output contains unexpected fields")
    if not isinstance(answers, dict) or set(answers) != expected_keys:
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
            row["label"], code_steps = normalize_choice(value, dim.labels)
            row["normalization"] = {
                "version": LLM_NORMALIZATION_VERSION,
                "steps": normalization + code_steps,
            }
        else:
            lo, hi = (0, 1) if dim.kind == "noul" else dim.output_range
            row["value"] = bounded(value, lo, hi, dim.kind)
            row["normalization"] = {
                "version": LLM_NORMALIZATION_VERSION,
                "steps": normalization.copy(),
            }
        result.append(row)
    return result
