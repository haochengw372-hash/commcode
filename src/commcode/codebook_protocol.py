"""Validate researcher-approved, model-facing codebook decision cards.

This checks a card's structure. It does not infer construct meaning, select a
model from its name, or claim that a proposed route is empirically validated.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

ROUTES = frozenset({"jev", "llm", "hybrid", "human"})
STATUSES = frozenset({"candidate", "validated"})
MISSING_POLICIES = frozenset({"review", "unavailable", "not_applicable"})
QUESTION_TYPES = frozenset({"noul", "choice"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def validate_decision_card(card: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized copy or raise ValueError with the missing boundary.

    Validated routes need a calibration manifest; that manifest is still a
    research artifact whose performance must be checked separately.
    """
    if not isinstance(card, Mapping):
        raise ValueError("Decision card must be an object")
    required = {
        "code_id", "source_codebook_sha256", "coding_unit", "definition",
        "final_labels", "evidence_fields", "applicability", "include", "exclude",
        "boundary_cases", "missing_evidence_policy", "proposed_route",
        "route_status", "route_rationale",
    }
    missing = required - set(card)
    if missing:
        raise ValueError(f"Missing decision-card fields: {sorted(missing)}")
    for field in (
        "code_id", "coding_unit", "definition", "applicability", "include",
        "exclude", "route_rationale",
    ):
        if not isinstance(card[field], str) or not card[field].strip():
            raise ValueError(f"{field} must be nonempty text")
    if not isinstance(card["source_codebook_sha256"], str) or not _SHA256.fullmatch(
        card["source_codebook_sha256"]
    ):
        raise ValueError("source_codebook_sha256 must be a SHA-256 hex digest")
    labels = card["final_labels"]
    if (
        not isinstance(labels, Mapping)
        or not 2 <= len(labels) <= 255
        or any(not isinstance(key, str) or not key.strip()
               or not isinstance(value, str) or not value.strip()
               for key, value in labels.items())
    ):
        raise ValueError("final_labels needs 2-255 named, defined categories")
    fields = card["evidence_fields"]
    if (
        not isinstance(fields, (list, tuple)) or not fields
        or any(not isinstance(value, str) or not value.strip() for value in fields)
        or len(set(fields)) != len(fields)
    ):
        raise ValueError("evidence_fields must be unique, nonempty names")
    boundary_cases = card["boundary_cases"]
    if (
        not isinstance(boundary_cases, (list, tuple)) or not boundary_cases
        or any(not isinstance(value, str) or not value.strip() for value in boundary_cases)
    ):
        raise ValueError("boundary_cases needs at least one concrete boundary")
    if card["missing_evidence_policy"] not in MISSING_POLICIES:
        raise ValueError("Invalid missing_evidence_policy")
    if card["proposed_route"] not in ROUTES or card["route_status"] not in STATUSES:
        raise ValueError("Invalid proposed_route or route_status")
    if card["route_status"] == "validated":
        validation = card.get("validation")
        if not isinstance(validation, Mapping) or not all(
            isinstance(validation.get(key), str) and validation[key].strip()
            for key in ("manifest_sha256", "calibration_split", "metric_report")
        ) or not _SHA256.fullmatch(validation["manifest_sha256"]):
            raise ValueError("Validated route needs a calibration manifest and metric report")
    question = card.get("jev_question")
    if card["proposed_route"] in {"jev", "hybrid"} and question is None:
        raise ValueError("Jev or hybrid route needs an explicit Jev question")
    if question is not None:
        _validate_question(question, labels, fields)
    return {
        **dict(card),
        "final_labels": dict(labels),
        "evidence_fields": list(fields),
        "boundary_cases": list(boundary_cases),
    }


def _validate_question(question: Any, labels: Mapping[str, str], fields: list[str]) -> None:
    if not isinstance(question, Mapping) or question.get("type") not in QUESTION_TYPES:
        raise ValueError("jev_question needs a supported question type")
    instructions = question.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("jev_question needs complete instructions")
    if not any(re.search(r"\b" + re.escape(field) + r"\b", instructions) for field in fields):
        raise ValueError("jev_question must name an available evidence field")
    criteria = question.get("criteria")
    if not isinstance(criteria, Mapping) or any(
        not isinstance(value, str) or not value.strip() for value in criteria.values()
    ):
        raise ValueError("jev_question criteria must define each answer")
    if question["type"] == "choice" and set(criteria) != set(labels):
        raise ValueError("Choice criteria must match every final label")
    if question["type"] == "noul" and set(criteria) != {"true", "false"}:
        raise ValueError("Noul criteria must define true and false")


def build_jev_request(
    card: Mapping[str, Any], evidence: Mapping[str, Any], *, model: str = "jev-latest"
) -> dict[str, Any]:
    """Build a minimal TypeSafe request for one candidate Jev coding item.

    Extra evidence keys, including human labels, never enter the request. This
    prepares an evaluation; it does not approve a candidate route for use.
    """
    validated = validate_decision_card(card)
    if validated["proposed_route"] not in {"jev", "hybrid"}:
        raise ValueError("This card has no Jev route")
    if not isinstance(evidence, Mapping):
        raise ValueError("Evidence must be a mapping")
    fields = validated["evidence_fields"]
    absent = [field for field in fields if field not in evidence]
    if absent:
        raise ValueError(f"Missing evidence fields: {absent}")
    empty = [field for field in fields if evidence[field] is None or evidence[field] == ""]
    if empty:
        raise ValueError(f"Empty evidence fields: {empty}")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("A Jev model name is required")
    return {
        "model": model,
        "state": {
            **{field: evidence[field] for field in fields},
            "coding_rules": {
                "definition": validated["definition"],
                "include": validated["include"],
                "exclude": validated["exclude"],
                "applicability": validated["applicability"],
            },
        },
        "questions": {validated["code_id"]: validated["jev_question"]},
    }
