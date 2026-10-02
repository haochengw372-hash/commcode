"""A route remains a candidate until its calibration evidence is supplied."""

from __future__ import annotations

import pytest

from commcode.codebook_protocol import build_jev_request, validate_decision_card


def card() -> dict:
    return {
        "code_id": "sentence_focus",
        "source_codebook_sha256": "a" * 64,
        "coding_unit": "target_sentence",
        "definition": "Which party does the target sentence refer to?",
        "final_labels": {
            "democrat": "Only Democrats", "republican": "Only Republicans",
            "both": "Both", "neither": "Neither",
        },
        "evidence_fields": ["target_sentence", "article_context"],
        "applicability": "Every nonempty sentence",
        "include": "Party, policy, or politician references",
        "exclude": "Do not infer party from sentiment alone",
        "boundary_cases": ["Party appears only in a neighboring sentence"],
        "missing_evidence_policy": "review",
        "proposed_route": "jev",
        "route_status": "candidate",
        "route_rationale": "A bounded referent decision with four categories",
        "jev_question": {
            "type": "choice",
            "instructions": (
                "Classify only `target_sentence`; use `article_context` for reference resolution."
            ),
            "criteria": {
                "democrat": "Democrats only", "republican": "Republicans only",
                "both": "Both parties", "neither": "Neither party",
            },
        },
    }


def test_candidate_card_is_valid_and_does_not_claim_validation():
    normalized = validate_decision_card(card())
    assert normalized["route_status"] == "candidate"
    assert list(normalized["final_labels"]) == ["democrat", "republican", "both", "neither"]


def test_validated_route_requires_calibration_manifest():
    item = card()
    item["route_status"] = "validated"
    with pytest.raises(ValueError, match="calibration manifest"):
        validate_decision_card(item)
    item["validation"] = {
        "manifest_sha256": "b" * 64,
        "calibration_split": "article-disjoint calibration",
        "metric_report": "reports/focus_calibration.json",
    }
    assert validate_decision_card(item)["route_status"] == "validated"


def test_choice_cannot_omit_final_category():
    item = card()
    del item["jev_question"]["criteria"]["neither"]
    with pytest.raises(ValueError, match="every final label"):
        validate_decision_card(item)


def test_jev_question_must_reference_available_evidence():
    item = card()
    item["jev_question"]["instructions"] = "Classify the unrelated headline."
    with pytest.raises(ValueError, match="evidence field"):
        validate_decision_card(item)


def test_llm_candidate_need_not_have_jev_question():
    item = card()
    item["proposed_route"] = "llm"
    del item["jev_question"]
    assert validate_decision_card(item)["proposed_route"] == "llm"


def test_noul_requires_yes_and_no_criteria():
    item = card()
    item["final_labels"] = {"present": "Present", "absent": "Absent"}
    item["jev_question"] = {
        "type": "noul", "instructions": "Does `target_sentence` mention a party?",
        "criteria": {"true": "Party present", "false": "No party"},
    }
    assert validate_decision_card(item)["jev_question"]["type"] == "noul"
    del item["jev_question"]["criteria"]["false"]
    with pytest.raises(ValueError, match="true and false"):
        validate_decision_card(item)


def test_request_excludes_human_labels_and_requires_evidence():
    item = card()
    request = build_jev_request(item, {
        "target_sentence": "Example", "article_context": "Context", "human_label": "both",
    }, model="jev-1.13.0")
    assert request["model"] == "jev-1.13.0"
    assert "human_label" not in request["state"]
    assert set(request["questions"]) == {"sentence_focus"}
    assert request["state"]["coding_rules"]["exclude"] == item["exclude"]
    with pytest.raises(ValueError, match="Missing evidence"):
        build_jev_request(item, {"target_sentence": "Example"})
