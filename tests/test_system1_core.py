import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from commcode.system1 import (
    Book,
    DispatchUncertain,
    EvidencePacket,
    Journal,
    Provider,
    System1Encoder,
    ValidationError,
    build_jev_request,
    build_llm_request,
    parse_jev_response,
    parse_llm_response,
)


def inputs():
    book = Book.from_dict(
        {
            "book_id": "neutral",
            "original_text": "Agree or disagree. Low. High.",
            "dimensions": [
                {
                    "id": "stance",
                    "question": "Is the view endorsed?",
                    "kind": "choice",
                    "labels": ["Neutral", "Agree", "Disagree"],
                    "criteria": {"Neutral": None, "Agree": "Agree", "Disagree": "disagree"},
                },
                {
                    "id": "fraction",
                    "question": "Would a rater agree?",
                    "kind": "noul",
                    "estimand": "rater_fraction",
                },
                {
                    "id": "rating",
                    "question": "How high?",
                    "kind": "score",
                    "criteria": ["Low.", "High."],
                    "output_range": [0, 1],
                    "estimand": "continuous_rating",
                },
            ],
        }
    )
    packet = EvidencePacket.from_dict(
        {
            "units": [
                {
                    "unit_id": "id1",
                    "group_id": "subject9",
                    "text": "A quote followed by rebuttal.",
                    "context": {
                        "article": "Full article " * 1000,
                        "gold": "Agree",
                        "speaker_id": "secret",
                        "nested": {"rater_id": "private", "inferences": "Quoted view"},
                    },
                }
            ]
        }
    )
    return packet, book


def response():
    return {
        "model": "jev-actual",
        "answers": {
            "u0_d0": {
                "type": "choice",
                "choice": "Neutral",
                "probabilities": {"Neutral": 0.8, "Agree": 0.1, "Disagree": 0.1},
                "confidence": 0.6,
            },
            "u0_d1": {"type": "noul", "noul": 0.7},
            "u0_d2": {
                "type": "score",
                "score": 0.25,
                "probabilities": {"0": 0.75, "1": 0.25},
                "legend": {"0": "Low.", "1": "High."},
                "confidence": 0.5,
            },
        },
        "usage": {"input_tokens": 10, "output_tokens": 3},
    }


def test_input_preservation_and_metadata_whitelist():
    packet, book = inputs()
    body = build_jev_request(packet, book)
    encoded = json.dumps(body)
    assert "subject9" not in encoded and "id1" not in encoded and "private" not in encoded
    assert "secret" not in encoded and "gold" not in encoded
    assert body["state"]["units"][0]["context"]["article"] == "Full article " * 1000
    assert body["state"]["units"][0]["context"]["nested"]["inferences"] == "Quoted view"
    assert len(body["questions"]) == 3
    assert "units[0]" in body["questions"]["u0_d0"]["instructions"]
    assert "Is the view endorsed?" in body["questions"]["u0_d0"]["instructions"]
    assert Book.from_dict(book.to_dict()) == book


def test_prompt_families_and_original_labels():
    packet, book = inputs()
    compact = build_jev_request(packet, book)
    anchored = build_jev_request(packet, book, prompt_family="anchored")
    verbose = build_jev_request(packet, book, prompt_family="verbose")
    boundary = build_jev_request(packet, book, prompt_family="boundary")
    assert len({json.dumps(x) for x in (compact, anchored, verbose, boundary)}) == 4
    assert compact["questions"]["u0_d0"]["criteria"]["Neutral"] is None
    assert anchored["questions"]["u0_d0"]["criteria"]["Agree"] == "Agree"
    assert "Target text" in verbose["questions"]["u0_d0"]["instructions"]
    assert "Independently verify" in boundary["questions"]["u0_d0"]["instructions"]
    targets = [{"unit_id": "id1", "dimension_id": "stance"}]
    assert set(build_jev_request(packet, book, targets)["questions"]) == {"u0_d0"}


def test_typed_primitives_native_noul_and_original_score_range():
    packet, book = inputs()
    result = parse_jev_response(response(), packet, book)
    assert result[0]["label"] == "Neutral"
    assert result[1]["value"] == 0.7 and "confidence" not in result[1]
    assert result[2]["value"] == 0.25
    book.dimensions[2].output_range = [1, 7]
    assert parse_jev_response(response(), packet, book)[2]["value"] == 2.5


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), -1, 2])
def test_invalid_noul_requires_review(bad):
    packet, book = inputs()
    answer = response()
    answer["answers"]["u0_d1"]["noul"] = bad
    with pytest.raises(ValidationError):
        parse_jev_response(answer, packet, book)


def test_missing_extra_duplicate_and_illegal_outputs():
    packet, book = inputs()
    for key in ("missing", "extra", "illegal"):
        answer = response()
        if key == "missing":
            del answer["answers"]["u0_d0"]
        elif key == "extra":
            answer["answers"]["extra"] = {}
        else:
            answer["answers"]["u0_d0"]["choice"] = "0"
        with pytest.raises(ValidationError):
            parse_jev_response(answer, packet, book)
    llm = {"choices": [{"message": {"content": '{"answers":{"u0_d0":"Agree","u0_d0":"Neutral"}}'}}]}
    with pytest.raises(ValidationError):
        parse_llm_response(llm, packet, book)


def test_score_limits_source_spans_and_missing_evidence():
    packet, book = inputs()
    for count in (1, 11):
        book.dimensions[2].criteria = ["anchor"] * count
        with pytest.raises(ValidationError):
            book.validate()
    book = inputs()[1]
    book.dimensions[0].source_spans = "invented"
    with pytest.raises(ValidationError):
        book.validate()
    with pytest.raises(ValidationError):
        EvidencePacket.from_dict([{"unit_id": "missing", "text": ""}])


def test_llm_original_scale_and_exact_scalar_output():
    packet, book = inputs()
    body = build_llm_request(packet, book)
    assert body["temperature"] == 0
    llm = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {"answers": {"u0_d0": "Neutral", "u0_d1": 0.7, "u0_d2": 0.25}}
                    )
                }
            }
        ]
    }
    result = parse_llm_response(llm, packet, book)
    assert [x.get("label", x.get("value")) for x in result] == ["Neutral", 0.7, 0.25]


def test_paid_raw_reparse_no_duplicate_and_thread_safe_claim(tmp_path):
    packet, book = inputs()
    calls = []
    provider = Provider(transport=lambda *args: calls.append(args) or response())
    body = build_jev_request(packet, book)
    journal = Journal(tmp_path)
    with ThreadPoolExecutor(2) as pool:
        records = list(pool.map(lambda _: provider.invoke(body, journal), range(2)))
    assert len(calls) == 1
    assert sorted(x["cache_hit"] for x in records) == [False, True]
    assert "Authorization" not in json.dumps(records)
    assert records[0]["response"]["model"] == "jev-actual"
    parse_jev_response(Provider.response(records[0]), packet, book)
    provider.invoke(body, journal, cache_epoch="fresh")
    assert len(calls) == 2


def test_paid_invalid_json_survives_and_never_resends(tmp_path):
    packet, book = inputs()
    calls = []
    provider = Provider(
        transport=lambda *args: calls.append(args) or {"_http_status": 200, "_http_body": "{broken"}
    )
    body = build_jev_request(packet, book)
    record = provider.invoke(body, Journal(tmp_path))
    with pytest.raises(ValueError):
        Provider.response(record)
    assert provider.invoke(body, Journal(tmp_path))["cache_hit"]
    assert len(calls) == 1
    assert "{broken" in (tmp_path / f"{record['signature']}.json").read_text()


def test_uncertain_dispatch_never_automatically_retries(tmp_path):
    packet, book = inputs()
    calls = []

    def fail(*args):
        calls.append(args)
        raise TimeoutError("secret-not-logged")

    provider = Provider(transport=fail)
    for _ in range(2):
        with pytest.raises(DispatchUncertain):
            provider.invoke(build_jev_request(packet, book), Journal(tmp_path))
    assert len(calls) == 1
    saved = next(tmp_path.glob("*.json")).read_text()
    assert "secret-not-logged" not in saved
    assert "uncertain" in saved


def test_endpoint_changes_and_corruption_checked(tmp_path):
    packet, book = inputs()
    body = build_jev_request(packet, book)
    provider = Provider(transport=lambda *args: response())
    record = provider.invoke(body, Journal(tmp_path))
    second = Provider(
        endpoint="https://other.example/v1/systemone", transport=lambda *args: response()
    )
    assert second.invoke(body, Journal(tmp_path))["signature"] != record["signature"]
    path = tmp_path / f"{record['signature']}.json"
    corrupted = json.loads(path.read_text())
    corrupted["response"]["model"] = "tampered"
    path.write_text(json.dumps(corrupted))
    with pytest.raises(ValidationError):
        provider.invoke(body, Journal(tmp_path))


def test_encoder_exact_target_merge_and_postprocess_coverage(tmp_path):
    packet, book = inputs()
    jev = Provider(transport=lambda *args: response())
    llm = Provider(
        kind="llm",
        transport=lambda *args: {
            "choices": [{"message": {"content": '{"answers":{"u0_d0":"Disagree"}}'}}]
        },
    )
    engine = System1Encoder(jev, llm)
    result = engine.encode(
        packet,
        book,
        output_dir=tmp_path,
        selector=lambda rows: [{"unit_id": "id1", "dimension_id": "stance"}],
    )
    assert result["predictions"][0]["label"] == "Disagree"
    assert len(result["stages"]) == 2
    with pytest.raises(ValidationError):
        engine.encode(packet, book, output_dir=tmp_path, postprocess=lambda rows: rows[:-1])
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "review_required"


def test_explicit_rejected_retry_preserves_first_attempt(tmp_path):
    packet, book = inputs()
    calls = []

    def transport(*args):
        calls.append(args)
        if len(calls) == 1:
            return {"_http_status": 429, "_http_body": '{"error":"rate limited"}'}
        return response()

    from commcode.system1 import ProviderRejected

    provider = Provider(transport=transport)
    body = build_jev_request(packet, book)
    journal = Journal(tmp_path)
    with pytest.raises(ProviderRejected):
        provider.invoke(body, journal)
    with pytest.raises(DispatchUncertain):
        provider.invoke(body, journal)
    assert len(calls) == 1
    second = provider.invoke(body, journal, retry_rejected=True)
    assert second["attempt"] == 2
    assert len(list(tmp_path.glob("*.attempt1.json"))) == 1
    assert provider.invoke(body, journal)["cache_hit"]
    assert len(calls) == 2


def test_selector_outside_initial_coverage_rejected_before_llm(tmp_path):
    packet, book = inputs()
    jev_answer = response()
    jev_answer["answers"] = {"u0_d0": jev_answer["answers"]["u0_d0"]}
    calls = []
    engine = System1Encoder(
        Provider(transport=lambda *args: jev_answer),
        Provider(kind="llm", transport=lambda *args: calls.append(args)),
    )
    with pytest.raises(ValidationError):
        engine.encode(
            packet,
            book,
            output_dir=tmp_path,
            targets=[{"unit_id": "id1", "dimension_id": "stance"}],
            selector=lambda rows: [{"unit_id": "id1", "dimension_id": "rating"}],
        )
    assert not calls


def test_source_span_matches_declared_field_and_source_answer_mapping():
    packet, book = inputs()
    book.dimensions[0].source_spans = [{"field": "criteria.Agree", "start": 0, "end": 5}]
    book.validate()
    book.dimensions[0].criteria["Agree"] = "changed"
    with pytest.raises(ValidationError):
        book.validate()
    dim = inputs()[1].dimensions[0]
    dim.original_answer_mapping = {"Neutral": "No opinion", "Agree": "Yes", "Disagree": "No"}
    body = build_jev_request(
        packet, Book("mapped", "Yes. No. No opinion. Agree or disagree.", [dim])
    )
    assert body["state"]["dimensions"][0]["original_answer_mapping"]["Agree"] == "Yes"


def test_identical_full_context_stored_once_with_explicit_question_paths():
    packet, book = inputs()
    original = packet.units[0]["context"]["article"]
    packet.units.append(dict(packet.units[0], unit_id="id2", text="Different target."))
    request = build_jev_request(packet, book)
    assert len(request["state"]["shared_contexts"]) == 1
    assert request["state"]["shared_contexts"][0]["article"] == original
    assert request["state"]["units"][0]["context"] == {"shared_context_index": 0}
    assert request["state"]["units"][1]["context"] == {"shared_context_index": 0}
    assert "shared_contexts[0]" in request["questions"]["u1_d0"]["instructions"]
    assert "units[1].text" in request["questions"]["u1_d0"]["instructions"]
    assert packet.units[0]["context"]["article"] == original
    assert json.dumps(request).count(original) == 1
    llm = build_llm_request(packet, book)
    llm_state = json.loads(llm["messages"][1]["content"])["state"]
    assert llm_state == request["state"]


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example.com/v1",
        "https:///missing-host",
        "https://user:password@example.com/v1",
        "https://example.com/v1?api_key=secret",
    ],
)
def test_endpoint_security_rejects_remote_plaintext_and_credentials(endpoint):
    with pytest.raises(ValidationError):
        Provider(endpoint=endpoint)


def test_default_llm_configuration_and_ungrounded_criterion_rejected():
    packet, book = inputs()
    assert Provider(kind="llm").model == "deepseek-flash"
    request = build_llm_request(packet, book)
    assert request["model"] == "deepseek-flash"
    assert request["thinking"] == {"type": "disabled"}
    book.dimensions[0].criteria["Neutral"] = "a newly invented rule"
    with pytest.raises(ValidationError):
        build_jev_request(packet, book, prompt_family="anchored")
