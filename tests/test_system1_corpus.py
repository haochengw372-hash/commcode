import json
import math
from threading import Lock

import pytest

from commcode.system1 import Book, Provider, ValidationError, encode_corpus


def book(dimensions=1):
    return Book.from_dict(
        {
            "book_id": "generic",
            "original_text": "Is the assertion endorsed? Yes. No.",
            "dimensions": [
                {
                    "id": f"d{n}",
                    "question": "Is the assertion endorsed?",
                    "kind": "choice",
                    "labels": ["0", "1"],
                    "original_answer_mapping": {"0": "No", "1": "Yes"},
                }
                for n in range(dimensions)
            ],
        }
    )


def units(n, source="Complete source " * 500):
    return [
        {
            "unit_id": f"opaque-{i}",
            "group_id": "private-group",
            "text": f"Evidence number {i}.",
            "context": {"full_original_source": source, "rater_id": "private-rater"},
            "gold": "must-not-be-sent",
        }
        for i in range(n)
    ]


class MockTransport:
    def __init__(self, llm=False, malformed=False):
        self.llm = llm
        self.malformed = malformed
        self.calls = []
        self.lock = Lock()

    def __call__(self, endpoint, body, headers):
        with self.lock:
            self.calls.append(body)
        if self.malformed:
            return {"_http_status": 200, "_http_body": "not-json"}
        if self.llm:
            content = json.loads(body["messages"][1]["content"])
            return {
                "model": "actual-llm",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"answers": {key: "1" for key in content["questions"]}}
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            }
        answers = {}
        for key, question in body["questions"].items():
            if question["type"] == "noul":
                answers[key] = {"type": "noul", "noul": 0.2}
            else:
                answers[key] = {
                    "type": "choice",
                    "choice": "0",
                    "probabilities": {"0": 0.8, "1": 0.2},
                    "confidence": 0.6,
                }
        return {
            "model": "actual-jev",
            "answers": answers,
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }


def test_thousand_units_are_batched_all_retained_and_resume_without_new_calls(tmp_path):
    transport = MockTransport()
    provider = Provider(transport=transport)
    source_units = units(1000)
    result = encode_corpus(
        source_units,
        book(),
        output_dir=tmp_path,
        jev_provider=provider,
        prompt_family="native",
        workers=4,
    )
    assert result["status"] == "completed"
    assert len(result["predictions"]) == 1000
    assert len(transport.calls) == math.ceil(1000 / 16)
    assert result["usage"]["new_dispatches"] == len(transport.calls)
    assert result["usage"]["new_dispatch_tokens"]["input_tokens"] == 10 * len(transport.calls)
    assert result["wall_seconds"] > 0
    assert all(
        row["kind"] == "choice" and row["native_primitive"] == "noul"
        for row in result["predictions"]
    )
    for body in transport.calls:
        encoded = json.dumps(body)
        assert "private-rater" not in encoded and "private-group" not in encoded
        assert "must-not-be-sent" not in encoded and "opaque-" not in encoded
        assert len(body["state"]["shared_contexts"]) == 1
        assert (
            body["state"]["shared_contexts"][0]["full_original_source"]
            == (source_units[0]["context"]["full_original_source"])
        )
    before = len(transport.calls)
    replay = encode_corpus(
        source_units,
        book(),
        output_dir=tmp_path,
        jev_provider=provider,
        prompt_family="native",
        workers=4,
    )
    assert replay["status"] == "completed"
    assert replay["predictions"] == result["predictions"]
    assert len(transport.calls) == before
    assert replay["usage"]["new_dispatches"] == 0
    assert replay["usage"]["new_dispatch_tokens"] == {}


class FrozenCalibration:
    def __init__(self, codebook):
        self.dimensions = codebook.to_dict()["dimensions"]
        self.calls = []

    def to_dict(self):
        return {"test_frozen_model": True, "dimensions": self.dimensions}

    def apply_rows(self, rows):
        self.calls.append(rows)
        return [
            dict(
                row,
                probabilities={"0": 0.6, "1": 0.4},
                label="0",
                risk=int(row["unit_id"].split("-")[1]) / 20,
            )
            for row in rows
        ]


def test_global_packet_budget_calibrates_once_before_actual_target_teacher(tmp_path):
    jev_transport, llm_transport = MockTransport(), MockTransport(llm=True)
    codebook = book()
    calibration = FrozenCalibration(codebook)
    result = encode_corpus(
        units(16),
        codebook,
        output_dir=tmp_path,
        jev_provider=Provider(transport=jev_transport),
        llm_provider=Provider(kind="llm", transport=llm_transport),
        calibrator=calibration,
        prompt_family="native",
        max_units=4,
        policy={"budget_fraction": 0.25, "cell_fraction": 0.5},
    )
    assert result["status"] == "completed"
    assert len(jev_transport.calls) == 4
    assert len(llm_transport.calls) == 1
    assert len(calibration.calls) == 1
    assert result["packet_budget_cap"] == 1
    assert len(result["selected_targets"]) == 2
    assert {x["unit_id"] for x in result["selected_targets"]} == {"opaque-14", "opaque-15"}
    final = {row["unit_id"]: row for row in result["predictions"]}
    assert final["opaque-15"]["label"] == "1"  # Teacher was not recalibrated to base label 0.
    assert final["opaque-0"]["label"] == "0"
    teacher = json.loads(llm_transport.calls[0]["messages"][1]["content"])
    assert len(teacher["questions"]) == 2
    assert len(teacher["state"]["units"]) == 4
    assert "risk" not in teacher["state"] and "calibration" not in teacher["state"]
    assert result["usage"]["new_dispatches"] == 5
    replay = encode_corpus(
        units(16),
        codebook,
        output_dir=tmp_path,
        jev_provider=Provider(transport=jev_transport),
        llm_provider=Provider(kind="llm", transport=llm_transport),
        calibrator=calibration,
        prompt_family="native",
        max_units=4,
        policy={"budget_fraction": 0.25, "cell_fraction": 0.5},
    )
    assert replay["usage"]["new_dispatches"] == 0
    assert len(jev_transport.calls) == 4 and len(llm_transport.calls) == 1


def test_question_splitting_caps_actual_physical_calls_with_same_unit(tmp_path):
    jev_transport, llm_transport = MockTransport(), MockTransport(llm=True)
    result = encode_corpus(
        units(1),
        book(20),
        output_dir=tmp_path,
        jev_provider=Provider(transport=jev_transport),
        llm_provider=Provider(kind="llm", transport=llm_transport),
        cell_budget=5,
        prompt_family="native",
        policy={"budget_fraction": 0.5, "cell_fraction": 0.4},
    )
    assert result["status"] == "completed"
    assert result["base_packet_count"] == len(jev_transport.calls) == 4
    assert len(llm_transport.calls) == result["packet_budget_cap"] == 2
    assert len(result["predictions"]) == 20
    assert len(result["selected_targets"]) == 4
    questions = [key for body in jev_transport.calls for key in body["questions"]]
    assert len(set(questions)) == len(questions) == 20
    assert all(len(body["questions"]) == 5 for body in jev_transport.calls)


def test_single_target_too_large_requires_review_without_clipping_or_dispatch(tmp_path):
    transport = MockTransport()
    result = encode_corpus(
        units(1, source="Full source " * 1000),
        book(),
        output_dir=tmp_path,
        jev_provider=Provider(transport=transport),
        packet_budget=1000,
    )
    assert result["status"] == "review_required"
    assert result["predictions"] == []
    assert result["errors"][0]["reason"] == "single_target_envelope_exceeds_budget"
    assert not transport.calls


def test_byte_envelope_splitting_keeps_complete_source(tmp_path):
    transport = MockTransport()
    result = encode_corpus(
        units(16, source="Full source " * 100),
        book(5),
        output_dir=tmp_path,
        jev_provider=Provider(transport=transport),
        packet_budget=9000,
        prompt_family="native",
    )
    assert result["status"] == "completed"
    assert len(result["predictions"]) == 80
    assert len(transport.calls) > 1
    from commcode.system1.providers import canonical

    assert all(len(canonical(body)) <= 9000 for body in transport.calls)
    assert all("Full source " * 100 in json.dumps(body) for body in transport.calls)


def test_failed_paid_parse_is_preserved_and_resume_does_not_bill_again(tmp_path):
    transport = MockTransport(malformed=True)
    provider = Provider(transport=transport)
    result = encode_corpus(units(2), book(), output_dir=tmp_path, jev_provider=provider)
    assert result["status"] == "review_required"
    assert result["usage"]["new_dispatches"] == 1
    assert len(list((tmp_path / "journal").glob("*.json"))) == 1
    second = encode_corpus(units(2), book(), output_dir=tmp_path, jev_provider=provider)
    assert second["status"] == "review_required"
    assert second["usage"]["new_dispatches"] == 0
    assert len(transport.calls) == 1


def test_changed_workflow_requires_explicit_new_epoch(tmp_path):
    transport = MockTransport()
    provider = Provider(transport=transport)
    encode_corpus(units(2), book(), output_dir=tmp_path, jev_provider=provider)
    with pytest.raises(ValidationError):
        encode_corpus(units(2), book(), output_dir=tmp_path, jev_provider=provider, max_units=1)
    assert len(transport.calls) == 1
    changed = encode_corpus(
        units(2),
        book(),
        output_dir=tmp_path,
        jev_provider=provider,
        max_units=1,
        cache_epoch="explicit-new",
    )
    assert changed["status"] == "completed"
    assert len(list(tmp_path.glob("workflow.*.json"))) == 1


def test_unfitted_population_probability_is_not_used_as_human_disagreement_risk(tmp_path):
    transport = MockTransport()
    llm_transport = MockTransport(llm=True)
    numeric = Book.from_dict(
        {
            "book_id": "population",
            "original_text": "Do you agree?",
            "dimensions": [
                {
                    "id": "agreement",
                    "question": "Do you agree?",
                    "kind": "noul",
                    "estimand": "rater_fraction",
                }
            ],
        }
    )
    result = encode_corpus(
        units(4),
        numeric,
        output_dir=tmp_path,
        jev_provider=Provider(transport=transport),
        llm_provider=Provider(kind="llm", transport=llm_transport),
        policy={"budget_fraction": 1.0},
    )
    assert result["status"] == "completed"
    assert result["selected_targets"] == []
    assert not llm_transport.calls
    assert all(row["value"] == 0.2 for row in result["predictions"])


def test_context_grouping_keeps_full_source_and_restores_input_order(tmp_path):
    source_units = units(8)
    for i, unit in enumerate(source_units):
        unit["context"] = {"full_original_source": f"Complete source {i % 2}."}
    transport = MockTransport()
    result = encode_corpus(
        source_units,
        book(),
        output_dir=tmp_path,
        jev_provider=Provider(transport=transport),
        max_units=4,
    )
    assert result["status"] == "completed"
    assert len(transport.calls) == 2
    assert all(len(body["state"]["shared_contexts"]) == 1 for body in transport.calls)
    assert [r["unit_id"] for r in result["predictions"]] == [u["unit_id"] for u in source_units]


def test_cli_corpus_real_local_http_and_cache_resume(tmp_path):
    import os
    import subprocess
    import sys
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from pathlib import Path
    from threading import Thread

    transport = MockTransport()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            response = json.dumps(transport(self.path, body, {})).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    codebook_path, units_path = tmp_path / "book.json", tmp_path / "units.json"
    codebook_path.write_text(json.dumps(book().to_dict()))
    units_path.write_text(json.dumps(units(5)))
    command = [
        sys.executable,
        "-m",
        "commcode.system1",
        "--codebook",
        str(codebook_path),
        "--units",
        str(units_path),
        "--output",
        str(tmp_path / "run"),
        "--prompt",
        "native",
        "--endpoint",
        f"http://127.0.0.1:{server.server_port}/v1/systemone",
    ]
    environment = dict(
        os.environ,
        TYPESAFE_API_KEY="local-test-credential",
        PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
    )
    try:
        first = subprocess.run(command, env=environment, text=True, capture_output=True, check=True)
        assert json.loads(first.stdout)["cells"] == 5
        second = subprocess.run(
            command, env=environment, text=True, capture_output=True, check=True
        )
        assert json.loads(second.stdout)["usage"]["new_dispatches"] == 0
        assert len(transport.calls) == 1
        raw = next((tmp_path / "run" / "journal").glob("*.json")).read_text()
        assert "local-test-credential" not in raw
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_workflow_integrity_and_wrong_teacher_fail_before_dispatch(tmp_path):
    transport = MockTransport()
    provider = Provider(transport=transport)
    with pytest.raises(ValidationError):
        encode_corpus(
            units(2), book(), output_dir=tmp_path, jev_provider=provider, llm_provider=provider
        )
    assert not transport.calls
    encode_corpus(units(2), book(), output_dir=tmp_path, jev_provider=provider)
    path = tmp_path / "workflow.json"
    record = json.loads(path.read_text())
    record["sha256"] = "tampered"
    path.write_text(json.dumps(record))
    with pytest.raises(ValidationError):
        encode_corpus(units(2), book(), output_dir=tmp_path, jev_provider=provider)
    assert len(transport.calls) == 1
