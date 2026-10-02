"""Synthetic transport: typed parsing, 1,000-unit coverage, and free cache replay."""

import json
import tempfile
import threading
from pathlib import Path

from commcode.system1 import Book, Provider, encode_corpus

HERE = Path(__file__).resolve().parent
book = Book.from_dict(json.loads((HERE / "book.json").read_text()))
calls = []
lock = threading.Lock()


def synthetic_transport(endpoint, body, headers):
    del endpoint, headers
    with lock:
        calls.append(1)
    answers = {}
    for key, question in body["questions"].items():
        kind = question["type"]
        if kind == "choice":
            labels = list(question["criteria"])
            answers[key] = {
                "type": kind,
                "choice": labels[0],
                "probabilities": {label: float(i == 0) for i, label in enumerate(labels)},
            }
        elif kind == "noul":
            answers[key] = {"type": kind, "noul": 0.75}
        else:
            criteria = question["criteria"]
            answers[key] = {
                "type": kind,
                "score": 1.0,
                "legend": {str(i): label for i, label in enumerate(criteria)},
                "probabilities": {str(i): float(i == 1) for i in range(len(criteria))},
            }
    return {
        "model": "synthetic-offline-fixture",
        "answers": answers,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


with tempfile.TemporaryDirectory() as output:
    provider = Provider(transport=synthetic_transport)
    small = json.loads((HERE / "units.json").read_text())
    direct = encode_corpus(
        small,
        book,
        output_dir=Path(output) / "direct",
        jev_provider=provider,
        prompt_family="verbose",
    )
    assert direct["status"] == "completed"
    units = [
        {"unit_id": f"synthetic-{i}", "text": "Please add a search box.", "context": {}}
        for i in range(1000)
    ]
    kwargs = {
        "output_dir": Path(output) / "corpus",
        "jev_provider": provider,
        "prompt_family": "native",
        "workers": 4,
        "max_units": 16,
        "policy": {"budget_fraction": 0.0},
    }
    result = encode_corpus(units, book, **kwargs)
    before_replay = len(calls)
    resumed = encode_corpus(units, book, **kwargs)
    assert result["status"] == resumed["status"] == "completed"
    assert len(result["predictions"]) == 3000
    assert len({row["unit_id"] for row in result["predictions"]}) == 1000
    assert len(calls) == before_replay
    assert result["selected_targets"] == []
    assert all(
        row["label"] == "Yes"
        for row in result["predictions"]
        if row["dimension_id"] == "suggestion"
    )
    assert all(
        0 <= row["value"] <= 1
        for row in result["predictions"]
        if row["dimension_id"] == "helpfulness"
    )
    print("Synthetic offline demo passed: 1,000 units, 3,000 cells, zero-call replay.")
