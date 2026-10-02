"""Synthetic transport demo: no network, credentials, or research data."""

import json
import tempfile
from pathlib import Path

from commcode.system1 import Book, EvidencePacket, Provider, System1Encoder

HERE = Path(__file__).resolve().parent
book = Book.from_dict(json.loads((HERE / "book.json").read_text()))
packet = EvidencePacket.from_dict(json.loads((HERE / "units.json").read_text()))


def synthetic_transport(endpoint, body, headers):
    del endpoint, headers
    return {
        "model": "synthetic-offline-fixture",
        "answers": {
            key: {
                "type": "choice",
                "choice": "suggestion",
                "probabilities": {"suggestion": 0.75, "other": 0.25},
            }
            for key in body["questions"]
        },
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


with tempfile.TemporaryDirectory() as output:
    provider = Provider(transport=synthetic_transport)
    encoder = System1Encoder(provider)
    result = encoder.encode(packet, book, output_dir=output)
    resumed = encoder.encode(packet, book, output_dir=output)
    assert result["status"] == "completed"
    assert len(result["predictions"]) == 2
    assert resumed["record"]["cache_hit"]
    print("Synthetic offline demo passed: two outputs and a resumed journal hit.")
