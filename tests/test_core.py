from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from commcode import (
    Code,
    Decision,
    HumanLabel,
    Unit,
    load_codebook_csv,
    load_human_labels_csv,
    load_structured_video_evidence,
    load_text_csv,
    score_decisions,
)


def _csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_text_codebook_and_human_reference(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.csv"
    book = tmp_path / "book.csv"
    _csv(corpus, ["id", "comment", "human"], [
        {"id": "1", "comment": "Helpful context", "human": "1"},
        {"id": "2", "comment": "", "human": "0"},
        {"id": "3", "comment": "An insult", "human": "0"},
    ])
    _csv(book, ["code_id", "definition"], [
        {"code_id": "constructive", "definition": "Contributes to discussion"}
    ])
    units = load_text_csv(corpus, id_column="id", text_column="comment")
    labels = load_human_labels_csv(corpus, id_column="id", code_columns={"human": "constructive"})
    assert [unit.unit_id for unit in units] == ["1", "3"]
    assert len(load_codebook_csv(book)) == 1
    assert len(labels) == 3
    decisions = [
        Decision("1", "constructive", "test", "present", "completed"),
        Decision("3", "constructive", "test", None, "review"),
    ]
    score = score_decisions(decisions, labels)["test:constructive"]
    assert score["accuracy"] == 1
    assert score["coverage"] == 0.5
    assert score["review"] == 1
    with pytest.raises(ValueError, match="Multiple human labels"):
        score_decisions(decisions, labels + [HumanLabel("1", "constructive", "absent", "b")])


def test_video_loader_preserves_all_supplied_segments(tmp_path: Path) -> None:
    path = tmp_path / "video.json"
    path.write_text(json.dumps({
        "video_id": "v1",
        "post": {"caption": "caption"},
        "transcript": {"text": "all spoken words"},
        "segments": [
            {"segment_id": f"s{i}", "screen_text": [f"frame {i}"],
             "visual_observations": ["observed"]}
            for i in range(7)
        ],
    }))
    unit = load_structured_video_evidence(path)
    assert unit.modality == "video"
    assert len(unit.state()["visual_segments"]) == 7
    assert unit.state()["transcript"] == "all spoken words"
    assert Unit.text("t", "text").state() == {"unit_type": "text", "text": "text"}
    with pytest.raises(ValueError, match="definition"):
        Code("missing", "")
