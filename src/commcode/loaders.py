"""Explicit CSV mappings and a structured-video-evidence adapter."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from .models import Code, HumanLabel, Unit


def load_codebook_csv(
    path: str | Path,
    *,
    id_column: str = "code_id",
    definition_column: str = "definition",
    name_column: str = "code_name",
) -> list[Code]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {id_column, definition_column}
        if not required.issubset(reader.fieldnames or []):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"Codebook lacks columns: {missing}")
        codes = [
            Code(
                code_id=(row.get(id_column) or "").strip(),
                definition=(row.get(definition_column) or "").strip(),
                name=(row.get(name_column) or "").strip(),
                include=(row.get("inclusion_criteria") or "").strip(),
                exclude=(row.get("exclusion_criteria") or "").strip(),
            )
            for row in reader
        ]
    ids = [code.code_id for code in codes]
    if len(ids) != len(set(ids)):
        raise ValueError("Codebook has duplicate code IDs")
    return codes


def load_text_csv(
    path: str | Path,
    *,
    id_column: str,
    text_column: str,
    limit: int | None = None,
) -> list[Unit]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    result: list[Unit] = []
    seen: set[str] = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {id_column, text_column}
        if not required.issubset(reader.fieldnames or []):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"Corpus lacks columns: {missing}")
        for row in reader:
            unit_id = (row.get(id_column) or "").strip()
            content = (row.get(text_column) or "").strip()
            if not unit_id or not content:
                continue
            if unit_id in seen:
                raise ValueError(f"Duplicate unit ID: {unit_id}")
            seen.add(unit_id)
            result.append(Unit.text(unit_id, content, source_id=str(path)))
            if len(result) == limit:
                break
    return result


def load_human_labels_csv(
    path: str | Path,
    *,
    id_column: str,
    code_columns: Mapping[str, str],
    positive_values: Iterable[str] = ("1", "1.0", "true", "yes", "present"),
    negative_values: Iterable[str] = ("0", "0.0", "false", "no", "absent"),
    annotator_column: str = "",
) -> list[HumanLabel]:
    """Import wide human labels with an explicit column-to-code mapping."""
    positive = {value.casefold() for value in positive_values}
    negative = {value.casefold() for value in negative_values}
    if positive & negative:
        raise ValueError("Positive and negative value sets overlap")
    result: list[HumanLabel] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {id_column, *code_columns}
        if annotator_column:
            required.add(annotator_column)
        if not required.issubset(reader.fieldnames or []):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"Labels lack columns: {missing}")
        for row in reader:
            unit_id = (row.get(id_column) or "").strip()
            if not unit_id:
                continue
            annotator = (row.get(annotator_column) or "").strip() if annotator_column else ""
            for source_column, code_id in code_columns.items():
                value = (row.get(source_column) or "").strip().casefold()
                if not value:
                    continue
                if value in positive:
                    label = "present"
                elif value in negative:
                    label = "absent"
                else:
                    raise ValueError(f"Unexpected label {value!r} in {source_column}")
                result.append(HumanLabel(unit_id, code_id, label, annotator))
    return result


def load_structured_video_evidence(path: str | Path) -> Unit:
    """Load full video evidence emitted by the existing standalone CLI."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    transcript = payload.get("transcript") or {}
    transcript_text = transcript.get("text") or " ".join(
        str(segment.get("text") or "") for segment in transcript.get("segments") or []
    )
    visual = [
        {
            "segment_id": segment.get("segment_id"),
            "start_seconds": segment.get("start_seconds"),
            "end_seconds": segment.get("end_seconds"),
            "screen_text": segment.get("screen_text") or [],
            "visual_observations": segment.get("visual_observations") or [],
        }
        for segment in payload.get("segments") or []
    ]
    return Unit.video(
        str(payload["video_id"]),
        caption=str((payload.get("post") or {}).get("caption") or ""),
        transcript=transcript_text,
        visual_segments=visual,
        source_id=str(source),
        warnings=tuple(payload.get("warnings") or ()),
    )
