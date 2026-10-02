"""Run independent model arms with resumable, auditable outputs."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from .metrics import score_decisions
from .models import BatchResult, Code, Decision, HumanLabel, Unit


class Backend(Protocol):
    model: str

    def encode(self, unit: Unit, codes: list[Code]) -> BatchResult: ...


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _save_records(path: Path, records: dict[tuple[str, str], BatchResult]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(_canonical(asdict(result)) + "\n" for result in records.values()),
        encoding="utf-8",
    )
    temporary.replace(path)


def _from_record(payload: dict) -> BatchResult:
    return BatchResult(
        unit_id=payload["unit_id"],
        model=payload["model"],
        decisions=tuple(Decision(**item) for item in payload["decisions"]),
        elapsed_seconds=payload["elapsed_seconds"],
        api_cost_usd=payload.get("api_cost_usd"),
        input_tokens=payload.get("input_tokens"),
        output_tokens=payload.get("output_tokens"),
        error=payload.get("error", ""),
    )


def annotate(
    units: Sequence[Unit],
    codes: Sequence[Code],
    backends: Sequence[Backend],
    *,
    output_dir: str | Path,
    reference: Sequence[HumanLabel] = (),
) -> list[BatchResult]:
    """Code each unit independently; resume completed/review arms and retry errors."""
    if not units or not codes or not backends:
        raise ValueError("Units, codes, and backends must all be nonempty")
    ids = [unit.unit_id for unit in units]
    models = [backend.model for backend in backends]
    code_ids = [code.code_id for code in codes]
    if len(ids) != len(set(ids)) or len(models) != len(set(models)):
        raise ValueError("Unit IDs and backend model names must be unique")
    if len(code_ids) != len(set(code_ids)):
        raise ValueError("Code IDs must be unique")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    unit_hashes = {unit.unit_id: _sha(unit.state()) for unit in units}
    fingerprint = _sha(
        {
            "units": unit_hashes,
            "codes": [asdict(code) for code in codes],
            "models": models,
        }
    )
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("Existing run has different units, codebook, or models")
    else:
        manifest = {
            "schema_version": "commcode_run_v1",
            "created_at": datetime.now(UTC).isoformat(),
            "fingerprint": fingerprint,
            "unit_hashes": unit_hashes,
            "code_ids": code_ids,
            "models": models,
            "evidence_policy": "all supplied evidence; no automatic clipping",
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    records_path = destination / "records.jsonl"
    records: dict[tuple[str, str], BatchResult] = {}
    if records_path.exists():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            result = _from_record(json.loads(line))
            records[(result.unit_id, result.model)] = result
    wall_times: dict[str, float] = {}
    for unit in units:
        pending = [
            backend
            for backend in backends
            if (unit.unit_id, backend.model) not in records
            or records[(unit.unit_id, backend.model)].error
        ]
        if not pending:
            continue
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(pending)) as pool:
            futures = {
                pool.submit(backend.encode, unit, list(codes)): backend for backend in pending
            }
            for future in as_completed(futures):
                backend = futures[future]
                try:
                    result = future.result()
                    if result.unit_id != unit.unit_id or result.model != backend.model:
                        raise ValueError("Backend returned a different unit or model")
                    if {item.code_id for item in result.decisions} != set(code_ids):
                        raise ValueError("Backend returned missing or duplicate code IDs")
                except Exception as error:
                    result = BatchResult(
                        unit_id=unit.unit_id,
                        model=backend.model,
                        decisions=tuple(
                            Decision(unit.unit_id, code.code_id, backend.model, None, "error")
                            for code in codes
                        ),
                        elapsed_seconds=time.perf_counter() - start,
                        error=type(error).__name__,
                    )
                records[(unit.unit_id, backend.model)] = result
                with (destination / "attempts.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(_canonical(asdict(result)) + "\n")
                _save_records(records_path, records)
        wall_times[unit.unit_id] = time.perf_counter() - start
    summaries = {}
    for model in models:
        values = [result for result in records.values() if result.model == model]
        durations = [value.elapsed_seconds for value in values if not value.error]
        costs = [value.api_cost_usd for value in values]
        summaries[model] = {
            "calls": len(values),
            "completed": sum(
                all(item.status == "completed" for item in value.decisions) for value in values
            ),
            "review": sum(
                any(item.status == "review" for item in value.decisions) for value in values
            ),
            "error": sum(bool(value.error) for value in values),
            "median_seconds": statistics.median(durations) if durations else None,
            "total_seconds": sum(durations),
            "known_api_cost_usd": sum(cost for cost in costs if cost is not None),
            "cost_coverage": sum(cost is not None for cost in costs),
        }
    all_decisions = [decision for result in records.values() for decision in result.decisions]
    (destination / "summary.json").write_text(
        json.dumps(
            {
                "models": summaries,
                "wall_seconds_by_unit_this_invocation": wall_times,
                "human_reference": (
                    score_decisions(all_decisions, list(reference)) if reference else {}
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return list(records.values())
