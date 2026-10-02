"""Corpus batching and global budgeted routing over unchanged source evidence."""

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .calibration import Calibrator, select_targets
from .models import Book, EvidencePacket, ValidationError, evidence_only, finite_number
from .parsing import parse_jev_response, parse_llm_response
from .providers import Journal, Provider, atomic_json, canonical, digest
from .requests import build_jev_request, build_llm_request

CORPUS_VERSION = "corpus-1"


def _coerce_calibrator(calibrator, book):
    if isinstance(calibrator, dict):
        calibrator = Calibrator.from_dict(calibrator)
    if calibrator is not None:
        original = {dim.id: dim.to_dict() for dim in book.dimensions}
        fitted = {dim["id"]: dim for dim in calibrator.dimensions}
        if set(fitted) != set(original):
            raise ValidationError("Calibration dimensions differ from the original book")
        for did in original:
            for field in ("kind", "question", "labels", "output_range"):
                if field in fitted[did] and fitted[did][field] != original[did].get(field):
                    raise ValidationError(
                        "Calibration changes an original question or response scale"
                    )
    return calibrator


def _validate_rows(rows, book, expected):
    pairs = [(row["unit_id"], row["dimension_id"]) for row in rows]
    if len(set(pairs)) != len(pairs) or set(pairs) != expected:
        raise ValidationError("Corpus output has missing, extra or duplicate target cells")
    dimensions = {dim.id: dim for dim in book.dimensions}
    for row in rows:
        dim = dimensions[row["dimension_id"]]
        if dim.kind == "choice":
            if row.get("label") not in dim.labels:
                raise ValidationError("Corpus label outside original response space")
        else:
            value = finite_number(row.get("value"), "corpus value")
            lo, hi = (0, 1) if dim.kind == "noul" else dim.output_range
            if not lo <= value <= hi:
                raise ValidationError("Corpus value outside original response scale")


def _builder(provider):
    return build_jev_request if provider.kind == "jev" else build_llm_request


def _parse(provider, record, packet, book, targets, family):
    response = Provider.response(record)
    if provider.kind == "jev":
        return parse_jev_response(response, packet, book, targets, prompt_family=family)
    return parse_llm_response(response, packet, book, targets)


def _partition(
    units, book, provider, family, packet_budget, cell_budget, max_units, llm_provider=None
):
    """Split physical request envelopes, preserving every original target and full context."""
    plans, errors = [], []
    builder = _builder(provider)

    def divide(selected_units, targets):
        packet = EvidencePacket.from_dict(selected_units)
        body = builder(packet, book, targets, family, "A", provider.model)
        teacher_body = (
            build_llm_request(packet, book, targets, family, "A", llm_provider.model)
            if llm_provider is not None
            else None
        )
        teacher_fits = teacher_body is None or len(canonical(teacher_body)) <= packet_budget
        if len(targets) <= cell_budget and len(canonical(body)) <= packet_budget and teacher_fits:
            plans.append({"packet": packet, "targets": targets, "request": body})
            return
        if len(selected_units) > 1:
            middle = len(selected_units) // 2
            for subset in (selected_units[:middle], selected_units[middle:]):
                ids = {u["unit_id"] for u in subset}
                divide(subset, [t for t in targets if t["unit_id"] in ids])
        elif len(targets) > 1:
            middle = len(targets) // 2
            divide(selected_units, targets[:middle])
            divide(selected_units, targets[middle:])
        else:
            errors.append(
                {
                    "status": "review_required",
                    "reason": "single_target_envelope_exceeds_budget",
                    "unit_id": selected_units[0]["unit_id"],
                    "dimension_id": targets[0]["dimension_id"],
                    "request_bytes": len(canonical(body)),
                }
            )

    # Gather identical declared context without sending grouping identifiers to the model.
    # Final exported rows are restored to the user's original input order.
    context_groups = {}
    for unit in units:
        key = digest(evidence_only(unit.get("context", {})))
        context_groups.setdefault(key, []).append(unit)
    ordered_units = [unit for group in context_groups.values() for unit in group]
    for start in range(0, len(ordered_units), max_units):
        subset = ordered_units[start : start + max_units]
        targets = [
            {"unit_id": u["unit_id"], "dimension_id": d.id} for u in subset for d in book.dimensions
        ]
        divide(subset, targets)
    for n, plan in enumerate(plans):
        plan["packet_id"] = f"p{n:06d}"
    return plans, errors


def _usage(records):
    total, fresh = {}, {}
    for record in records:
        try:
            usage = Provider.response(record).get("usage", {})
        except (ValueError, TypeError, KeyError):
            continue
        for key, value in usage.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                total[key] = total.get(key, 0) + value
                if not record["cache_hit"]:
                    fresh[key] = fresh.get(key, 0) + value
    return {
        "all_recorded_tokens": total,
        "new_dispatch_tokens": fresh,
        "record_count": len(records),
        "new_dispatches": sum(not r["cache_hit"] for r in records),
        "cache_hits": sum(r["cache_hit"] for r in records),
        "new_api_seconds_sum": sum(
            r.get("elapsed_seconds", 0) for r in records if not r["cache_hit"]
        ),
        "cost": (
            "Use recorded provider usage and the applicable tariff; no billed cost is invented."
        ),
    }


def encode_corpus(
    units,
    book,
    *,
    output_dir="commcode_run",
    calibrator=None,
    prompt_family="verbose",
    packet_budget=180000,
    cell_budget=256,
    max_units=16,
    workers=4,
    jev_provider=None,
    llm_provider=None,
    policy=None,
    cache_epoch="v1",
):
    """Run full-corpus base coding, optional calibration, then capped LLM replacements.

    ``policy`` accepts ``budget_fraction`` and ``cell_fraction`` (both 0..1),
    and ``route_unfitted_numeric`` (default false). Zero routing budget is the default.
    ``jev_provider`` may also be a configured LLM provider for a direct baseline.
    Inputs contain source units and the original book; this API accepts no references.
    """
    started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    book = Book.from_dict(book) if isinstance(book, dict) else book
    packet = units if isinstance(units, EvidencePacket) else EvidencePacket.from_dict(units)
    packet.validate()
    book.validate()
    for value, name in (
        (packet_budget, "packet_budget"),
        (cell_budget, "cell_budget"),
        (max_units, "max_units"),
        (workers, "workers"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValidationError(f"{name} must be a positive integer")
    if workers > 32:
        raise ValidationError("workers must not exceed 32")
    provider = jev_provider or Provider()
    calibrator = _coerce_calibrator(calibrator, book)
    if (
        provider.kind == "llm"
        and calibrator is not None
        and any(dim.kind == "choice" for dim in book.dimensions)
    ):
        raise ValidationError("Choice calibration requires a probability-emitting base provider")
    if llm_provider is not None and llm_provider.kind != "llm":
        raise ValidationError("The replacement provider must accept LLM requests")
    config = dict(policy or {})
    unknown = set(config) - {"budget_fraction", "cell_fraction", "route_unfitted_numeric"}
    if unknown:
        raise ValidationError("Unsupported corpus routing policy field")
    config.setdefault("budget_fraction", 0.0)
    config.setdefault("cell_fraction", 1.0)
    config.setdefault("route_unfitted_numeric", False)
    if not isinstance(config["route_unfitted_numeric"], bool):
        raise ValidationError("route_unfitted_numeric must be boolean")
    for key in ("budget_fraction", "cell_fraction"):
        value = finite_number(config[key], key)
        if not 0 <= value <= 1:
            raise ValidationError("Routing fractions must lie in 0 to 1")
    if config["budget_fraction"] and llm_provider is None:
        raise ValidationError("A nonzero routing budget requires an explicit LLM provider")
    plans, errors = _partition(
        packet.units,
        book,
        provider,
        prompt_family,
        packet_budget,
        cell_budget,
        max_units,
        llm_provider,
    )
    fingerprint = {
        "version": CORPUS_VERSION,
        "request_plan_sha256": digest([digest(plan["request"]) for plan in plans]),
        "book_sha256": digest(book.to_dict()),
        "units_sha256": digest(packet.to_dict()),
        "calibration_sha256": digest(calibrator.to_dict()) if calibrator else None,
        "prompt_family": prompt_family,
        "packet_budget": packet_budget,
        "cell_budget": cell_budget,
        "max_units": max_units,
        "workers": workers,
        "policy": config,
        "cache_epoch": cache_epoch,
        "base_provider": {
            "kind": provider.kind,
            "endpoint": provider.endpoint,
            "model": provider.model,
        },
        "llm_provider": {"endpoint": llm_provider.endpoint, "model": llm_provider.model}
        if llm_provider
        else None,
    }
    manifest_path = output / "workflow.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if digest(previous["fingerprint"]) != previous["sha256"]:
            raise ValidationError("Workflow manifest integrity mismatch")
        if (
            previous["fingerprint"]["cache_epoch"] == cache_epoch
            and previous["fingerprint"] != fingerprint
        ):
            raise ValidationError(
                "Workflow changed: use a new output directory or explicit cache epoch"
            )
        if previous["fingerprint"] != fingerprint:
            archive = output / "history" / previous["sha256"]
            for filename in (
                "workflow.json",
                "packets.json",
                "base_predictions.json",
                "calibrated_predictions.json",
                "predictions.json",
                "selected_targets.json",
                "result.json",
            ):
                source = output / filename
                if source.exists():
                    atomic_json(archive / filename, json.loads(source.read_text(encoding="utf-8")))
            atomic_json(output / f"workflow.{previous['sha256']}.json", previous)
    manifest = {
        "fingerprint": fingerprint,
        "sha256": digest(fingerprint),
        "data_rights": "Input datasets retain their original rights; the software license "
        "does not license supplied third-party materials.",
    }
    atomic_json(manifest_path, manifest)
    atomic_json(
        output / "packets.json",
        [
            {
                "packet_id": p["packet_id"],
                "targets": p["targets"],
                "request_bytes": len(canonical(p["request"])),
            }
            for p in plans
        ],
    )
    journal = Journal(output / "journal")
    records, base_rows = [], []

    def invoke_plan(plan, stage_provider, body, targets):
        record = None
        signature = stage_provider.request_signature(body, cache_epoch)
        prior = journal.read(signature)
        try:
            record = stage_provider.invoke(body, journal, cache_epoch)
            rows = _parse(stage_provider, record, plan["packet"], book, targets, prompt_family)
            return {"record": record, "predictions": rows}
        except Exception as exc:
            if record is None:
                record = journal.read(signature)
                if record is not None:
                    record = dict(record, cache_hit=prior is not None)
            return {
                "record": record,
                "error": {
                    "stage": stage_provider.kind,
                    "packet_id": plan["packet_id"],
                    "status": "review_required",
                    "error_type": type(exc).__name__,
                },
            }

    with ThreadPoolExecutor(workers) as pool:
        results = list(
            pool.map(
                lambda plan: invoke_plan(plan, provider, plan["request"], plan["targets"]), plans
            )
        )
    for result in results:
        if result.get("record") is not None:
            records.append(result["record"])
        if "error" in result:
            errors.append(result["error"])
        else:
            base_rows.extend(result["predictions"])
    atomic_json(output / "base_predictions.json", base_rows)
    calibrated = calibrator.apply_rows(base_rows) if calibrator else base_rows
    atomic_json(output / "calibrated_predictions.json", calibrated)
    selected = []
    final_rows = calibrated
    if not errors:
        expected = {(u["unit_id"], d.id) for u in packet.units for d in book.dimensions}
        _validate_rows(calibrated, book, expected)
        # A unit may occur in several physical packets after question-envelope splitting.
        # Routing-only aliases let the shared selector cap actual packets, never logical documents.
        row_by_cell = {(r["unit_id"], r["dimension_id"]): r for r in calibrated}
        routed_rows, packet_by_alias, originals = [], {}, {}
        for plan in plans:
            for target in plan["targets"]:
                uid, did = target["unit_id"], target["dimension_id"]
                row = row_by_cell[(uid, did)]
                if (
                    "value" in row
                    and (
                        row.get("risk") is None or row.get("risk_source") == "unfitted_upper_bound"
                    )
                    and not config["route_unfitted_numeric"]
                ):
                    continue
                alias = f"{plan['packet_id']}:{uid}"
                packet_by_alias[alias] = plan["packet_id"]
                originals[alias] = uid
                routed_rows.append(dict(row, unit_id=alias))
        selected_aliases = select_targets(
            routed_rows,
            packet_by_unit=packet_by_alias,
            budget_fraction=config["budget_fraction"],
            cell_fraction=config["cell_fraction"],
        )
        selected = [dict(t, unit_id=originals[t["unit_id"]]) for t in selected_aliases]
        teacher_plans = []
        for plan in plans:
            targets = [
                {"unit_id": t["unit_id"], "dimension_id": t["dimension_id"]}
                for t in selected
                if t["packet_id"] == plan["packet_id"]
            ]
            if targets:
                body = build_llm_request(
                    plan["packet"], book, targets, prompt_family, "A", llm_provider.model
                )
                if len(canonical(body)) > packet_budget:
                    errors.append(
                        {
                            "stage": "llm",
                            "packet_id": plan["packet_id"],
                            "status": "review_required",
                            "reason": "teacher_envelope_exceeds_budget",
                        }
                    )
                else:
                    teacher_plans.append((plan, body, targets))
        with ThreadPoolExecutor(workers) as pool:
            teacher_results = list(
                pool.map(
                    lambda args: invoke_plan(args[0], llm_provider, args[1], args[2]), teacher_plans
                )
            )
        replacements = {}
        for result in teacher_results:
            if result.get("record") is not None:
                records.append(result["record"])
            if "error" in result:
                errors.append(result["error"])
            else:
                for row in result["predictions"]:
                    replacements[(row["unit_id"], row["dimension_id"])] = row
        final_rows = [replacements.get((r["unit_id"], r["dimension_id"]), r) for r in calibrated]
        _validate_rows(final_rows, book, expected)
    unit_order = {u["unit_id"]: n for n, u in enumerate(packet.units)}
    dim_order = {d.id: n for n, d in enumerate(book.dimensions)}
    final_rows = sorted(
        final_rows, key=lambda r: (unit_order[r["unit_id"]], dim_order[r["dimension_id"]])
    )
    result = {
        "status": "review_required" if errors else "completed",
        "predictions": final_rows,
        "selected_targets": selected,
        "errors": errors,
        "workflow": manifest,
        "base_packet_count": len(plans),
        "packet_budget_cap": math.floor(config["budget_fraction"] * len(plans)),
        "usage": _usage(records),
        "wall_seconds": time.perf_counter() - started,
        "stages": records,
        "raw_journal": str(journal.path),
    }
    atomic_json(output / "predictions.json", final_rows)
    atomic_json(output / "selected_targets.json", selected)
    atomic_json(output / "result.json", result)
    return result
