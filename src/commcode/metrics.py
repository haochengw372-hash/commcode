"""Human-reference comparisons with explicit missing and review denominators."""

from __future__ import annotations

from collections import defaultdict

from .models import Decision, HumanLabel


def score_decisions(
    decisions: list[Decision], reference: list[HumanLabel]
) -> dict[str, dict[str, float | int | None]]:
    """Score one model run against one chosen human label per unit and code."""
    human: dict[tuple[str, str], str] = {}
    for item in reference:
        key = (item.unit_id, item.code_id)
        if key in human:
            raise ValueError(
                f"Multiple human labels for {key}; choose or aggregate annotators explicitly"
            )
        human[key] = item.label
    seen: set[tuple[str, str, str]] = set()
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "review": 0, "error": 0}
    )
    for item in decisions:
        key = (item.model, item.unit_id, item.code_id)
        if key in seen:
            raise ValueError(f"Duplicate model decision: {key}")
        seen.add(key)
        truth = human.get((item.unit_id, item.code_id))
        if truth is None:
            continue
        row = counts[(item.model, item.code_id)]
        if item.status == "review":
            row["review"] += 1
        elif item.status == "error" or item.label is None:
            row["error"] += 1
        elif item.label == "present":
            row["tp" if truth == "present" else "fp"] += 1
        else:
            row["tn" if truth == "absent" else "fn"] += 1
    results: dict[str, dict[str, float | int | None]] = {}
    for (model, code_id), row in sorted(counts.items()):
        tp, fp, tn, fn = (row[key] for key in ("tp", "fp", "tn", "fn"))
        evaluated = tp + fp + tn + fn
        total = evaluated + row["review"] + row["error"]
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        results[f"{model}:{code_id}"] = {
            **row,
            "reference_n": total,
            "evaluated_n": evaluated,
            "coverage": evaluated / total if total else None,
            "accuracy": (tp + tn) / evaluated if evaluated else None,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "positive_rate": (tp + fp) / evaluated if evaluated else None,
            "reference_positive_rate": (tp + fn) / evaluated if evaluated else None,
        }
    return results
