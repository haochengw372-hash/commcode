from __future__ import annotations

import json
from pathlib import Path

import pytest

from commcode import Code, Decision, HumanLabel, Unit, annotate
from commcode.models import BatchResult


class FakeBackend:
    def __init__(self, model: str, *, fail_once: bool = False) -> None:
        self.model = model
        self.fail_once = fail_once
        self.calls = 0

    def encode(self, unit: Unit, codes: list[Code]) -> BatchResult:
        self.calls += 1
        if self.fail_once and self.calls == 1:
            raise RuntimeError("temporary")
        return BatchResult(
            unit.unit_id,
            self.model,
            tuple(Decision(unit.unit_id, code.code_id, self.model, "present", "completed")
                  for code in codes),
            0.1,
            api_cost_usd=0.0,
        )


def test_run_resumes_completed_arms_and_scores_human_reference(tmp_path: Path) -> None:
    good = FakeBackend("laya")
    flaky = FakeBackend("jev", fail_once=True)
    units = [Unit.text("a", "first"), Unit.text("b", "second")]
    codes = [Code("support", "Shows support")]
    human = [HumanLabel("a", "support", "present"), HumanLabel("b", "support", "absent")]
    annotate(units, codes, [good, flaky], output_dir=tmp_path, reference=human)
    assert good.calls == 2
    assert flaky.calls == 2
    # Only the failed first unit is retried; every completed arm is reused.
    records = annotate(units, codes, [good, flaky], output_dir=tmp_path, reference=human)
    assert good.calls == 2
    assert flaky.calls == 3
    assert len(records) == 4
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["models"]["jev"]["completed"] == 2
    assert summary["human_reference"]["laya:support"]["accuracy"] == 0.5
    assert len((tmp_path / "records.jsonl").read_text().splitlines()) == 4
    assert len((tmp_path / "attempts.jsonl").read_text().splitlines()) == 5
    with pytest.raises(ValueError, match="different units"):
        annotate(units, [Code("support", "Changed definition")], [good, flaky], output_dir=tmp_path)
