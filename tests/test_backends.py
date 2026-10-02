from __future__ import annotations

import json
from types import SimpleNamespace

from commcode.backends import DecisionBackend, DeepSeekBackend, Pricing
from commcode.models import Code, Unit


def test_laya_validates_complete_video_before_deciding(monkeypatch) -> None:
    calls = []

    def fake_post(url, body, key, timeout):
        calls.append((url, body))
        assert len(body["state"]["visual_segments"]) == 5
        if url.endswith("laya-validate"):
            return {"questions": {"a": {"complete": False}}}
        raise AssertionError("Oversize evidence must not be sent for inference")

    monkeypatch.setattr("commcode.backends._post_json", fake_post)
    backend = DecisionBackend(
        "multilingual",
        "http://127.0.0.1:8000/v1/systemone",
        validator_url="http://127.0.0.1:8000/laya-validate",
    )
    unit = Unit.video("v1", visual_segments=[{"segment_id": str(i)} for i in range(5)])
    result = backend.encode(unit, [Code("a", "A researcher definition")])
    assert len(calls) == 1
    assert result.decisions[0].status == "review"
    assert result.decisions[0].label is None


def test_jev_returns_typed_labels_and_batch_cost(monkeypatch) -> None:
    def fake_post(url, body, key, timeout):
        assert url.endswith("/evaluate")
        assert key == "test-key"
        assert body["state"]["text"] == "full comment"
        return {
            "answers": {
                "a": {"choice": "present", "probabilities": {"present": 0.8}},
                "b": {"choice": "absent", "probabilities": {"present": 0.2}},
            },
            "usage": {"inputTokens": 1000, "outputTokens": 100},
        }

    monkeypatch.setattr("commcode.backends._post_json", fake_post)
    backend = DecisionBackend(
        "typesafe-ai/jev",
        "https://ai-gateway.vercel.sh/v1/evaluate",
        api_key="test-key",
        pricing=Pricing(0.1, 0.2),
    )
    result = backend.encode(Unit.text("v1", "full comment"), [Code("a", "A"), Code("b", "B")])
    assert [item.label for item in result.decisions] == ["present", "absent"]
    assert result.api_cost_usd == 0.00012


def test_deepseek_uses_same_complete_evidence_and_rejects_missing_codes() -> None:
    class FakeResponses:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            payload = json.loads(kwargs["input"][0]["content"][0]["text"])
            assert payload["state"]["text"] == "full comment"
            return SimpleNamespace(
                output_text='{"codes":[{"code_id":"a","present":true}]}',
                usage={"input_tokens": 100, "output_tokens": 10},
            )

    fake = FakeResponses()
    backend = DeepSeekBackend(
        api_key="test-key", client=SimpleNamespace(responses=fake), pricing=Pricing(1, 2)
    )
    result = backend.encode(Unit.text("v1", "full comment"), [Code("a", "A")])
    assert result.decisions[0].label == "present"
    assert result.api_cost_usd == 0.00012
    assert len(fake.calls) == 1
