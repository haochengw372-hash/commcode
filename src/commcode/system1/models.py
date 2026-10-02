"""Dataset-neutral, reference-free research inputs."""

import math
from dataclasses import asdict, dataclass, field
from typing import Any


class ValidationError(ValueError):
    """Missing or invalid evidence requires review, never a negative label."""


def finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValidationError(f"{name} must be finite")
    return value


@dataclass
class Dimension:
    id: str
    question: str
    kind: str
    labels: list[str] = field(default_factory=list)
    criteria: Any = None
    estimand: str = "category"
    output_range: list[float] | None = None
    source_spans: Any = None
    direction: Any = None
    original_answer_mapping: dict | None = None
    positive_original_response: str | None = None
    conditional_on: Any = None

    @classmethod
    def from_dict(cls, value: dict) -> "Dimension":
        return cls(**{k: value[k] for k in cls.__dataclass_fields__ if k in value})

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if not isinstance(self.id, str) or not self.id or not self.question.strip():
            raise ValidationError("Dimension needs an id and original question")
        if self.kind not in ("choice", "noul", "score"):
            raise ValidationError("Unsupported primitive")
        if self.estimand not in ("category", "rater_fraction", "continuous_rating"):
            raise ValidationError("Unsupported estimand")
        if self.kind == "choice":
            if (
                len(self.labels) < 2
                or any(not isinstance(x, str) for x in self.labels)
                or len(set(self.labels)) != len(self.labels)
            ):
                raise ValidationError("Choice requires distinct string labels")
            if self.criteria is not None and (
                not isinstance(self.criteria, dict) or set(self.criteria) != set(self.labels)
            ):
                raise ValidationError("Choice criteria must cover exactly the original labels")
        if self.original_answer_mapping is not None:
            if (
                self.kind != "choice"
                or not isinstance(self.original_answer_mapping, dict)
                or set(self.original_answer_mapping) != set(self.labels)
            ):
                raise ValidationError("Original answer mapping must cover all choice labels")
        if self.kind == "noul" and self.criteria is not None:
            if not isinstance(self.criteria, dict) or set(self.criteria) != {"true", "false"}:
                raise ValidationError("Noul criteria must contain true and false")
        if self.kind == "score":
            if not isinstance(self.criteria, list) or not 2 <= len(self.criteria) <= 10:
                raise ValidationError("Score requires 2 to 10 ordered anchors")
            if not self.output_range or len(self.output_range) != 2:
                raise ValidationError("Score requires its original output_range")
            lo, hi = [finite_number(x, "output_range") for x in self.output_range]
            if hi <= lo:
                raise ValidationError("output_range must be increasing")


@dataclass
class Book:
    book_id: str
    original_text: str
    dimensions: list[Dimension]
    provenance: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict) -> "Book":
        result = cls(
            value["book_id"],
            value["original_text"],
            [Dimension.from_dict(x) for x in value["dimensions"]],
            value.get("provenance", {}),
        )
        result.validate()
        return result

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if not isinstance(self.original_text, str) or not self.original_text.strip():
            raise ValidationError("Original codebook text required")
        if not self.dimensions or len({x.id for x in self.dimensions}) != len(self.dimensions):
            raise ValidationError("Dimensions must be nonempty and unique")
        for dim in self.dimensions:
            dim.validate()
            if dim.criteria is not None:
                fragments = (
                    dim.criteria.values() if isinstance(dim.criteria, dict) else dim.criteria
                )
                normalized = " ".join(self.original_text.split())
                for fragment in fragments:
                    if isinstance(fragment, str) and " ".join(fragment.split()) not in normalized:
                        raise ValidationError("Criterion text absent from original codebook")
            validate_spans(dim.source_spans, self.original_text)
            if isinstance(dim.source_spans, list):
                for span in dim.source_spans:
                    if isinstance(span, dict) and {"field", "start", "end"} <= span.keys():
                        value = dim.to_dict()
                        for part in span["field"].split("."):
                            if (
                                isinstance(value, list)
                                and part.isdigit()
                                and int(part) < len(value)
                            ):
                                value = value[int(part)]
                            elif isinstance(value, dict) and part in value:
                                value = value[part]
                            else:
                                raise ValidationError("Source span field absent from dimension")
                        if value != self.original_text[span["start"] : span["end"]]:
                            raise ValidationError(
                                "Declared field differs from original source span"
                            )


def validate_spans(spans: Any, original: str) -> None:
    if spans is None:
        return
    if isinstance(spans, str):
        if spans not in original:
            raise ValidationError("Declared source excerpt absent from original codebook")
    elif isinstance(spans, list):
        for span in spans:
            validate_spans(span, original)
    elif isinstance(spans, dict):
        if "text" in spans or "excerpt" in spans:
            validate_spans(spans.get("text", spans.get("excerpt")), original)
        elif "start" in spans and "end" in spans:
            start, end = spans["start"], spans["end"]
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or not 0 <= start < end <= len(original)
            ):
                raise ValidationError("Source span outside original codebook")
        else:
            for span in spans.values():
                validate_spans(span, original)
    else:
        raise ValidationError("Invalid source spans")


# Metadata stays in local exports, never in provider state. Context is actual evidence only.
PRIVATE_KEYS = {
    "unit_id",
    "group_id",
    "speaker_id",
    "speaker",
    "rater",
    "rater_id",
    "raters",
    "gold",
    "label",
    "labels",
    "reference",
    "references",
    "human_label",
    "human_labels",
    "split",
    "dataset",
    "dataset_name",
    "source_locator",
    "provenance",
    "api_key",
    "token",
    "authorization",
    "password",
}


def evidence_only(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: evidence_only(v)
            for k, v in value.items()
            if str(k).lower() not in PRIVATE_KEYS
            and not str(k).lower().startswith(("gold_", "rater_", "human_label", "reference_"))
        }
    if isinstance(value, list):
        return [evidence_only(x) for x in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValidationError("Evidence contains a nonfinite number")
    return value


@dataclass
class EvidencePacket:
    units: list[dict]

    @classmethod
    def from_dict(cls, value: dict | list) -> "EvidencePacket":
        result = cls(value if isinstance(value, list) else value["units"])
        result.validate()
        return result

    def validate(self) -> None:
        ids = []
        for unit in self.units:
            uid = unit.get("unit_id")
            if not isinstance(uid, str) or not uid:
                raise ValidationError("Each unit needs a string unit_id")
            if not isinstance(unit.get("text"), str) or not unit["text"].strip():
                raise ValidationError("Missing source text requires review")
            if not isinstance(unit.get("context", {}), dict):
                raise ValidationError("Context must be explicitly declared evidence")
            ids.append(uid)
        if not ids or len(ids) != len(set(ids)):
            raise ValidationError("Units must be nonempty and unique")

    def to_dict(self) -> dict:
        return {"units": self.units}

    def state_units(self) -> list[dict]:
        return [
            {"n": n, "text": u["text"], "context": evidence_only(u.get("context", {}))}
            for n, u in enumerate(self.units)
        ]
