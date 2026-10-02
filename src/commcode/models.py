"""Platform-neutral coding units and binary research codebook entries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Label = Literal["present", "absent"]
Modality = Literal["text", "video"]


@dataclass(frozen=True)
class Code:
    code_id: str
    definition: str
    name: str = ""
    include: str = ""
    exclude: str = ""

    def __post_init__(self) -> None:
        if not self.code_id.strip() or not self.definition.strip():
            raise ValueError("Every code needs an ID and a researcher-authored definition")


@dataclass(frozen=True)
class Unit:
    unit_id: str
    modality: Modality
    evidence: dict[str, Any]
    source_id: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.unit_id.strip():
            raise ValueError("unit_id is required")
        if self.modality not in ("text", "video"):
            raise ValueError("modality must be text or video")
        if self.modality == "text" and not str(self.evidence.get("text") or "").strip():
            raise ValueError("Text units need nonempty text evidence")
        if self.modality == "video" and not any(
            self.evidence.get(key) for key in ("caption", "transcript", "visual_segments")
        ):
            raise ValueError("Video units need caption, transcript, or visual segments")

    @classmethod
    def text(cls, unit_id: str, content: str, *, source_id: str = "") -> Unit:
        return cls(unit_id, "text", {"text": content}, source_id)

    @classmethod
    def video(
        cls,
        unit_id: str,
        *,
        caption: str = "",
        transcript: str = "",
        visual_segments: list[dict[str, Any]] | None = None,
        source_id: str = "",
        warnings: tuple[str, ...] = (),
    ) -> Unit:
        return cls(
            unit_id,
            "video",
            {
                "caption": caption,
                "transcript": transcript,
                "visual_segments": visual_segments or [],
            },
            source_id,
            warnings,
        )

    def state(self) -> dict[str, Any]:
        """Return all supplied evidence, without clipping or sampling."""
        return {"unit_type": self.modality, **self.evidence}


@dataclass(frozen=True)
class HumanLabel:
    unit_id: str
    code_id: str
    label: Label
    annotator_id: str = ""


@dataclass(frozen=True)
class Decision:
    unit_id: str
    code_id: str
    model: str
    label: Label | None
    status: Literal["completed", "review", "error"]
    p_present: float | None = None
    error: str = ""


@dataclass(frozen=True)
class BatchResult:
    unit_id: str
    model: str
    decisions: tuple[Decision, ...]
    elapsed_seconds: float
    api_cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str = ""
