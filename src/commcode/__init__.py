"""CommCode: reproducible content annotation across text and video evidence."""

from .codebook_protocol import build_jev_request, validate_decision_card
from .engine import annotate
from .loaders import (
    load_codebook_csv,
    load_human_labels_csv,
    load_structured_video_evidence,
    load_text_csv,
)
from .metrics import score_decisions
from .models import BatchResult, Code, Decision, HumanLabel, Unit

__all__ = [
    "Code",
    "BatchResult",
    "build_jev_request",
    "Decision",
    "HumanLabel",
    "Unit",
    "annotate",
    "load_codebook_csv",
    "load_human_labels_csv",
    "load_structured_video_evidence",
    "load_text_csv",
    "score_decisions",
    "validate_decision_card",
]
