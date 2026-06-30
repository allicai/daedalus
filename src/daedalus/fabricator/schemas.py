from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

ArtifactClass = Literal["A", "B"]

SafetyResult = Literal[
    "passed",            # both test passes: no existing tests broke, gold patch still fixes
    "failed_tests",      # fabrication broke a previously-passing test
    "failed_gold_patch", # gold patch no longer resolves the issue when fabrication is applied
    "patch_error",       # diff could not be applied to the repo
    "unavailable",       # Docker not running or image not pullable
    "error",             # unexpected error during check
]


class DistractorInfo(BaseModel):
    symbol: str
    file: str
    role: str
    ownership: str


class FabricatedArtifact(BaseModel):
    artifact_id: str
    variant_id: str
    source_instance_id: str
    base_commit: str
    repo: str

    distractor: DistractorInfo

    target_file: str
    artifact_class: ArtifactClass
    original_text: str   # exact text replaced in the file
    modified_text: str   # replacement text
    diff: str            # unified diff (generated from original/modified)
    reasoning: str       # fabricator's stated justification

    safety_result: SafetyResult = "unavailable"
    safety_notes: str = ""

    review_decision: Literal["approved", "rejected", "pending"] = "pending"
    review_notes: str = ""

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = "1.0"


class RejectedArtifact(BaseModel):
    """Auto-rejected fabrication candidate — recorded for Class C violation rate tracking."""
    rejection_id: str
    variant_id: str
    source_instance_id: str
    base_commit: str
    repo: str

    target_file: str
    proposed_class: str       # what the LLM proposed (A or B)
    rejection_stage: Literal["class_c", "gold_patch_overlap", "text_not_found", "too_short", "no_change"]
    rejection_reason: str     # human-readable explanation of what triggered rejection

    original_text: str
    modified_text: str

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = "1.0"


class FabricationReview(BaseModel):
    artifact_id: str
    variant_id: str
    source_instance_id: str
    decision: Literal["approved", "rejected"]
    notes: str = ""
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
