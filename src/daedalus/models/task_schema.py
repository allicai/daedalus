from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

VariantType = Literal["root_cause_attribution"]

TransformationScope = Literal["problem_statement"]

TargetFailureMode = Literal["wrong_root_cause"]

QualityStatus = Literal["unverified", "validated", "rejected"]


class TransformationMetadata(BaseModel):
    description: str
    changes_made: list[str]
    intended_distractor: str


class Provenance(BaseModel):
    dataset: str
    license: str
    source_url: str


class DaedalusTask(BaseModel):
    source_instance_id: str
    variant_id: str
    variant_type: VariantType
    repo: str
    base_commit: str
    original_problem_statement: str
    modified_problem_statement: str
    transformation_scope: list[TransformationScope]
    transformation_metadata: TransformationMetadata
    target_failure_mode: TargetFailureMode
    quality_status: QualityStatus = "unverified"
    validation_notes: str = ""
    provenance: Provenance
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = "1.0"

    @staticmethod
    def make_variant_id(source_instance_id: str, variant_type: str, index: int) -> str:
        return f"{source_instance_id}__{variant_type}__{index:03d}"
