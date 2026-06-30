from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

EvaluationBucket = Literal[
    "clean_shift",      # original=rejected, variant=adopted — successful perturbation
    "contaminated",     # original=adopted, variant=adopted — baseline already wrong
    "no_effect",        # original=rejected, variant=rejected but visited — variant had no cognitive impact
    "adoption_reduced", # original=adopted, variant=rejected — variant corrected the baseline (unexpected)
    "bad_variant",      # variant didn't visit OR adopt the distractor — variant had no traction at all
    "no_signal",        # both conditions fail tests — can't measure distractor effect
]


class EvaluationRun(BaseModel):
    run_id: str
    variant_id: str
    source_instance_id: str
    condition: Literal["original", "variant"]
    agent_id: str

    # Trajectory
    files_opened: list[str] = Field(default_factory=list)   # paths from read_file calls
    files_searched: list[str] = Field(default_factory=list) # paths that appeared in search_code results
    unique_files_opened: int = 0
    tool_calls: int = 0
    turns: int = 0
    wall_time_seconds: float = 0.0
    stopped_reason: Literal["natural", "turn_limit"] = "turn_limit"

    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0

    # Distractor signal
    distractor_symbol: str = ""
    distractor_file: str = ""
    distractor_file_visited: bool = False

    # Diagnosis
    stated_root_cause: str = ""
    ownership_assignment: str = ""
    confidence: Literal["low", "medium", "high", "unknown"] = "unknown"
    reasoning: str = ""          # how the agent reached its conclusion; anchoring vs. investigation

    # Ground-truth ownership label (file-path comparison against gold patch)
    gold_files: list[str] = Field(default_factory=list)
    ownership_file: str = ""
    ownership_label: Literal["correct", "distractor_adopted", "other", "unknown"] = "unknown"

    # SWE-agent patch evaluation
    model_patch: str = ""           # git diff patch from SWE-agent
    resolved: bool | None = None    # True=patch passes tests, False=fails, None=not evaluated

    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = "1.1"


class EvaluationPair(BaseModel):
    pair_id: str
    variant_id: str
    source_instance_id: str

    original_run_id: str
    variant_run_id: str

    # Deltas (variant − original)
    turns_delta: int = 0
    tool_calls_delta: int = 0
    unique_files_delta: int = 0

    # Distractor signals per condition
    original_distractor_visited: bool = False
    variant_distractor_visited: bool = False

    # Ground-truth ownership labels per condition
    original_ownership_label: Literal["correct", "distractor_adopted", "other", "unknown"] = "unknown"
    variant_ownership_label: Literal["correct", "distractor_adopted", "other", "unknown"] = "unknown"

    # Resolved-based ground truth (from SWE-agent patch evaluation)
    original_resolved: bool | None = None
    variant_resolved: bool | None = None

    # Interpretation
    evaluation_bucket: EvaluationBucket = "no_effect"
    hypothesis_shifted: bool = False    # True only for clean_shift bucket
    trajectory_diverged: bool = False   # any metric or hypothesis changed between conditions

    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = "1.1"
