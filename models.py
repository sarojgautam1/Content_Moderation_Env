from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal


# -----------------------------
# OBSERVATION
# -----------------------------
class Observation(BaseModel):
    content: str = Field(..., description="The original content string.")
    report_reason: str = Field(..., description="Reason for reporting content.")

    user_strike_count: Optional[int] = Field(
        None,
        description="User strike count (only visible after requesting history)."
    )

    policy_snippet: str = Field(
        "",
        description="Policy retrieved from handbook."
    )

    fact_check_result: str = Field(
        "",
        description="Fact-check result if requested."
    )

    step_number: int = Field(
        ...,
        description="Current step number in episode."
    )


# -----------------------------
# ACTION
# -----------------------------
class Action(BaseModel):
    action_type: Literal[
        "uphold_report",
        "dismiss_report",
        "issue_warning",
        "escalate_to_human",
        "search_handbook",
        "request_user_history",
        "request_fact_check",
        "do_nothing"
    ]

    argument: Optional[str] = Field(
        None,
        description="Optional argument (query or violation code)."
    )


# -----------------------------
# ENVIRONMENT STATE (IMPORTANT)
# -----------------------------
class EnvState(BaseModel):
    """Internal environment state (hidden from agent)."""

    task_id: str
    user_id: str
    content: str
    report_reason: str

    step_count: int = 0
    max_steps: int = 10
    done: bool = False

    # Retrieved info
    retrieved_policy: str = ""
    retrieved_fact_check: str = ""
    retrieved_user_history: Optional[Dict] = None

    # Flags
    policy_requested: bool = False
    fact_checked: bool = False
    history_requested: bool = False

    def to_observation(self) -> Observation:
        return Observation(
            content=self.content,
            report_reason=self.report_reason,
            user_strike_count=(
                self.retrieved_user_history["strikes"]
                if self.retrieved_user_history
                else None
            ),
            policy_snippet=self.retrieved_policy,
            fact_check_result=self.retrieved_fact_check,
            step_number=self.step_count
        )


# -----------------------------
# TASK MODEL
# -----------------------------
class TaskModel(BaseModel):
    id: str
    difficulty: str
    user_id: str
    content: str
    report_reason: str

    expected_final_action: Literal[
        "uphold_report",
        "dismiss_report",
        "issue_warning",
        "escalate_to_human"
    ]

    expected_violation_code: Optional[str] = None

    required_tools: List[str] = Field(default_factory=list)


# -----------------------------
# TRAJECTORY
# -----------------------------
class TrajectoryStep(BaseModel):
    step_number: int
    observation: Observation
    action: Action
    reward: float = Field(
        0.0,
        description="Reward value (can be negative or positive)"
    )


# -----------------------------
# REWARD
# -----------------------------
class Reward(BaseModel):
    value: float = Field(
        ...,
        description="Reward value (can be negative or positive)"
    )


# -----------------------------
# STEP OUTPUT
# -----------------------------
class StepOutput(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict