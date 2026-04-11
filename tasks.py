from models import TaskModel
from typing import List

TASKS = [
    TaskModel(
        id="task_easy_001",
        difficulty="Easy",
        user_id="user_clean_001",
        content="People from that specific religion are all dirty criminals and should be thrown out of our city. We don't want them here.",
        report_reason="Hate Speech / Slurs",
        expected_final_action="uphold_report",
        expected_violation_code="hate_speech",
        required_tools=["search_handbook"],
        grader="hate_speech_grader"
    ),
    TaskModel(
        id="task_medium_001",
        difficulty="Medium",
        user_id="user_warned_002",
        content="Don't trust the global doctors! Drinking pure lemon juice cures all viruses immediately. Big pharma is hiding this miraculous truth.",
        report_reason="Medical Misinformation",
        expected_final_action="uphold_report",
        expected_violation_code="health_misinfo",
        required_tools=["search_handbook", "request_fact_check"],
        grader="misinfo_grader"
    ),
    TaskModel(
        id="task_medium_002",
        difficulty="Medium",
        user_id="user_clean_001",
        content="I heard alien ships are hiding behind the moon. Just an opinion.",
        report_reason="Misinformation",
        expected_final_action="dismiss_report",
        expected_violation_code=None,
        required_tools=["search_handbook", "request_fact_check"],
        grader="misinfo_grader"
    ),
    TaskModel(
        id="task_hard_001",
        difficulty="Hard",
        user_id="user_repeat_003",
        content="I simply LOVE how our glorious Mayor is stealing our taxes to fund his personal yacht. Excellent leadership! (Satire account)",
        report_reason="Harassment / Bullying of a Public Figure",
        expected_final_action="escalate_to_human",
        expected_violation_code=None,
        required_tools=["search_handbook", "request_user_history"],
        grader="satire_escalation_grader"
    ),
    TaskModel(
        id="task_hard_002",
        difficulty="Hard",
        user_id="user_clean_001",
        content="I simply LOVE how our glorious Mayor is stealing our taxes to fund his personal yacht. Excellent leadership! (Satire account)",
        report_reason="Harassment / Bullying of a Public Figure",
        expected_final_action="dismiss_report",
        expected_violation_code=None,
        required_tools=["search_handbook", "request_user_history"],
        grader="satire_dismiss_grader"
    ),
    TaskModel(
        id="task_hard_003",
        difficulty="Hard",
        user_id="user_clean_001",
        content="Oh sure, they're SUCH wonderful people always causing trouble wherever they go.",
        report_reason="Harassment",
        expected_final_action="uphold_report",
        expected_violation_code="hate_speech",
        required_tools=["search_handbook"],
        grader="implicit_hate_grader"
    )
]


def get_task(task_id: str) -> TaskModel:
    for t in TASKS:
        if t.id == task_id:
            return t
    raise ValueError(f"Task {task_id} not found.")


def get_all_tasks() -> List[TaskModel]:
    return TASKS


def list_tasks():
    return [{"id": t.id, "grader": t.grader} for t in TASKS]
