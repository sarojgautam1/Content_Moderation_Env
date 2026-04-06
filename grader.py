from typing import List
from models import TrajectoryStep
from tasks import get_task
import random


def evaluate_trajectory(task_id: str, trajectory: List[TrajectoryStep]) -> float:
    try:
        task = get_task(task_id)
    except:
        return 0.0

    if not trajectory:
        return 0.0

    score = 0.0

    actions = [step.action.action_type for step in trajectory]
    final_action = trajectory[-1].action.action_type

    # ---------------------------
    # 1. TOOL USAGE (0.25)
    # ---------------------------
    required_tools = set(getattr(task, "required_tools", []))
    used_tools = set(actions)

    if required_tools:
        tool_score = len(required_tools & used_tools) / len(required_tools)
        score += 0.25 * tool_score
    else:
        score += 0.25

    # ❗ Penalize missing required tools 
    if required_tools: 
        missing_tools = required_tools - used_tools 
        if missing_tools: 
            score -= 0.1 * len(missing_tools)
    # ---------------------------
    # 2. ORDER AWARENESS (0.15)
    # ---------------------------
    decision_index = None
    for i, a in enumerate(actions):
        if a in ["uphold_report", "dismiss_report", "escalate_to_human"]:
            decision_index = i
            break

    if decision_index is not None:
        tools_before_decision = set(actions[:decision_index]) 
        required_before = required_tools.intersection( 
            {"search_handbook", "request_fact_check", "request_user_history"} 
        ) 
        if required_before.issubset(tools_before_decision): 
            score += 0.15 
        else: 
            score -= 0.05

    # ---------------------------
    # 3. FINAL DECISION (0.4)
    # ---------------------------
    if final_action == task.expected_final_action:
        score += 0.4
    else: 
        score += 0.05 # partial credit

    # ---------------------------
    # 4. EFFICIENCY BONUS
    # ---------------------------
    if len(actions) <= 3:
        score += 0.1
    elif len(actions) <= 5:
        score += 0.05
    else:
        score -= 0.05

    # ---------------------------
    # 5. PENALTIES
    # ---------------------------
    if len(actions) > 6:
        score -= 0.1

    if actions.count("do_nothing") > 1:
        score -= 0.1

    # 🔥 Important: Policy must be used 
    if "search_handbook" not in actions: 
        score -= 0.1

    # ---------------------------
    # 6. DIFFICULTY SCALING (KEY)
    # ---------------------------
    difficulty = task.difficulty.lower()

    if difficulty == "easy":
        score *= 1.0
    elif difficulty == "medium":
        score *= 0.85
    elif difficulty == "hard":
        score *= 0.7

    # ✅ Clamp AFTER scaling 
    score = max(0.0, min(score, 1.0))
    return round(score, 3)