from __future__ import annotations
import copy
import random

from models import Action, Observation, Reward, StepOutput, EnvState
from policy_db import search_handbook, request_fact_check, get_user_history
from tasks import get_task


class ModeratorEnv:
   

    

    def __init__(self, max_steps=7, seed=None):
        self.max_steps = max_steps
        self._rng = random.Random(seed)
        self._state: EnvState | None = None
        self.current_step = 0

    def reset(self, task_id=None):
        self.current_step = 0
        task = get_task(task_id or "task_easy_001")

        self._state = EnvState(
            task_id=task.id,
            user_id=task.user_id,
            content=task.content,
            report_reason=task.report_reason,
            done=False
        )

        return self._state.to_observation()

    def step(self, action_dict):
        if self._state is None:
            raise RuntimeError("Call reset() first")

        if self._state.done:
            raise RuntimeError("Episode finished. Call reset()")

        self.current_step += 1
        self._state.step_count = self.current_step

        action = Action(**action_dict)
        reward, info = self._dispatch(action)

        # Step limit penalty (NON-NEGATIVE)
        if self.current_step >= self.max_steps:
            self._state.done = True
            reward = max(0.0, reward - 0.3)

        return StepOutput(
            observation=self._state.to_observation(),
            reward=Reward(value=round(reward, 3)),
            done=self._state.done,
            info=info
        )

    def state(self):
        return copy.deepcopy(self._state)

    def _dispatch(self, action: Action):
        handlers = {
            "search_handbook": self._act_search,
            "request_user_history": self._act_history,
            "request_fact_check": self._act_fact_check,
            "uphold_report": self._act_uphold,
            "dismiss_report": self._act_dismiss,
            "issue_warning": self._act_warn,
            "escalate_to_human": self._act_escalate,
            "do_nothing": self._act_noop
        }
        return handlers.get(action.action_type, self._act_noop)(action)

    # -----------------------
    # TOOL ACTIONS
    # -----------------------

    def _act_search(self, action):
        query = action.argument or self._state.report_reason
        result = search_handbook(query)

        self._state.retrieved_policy = result
        self._state.policy_requested = True

        return 0.25, {"policy": "loaded"}

    def _act_history(self, action):
        history = get_user_history(self._state.user_id)

        self._state.retrieved_user_history = history
        self._state.history_requested = True

        return 0.25, {"history": True}

    def _act_fact_check(self, action):
        result = request_fact_check(self._state.content)

        self._state.retrieved_fact_check = result
        self._state.fact_checked = True

        return 0.25, {"fact_checked": True}

    # -----------------------
    # FINAL REWARD (FIXED)
    # -----------------------

   

    def _final_reward(self, correct):
        if self._state is None:
            return 0.0

        task = get_task(self._state.task_id)

        state = self._state
        required_tools = getattr(task, "required_tools", [])

        reward = 0.0

        # 🔻 Step penalty (discourage long reasoning)
        reward -= 0.01 * getattr(state, "steps", 0)

        # ✅ Tool usage rewards
        tools_used = getattr(state, "tools_used", {})

        if tools_used.get("search_handbook"):
            reward += 0.2

        if tools_used.get("request_fact_check"):
            reward += 0.2

        if tools_used.get("request_user_history"):
            reward += 0.2

        # ⚠️ Required tool not used → penalty
        if "request_user_history" in required_tools:
            if not getattr(state, "retrieved_user_history", None):
                reward -= 0.2

        # ✅ Final correctness
        if correct:
            reward += 1.0
        else:
            reward -= 0.5

        return round(reward, 3)

    # -----------------------
    # DECISION ACTIONS
    # -----------------------

    def _act_uphold(self, action):
        self._state.done = True
        task = get_task(self._state.task_id)
        correct = task.expected_final_action == "uphold_report"

        return self._final_reward(correct), {"correct": correct}

    def _act_dismiss(self, action):
        self._state.done = True
        task = get_task(self._state.task_id)
        correct = task.expected_final_action == "dismiss_report"

        return self._final_reward(correct), {"correct": correct}

    def _act_escalate(self, action):
        self._state.done = True
        task = get_task(self._state.task_id)
        correct = task.expected_final_action == "escalate_to_human"

        return self._final_reward(correct), {"correct": correct}

    def _act_warn(self, action):
        self._state.done = True

        if self._state.retrieved_user_history:
            self._state.retrieved_user_history["strikes"] += 1
        else:
            history = get_user_history(self._state.user_id)
            history["strikes"] += 1
            self._state.retrieved_user_history = history

        return 0.3, {"warning": True}

    def _act_noop(self, action):
        return 0.05, {"msg": "no_effect"}