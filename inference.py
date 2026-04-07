import os
import asyncio
from typing import List, Optional

from openai import OpenAI

from environment import ModeratorEnv
from models import Action, TrajectoryStep
from grader import evaluate_trajectory

# =========================
# CONFIG
# =========================
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

MAX_STEPS = 8

# Safe OpenAI client
client = None
if API_KEY:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# =========================
# LOGGING (STRICT FORMAT)
# =========================
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True
    )

# =========================
# POLICY (FINAL OPTIMIZED)
# =========================
def get_action(obs):

    reason = obs.report_reason.lower()
    content = obs.content.lower()

    # 1. ALWAYS GET POLICY FIRST
    if obs.policy_snippet == "":
        return {"action_type": "search_handbook", "argument": obs.report_reason}

    # 2. HATE
    if "hate" in reason or "slur" in reason:
        return {"action_type": "uphold_report", "argument": "hate_speech"}

    # 3. MISINFORMATION
    if "misinformation" in reason:
        if obs.fact_check_result == "":
            return {"action_type": "request_fact_check", "argument": obs.content}

        if obs.fact_check_result.startswith("FALSE"):
            return {"action_type": "uphold_report", "argument": "health_misinfo"}
        else:
            return {"action_type": "dismiss_report", "argument": None}

    # 4. HARASSMENT
    if "harassment" in reason:
        if obs.user_strike_count is None:
            return {"action_type": "request_user_history", "argument": None}

        if obs.user_strike_count >= 3:
            return {"action_type": "escalate_to_human", "argument": None}
        else:
            return {"action_type": "dismiss_report", "argument": None}

    return {"action_type": "dismiss_report", "argument": None}


# =========================
# MAIN LOOP
# =========================
async def main():
    env = ModeratorEnv()

    rewards = []
    trajectory = []
    steps_taken = 0
    score = 0.0
    success = False

    obs = env.reset()
    task_id = env._state.task_id

    log_start(task=task_id, env="moderator_env", model=MODEL_NAME)

    try:
        for step in range(1, MAX_STEPS + 1):

            action_dict = get_action(obs)

            try:
                result = env.step(action_dict)
                error = None
            except Exception as e:
                result = None
                error = str(e)

            if result:
                reward = result.reward.value
                done = result.done
                obs_next = result.observation
            else:
                reward = 0.0
                done = True
                obs_next = obs

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_dict["action_type"],
                reward=reward,
                done=done,
                error=error
            )

            trajectory.append(
                TrajectoryStep(
                    step_number=step,
                    observation=obs,
                    action=Action(**action_dict),
                    reward=reward
                )
            )

            obs = obs_next

            if done:
                break

        # Final score
        score = evaluate_trajectory(task_id, trajectory)
        score = max(0.0, min(score, 1.0))
        success = score > 0.5

    except Exception as e:
        print(f"[DEBUG] Fatal error: {e}", flush=True)

    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards
        )


if __name__ == "__main__":
    asyncio.run(main())