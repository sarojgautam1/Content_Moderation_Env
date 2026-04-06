from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from environment import ModeratorEnv
from inference import choose_action
from models import TrajectoryStep, Action
from grader import evaluate_trajectory

app = FastAPI()
env = ModeratorEnv()


class RunRequest(BaseModel):
    task_id: Optional[str] = None

@app.post("/reset")
def reset(req: RunRequest):
    obs = env.reset(req.task_id)
    return obs.model_dump(mode='json')

from typing import Dict

@app.post("/step")
def step(action: Dict):
    result = env.step(action)
    return result.model_dump(mode="json")

@app.get("/state")
def state():
    return env.state().model_dump(mode="json")

@app.get("/tasks")
def list_tasks():
    from tasks import list_tasks
    return list_tasks()

@app.post("/run")
def run(req: RunRequest):
    obs = env.reset(req.task_id)

    done = False
    total_reward = 0.0
    trajectory = []
    step_number = 0

    print("[START]")

    while not done:
        step_number += 1

        # Get action from model
        action_dict = choose_action(obs)

        # Step environment
        result = env.step(action_dict)

        # Log step (structured)
        print(f'[STEP] {{"step": {step_number}, "action": "{action_dict["action_type"]}", "reward": {result.reward.value}}}')

        # Store trajectory (CORRECT ORDER)
        trajectory.append(
            TrajectoryStep(
                step_number=step_number,
                observation=obs,
                action=Action(**action_dict),
                reward=result.reward.value
            )
        )

        # Update state
        obs = result.observation
        total_reward += result.reward.value
        done = result.done

    print("[END]")

    # Safety check
    if not trajectory:
        print("Warning: Empty trajectory")

    score = evaluate_trajectory(req.task_id, trajectory)

    return {
        "total_reward": round(total_reward, 4),
        "evaluation_score": score
    }