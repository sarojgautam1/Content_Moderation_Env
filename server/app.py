from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from environment import ModeratorEnv
from grader import evaluate_trajectory
from models import TrajectoryStep, Action, Observation
from tasks import get_all_tasks, list_tasks

app = FastAPI(title="Content Moderation Env")

# Single shared env instance (stateful per session)
env = ModeratorEnv()


# --------------------
# REQUEST MODELS
# --------------------
class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_easy_001"

class StepRequest(BaseModel):
    action_type: str
    argument: Optional[str] = None

class RunRequest(BaseModel):
    task_id: Optional[str] = "task_easy_001"

class EvaluateRequest(BaseModel):
    task_id: str
    trajectory: List[TrajectoryStep]


# --------------------
# ENDPOINTS
# --------------------

@app.get("/")
def root():
    return {"status": "ok", "message": "Content Moderation Env is running"}


@app.post("/reset")
def reset(req: ResetRequest = None):
    task_id = (req.task_id if req else None) or "task_easy_001"
    obs = env.reset(task_id)
    return obs.dict()


@app.post("/step")
def step(req: StepRequest):
    try:
        result = env.step({"action_type": req.action_type, "argument": req.argument})
        return {
            "observation": result.observation.dict(),
            "reward": result.reward.dict(),
            "done": result.done,
            "info": result.info
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    s = env.state()
    if s is None:
        return {"detail": "No active episode. Call /reset first."}
    return s.dict()


@app.post("/evaluate")
def evaluate(req: EvaluateRequest):
    score = evaluate_trajectory(req.task_id, req.trajectory)
    return {"score": score}


@app.post("/run")
def run(req: RunRequest = None):
    from inference import get_action
    from models import TrajectoryStep, Action

    task_id = (req.task_id if req else None) or "task_easy_001"
    obs = env.reset(task_id)

    trajectory = []
    total_reward = 0.0

    for step_num in range(1, 9):
        action_dict = get_action(obs)
        result = env.step(action_dict)

        trajectory.append(TrajectoryStep(
            step_number=step_num,
            observation=obs,
            action=Action(**action_dict),
            reward=result.reward.value
        ))

        total_reward += result.reward.value
        obs = result.observation

        if result.done:
            break

    score = evaluate_trajectory(task_id, trajectory)
    return {
        "total_reward": round(total_reward, 3),
        "evaluation_score": round(score, 3),
        "steps": len(trajectory)
    }


@app.get("/tasks")
def tasks():
    return list_tasks()
