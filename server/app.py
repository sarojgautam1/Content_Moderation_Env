import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import ModeratorEnv
from grader import evaluate_trajectory
from models import TrajectoryStep, Action, Observation, EnvState
from tasks import get_all_tasks, list_tasks

app = FastAPI(
    title="Content Moderation Env",
    version="1.0.0",
    description="Multi-step AI content moderation environment with tool usage and reasoning"
)

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
# CORE ENDPOINTS
# --------------------

@app.get("/")
def root():
    return {"status": "ok", "message": "Content Moderation Env is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    return {
        "name": "content-moderation-env",
        "description": "Multi-step AI content moderation environment with tool usage and reasoning",
        "version": "1.0.0",
        "author": "team"
    }


@app.get("/schema")
def schema():
    return {
        "action": {
            "type": "object",
            "properties": {
                "action_type": {
                    "type": "string",
                    "enum": [
                        "uphold_report", "dismiss_report", "issue_warning",
                        "escalate_to_human", "search_handbook",
                        "request_user_history", "request_fact_check", "do_nothing"
                    ]
                },
                "argument": {"type": "string", "nullable": True}
            },
            "required": ["action_type"]
        },
        "observation": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "report_reason": {"type": "string"},
                "user_strike_count": {"type": "integer", "nullable": True},
                "policy_snippet": {"type": "string"},
                "fact_check_result": {"type": "string"},
                "step_number": {"type": "integer"}
            }
        },
        "state": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "user_id": {"type": "string"},
                "content": {"type": "string"},
                "report_reason": {"type": "string"},
                "step_count": {"type": "integer"},
                "done": {"type": "boolean"}
            }
        }
    }


@app.api_route("/mcp", methods=["GET", "POST"])
async def mcp(request: dict = None):
    return {
        "jsonrpc": "2.0",
        "result": {
            "tools": [
                {"name": "search_handbook", "description": "Search the policy handbook"},
                {"name": "request_fact_check", "description": "Request a fact check"},
                {"name": "request_user_history", "description": "Get user history"},
                {"name": "uphold_report", "description": "Uphold a report"},
                {"name": "dismiss_report", "description": "Dismiss a report"},
                {"name": "escalate_to_human", "description": "Escalate to human moderator"},
                {"name": "issue_warning", "description": "Issue a warning"},
                {"name": "do_nothing", "description": "Take no action"}
            ]
        },
        "id": None
    }


# --------------------
# ENV ENDPOINTS
# --------------------

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
    all_tasks = get_all_tasks()
    return [
        {
            "id": t.id,
            "grader": t.grader,
            "difficulty": t.difficulty,
            "expected_final_action": t.expected_final_action,
            "required_tools": t.required_tools
        }
        for t in all_tasks
    ]


# --------------------
# MAIN (REQUIRED BY OPENENV VALIDATOR)
# --------------------

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
