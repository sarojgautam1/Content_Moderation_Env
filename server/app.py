from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, List

from environment import ModeratorEnv
from inference import get_action as choose_action
from models import TrajectoryStep, Action
from grader import evaluate_trajectory

app = FastAPI()
env = ModeratorEnv()


class RunRequest(BaseModel):
    task_id: Optional[str] = None


@app.post("/reset")
def reset(req: RunRequest = None):
    task_id = req.task_id if req else None
    obs = env.reset(task_id)
    return obs.model_dump(mode="json")


@app.post("/step")
def step(action: Dict):
    result = env.step(action)
    return result.model_dump(mode="json")


@app.get("/")
def read_root():
    return {"status": "Moderator Env Running"}

@app.get("/state")
def state():
    st = env.state()
    if st is None:
        return {"error": "Environment not reset"}
    return st.model_dump(mode="json")


@app.get("/tasks")
def list_tasks_route():
    from tasks import get_all_tasks
    tasks = get_all_tasks()
    return [{"task_id": t.id, "has_grader": True, **t.model_dump(mode="json")} for t in tasks]

class EvaluateRequest(BaseModel):
    task_id: str
    trajectory: List[TrajectoryStep]

@app.post("/evaluate")
def evaluate(req: EvaluateRequest):
    score = evaluate_trajectory(req.task_id, req.trajectory)
    return {"score": score}


@app.post("/run")
def run(req: RunRequest):
    obs = env.reset(req.task_id)

    done = False
    step_number = 0
    rewards = []
    trajectory = []

    print(f"[START] task={env._state.task_id} env=moderator_env model=rule_based")

    try:
        while not done and step_number < 8:
            step_number += 1

            action_dict = choose_action(obs)

            try:
                result = env.step(action_dict)
                error = None
            except Exception as e:
                result = None
                error = str(e)

            if result:
                reward = result.reward.value
                done = result.done
                next_obs = result.observation
            else:
                reward = 0.0
                done = True
                next_obs = obs

            rewards.append(reward)

            print(
                f"[STEP] step={step_number} action={action_dict['action_type']} "
                f"reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}"
            )

            trajectory.append(
                TrajectoryStep(
                    step_number=step_number,
                    observation=obs,
                    action=Action(**action_dict),
                    reward=reward
                )
            )

            obs = next_obs

        score = evaluate_trajectory(env._state.task_id, trajectory)
        score = max(0.0, min(score, 1.0))
        success = score > 0.5

    except Exception as e:
        print(f"[DEBUG] Fatal error: {e}")
        score = 0.0
        success = False

    finally:
        print(
            f"[END] success={str(success).lower()} steps={step_number} "
            f"score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}"
        )

    return {
        "score": score,
        "steps": step_number,
        "rewards": rewards
    }
def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
    
