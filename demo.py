from models import Observation, Action, TrajectoryStep
from grader import evaluate_trajectory
from tasks import get_task
from policy_db import search_handbook

def test_easy_task_perfect_score():
    print("--- Testing Easy Task (Perfect Trajectory) ---")
    task_id = "task_easy_001"
    task = get_task(task_id)
    
    # Step 1: Agent observes and searches the handbook
    obs1 = Observation(
        content=task.content,
        report_reason=task.report_reason,
        step_number=1
    )
    action1 = Action(action_type="search_handbook", argument="hate_speech")
    step1 = TrajectoryStep(step_number=1, observation=obs1, action=action1)
    
    # Agent gets handbook result (simulated internally)
    policy_result = search_handbook("hate_speech")
    
    # Step 2: Agent makes correct final decision based on handbook
    obs2 = Observation(
        content=task.content,
        report_reason=task.report_reason,
        policy_snippet=policy_result,
        step_number=2
    )
    # The expected action is uphold_report with violation code "hate_speech"
    action2 = Action(action_type="uphold_report", argument="hate_speech")
    step2 = TrajectoryStep(step_number=2, observation=obs2, action=action2)
    
    trajectory = [step1, step2]
    
    score = evaluate_trajectory(task_id, trajectory)
    print(f"Task: {task.difficulty} | Final Decision: {action2.action_type}")
    print(f"Expected Score: 1.0 | Actual Score: {score}\n")

def test_hard_task_wrong_decision():
    print("--- Testing Hard Task (Wrong Trajectory) ---")
    # This task requires escalating because the user has 3 strikes.
    task_id = "task_hard_001" 
    task = get_task(task_id)
    
    obs1 = Observation(content=task.content, report_reason=task.report_reason, step_number=1)
    
    # Agent doesn't search user history or handbook, just dismisses the report!
    # Missing required tools & wrong final argument
    action1 = Action(action_type="dismiss_report")
    step1 = TrajectoryStep(step_number=1, observation=obs1, action=action1)
    
    trajectory = [step1]
    
    score = evaluate_trajectory(task_id, trajectory)
    print(f"Task: {task.difficulty} | Final Decision: {action1.action_type}")
    print(f"Agent missed context and made wrong choice.")
    print(f"Score should be very low | Actual Score: {score}\n")

if __name__ == "__main__":
    test_easy_task_perfect_score()
    test_hard_task_wrong_decision()
