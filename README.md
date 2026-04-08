---
title: Content Moderation Env
sdk: docker
emoji: 💻
colorFrom: blue
colorTo: purple
short_description: AI moderation using reasoning, tools, and policy logic
---
# Content Policy & Trust Moderator (OpenEnv)

## Overview
This project implements a multi-step AI moderation environment where an agent behaves like a senior content moderator.

Unlike traditional classifiers, this system:

- Uses tools (policy lookup, fact-checking, user history)
- Operates under partial information
- Makes decisions step-by-step
- Produces explainable moderation outcomes

## Motivation
Modern content moderation systems require:

- Context awareness
- Policy grounding
- Sequential reasoning

This project simulates a real-world moderation workflow, enabling evaluation of:

- reasoning quality
- tool usage
- decision correctness

## System Architecture
**Environment(environment.py)**
-Implements OpenEnv interface:
    -reset(task_id)
    -step(action)
    -state()
-Simulates:
    -policy lookup
    -fact checking
    -user history
-Includes reward shaping

**Agent(inference.py)**
-Hybrid agent:
    -LLM-based (via OpenAI client)
    -Rule-based fallback (no API dependency)
-Produces structured logs:
    -[START], [STEP], [END]

**Tasks(tasks.py)**
-Includes multiple difficulty levels:

    -Easy → clear violations (hate speech)
    -Medium → misinformation
    -Hard → satire + user history

**Grader(grader.py)**
-Evaluates full trajectory (0.0 – 1.0) based on:

    -tool usage
    -reasoning order
    -final decision
    -efficiency

**Models(models.py)**
-Typed Pydantic models:
    -Observation
    -Action
    -Reward
    -State
    -TrajectoryStep

**API(main.py)**
-FastAPI endpoints:

    Endpoint	Description
    POST /reset	Initialize environment
    POST /step	Take action
    GET /state	Get current state
    POST /run	Run full episode
    GET /tasks	List available tasks

## Action Space
-The agent can perform the following actions:

    -search_handbook
    -request_fact_check
    -request_user_history
    -uphold_report
    -dismiss_report
    -escalate_to_human
    -do_nothing

## Observation Space
-Each observation includes:

    -content (str)
    -report_reason (str)
    -policy_snippet (str)
    -fact_check_result (str)
    -user_strike_count (int or null)

## Reward Design

-The environment uses shaped rewards:

    -Tool usage → positive reward
    -Correct decision → high reward
    -Missing required tools → penalty
    -Inefficient reasoning → penalty

-This encourages structured, explainable decision-making.

## Available Tasks

-Use the /tasks endpoint to list all available tasks.

-Example:
  GET /tasks

-Response:
  [
    {"task_id": "task_easy_001"},
    {"task_id": "task_medium_001"},
    {"task_id": "task_medium_002"},
    {"task_id": "task_hard_001"},
    {"task_id": "task_hard_002"},
    {"task_id": "task_hard_003"}
  ]

-Use any task_id from this list when calling /run or /reset.

## Example Flow
    [START]
    Step 1 → search_handbook
    Step 2 → request_fact_check
    Step 3 → uphold_report
    [END]

## Setup Instructions
    1. Install dependencies
    pip install -r requirements.txt

    2. Run locally
    uvicorn server.app:app --reload

    Open:
    http://localhost:8000/docs

    3. Run with Docker
    docker build -t moderator-env .
    docker run -p 7860:7860 moderator-env

    Open:
    http://localhost:7860/docs
    
    4. Run on HuggingFace
    https://sar1505-content-moderation-env.hf.space

    For the UI app:
    https://sar1505-content-moderation-env.hf.space/docs

## Run Example

    POST /run

    Request:
    {
      "task_id": "task_easy_001"
    }

    Response:
    {
      "score": 0.92,
      "steps": 2,
      "rewards": [0.25, 1.0]
    }

## API Usage Examples

### 1. Reset Environment

**Request:**

```json
POST /reset
{
  "task_id": "task_easy_001"
}
```

**Response:**

```json
{
  "content": "People from X are criminals",
  "report_reason": "Hate Speech",
  "policy_snippet": "",
  "fact_check_result": "",
  "user_strike_count": null
}
```

---

### 2. Take a Step

**Request:**

```json
POST /step
{
  "action_type": "search_handbook",
  "argument": "Hate Speech"
}
```

**Response:**

```json
{
  "observation": {...},
  "reward": {...},
  "done": false
}
```

---

### 3. Evaluate Trajectory

**Request:**

```json
POST /evaluate
{
  "task_id": "task_easy_001",
  "trajectory": [
    {
      "step_number": 1,
      "observation": {
        "content": "People from that specific religion are all dirty criminals and should be thrown out of our city. We don't want them here.",
        "report_reason": "Hate Speech / Slurs",
        "user_strike_count": null,
        "policy_snippet": "",
        "fact_check_result": "",
        "step_number": 1
      },
      "action": {
        "action_type": "search_handbook",
        "argument": "hate speech"
      },
      "reward": 0.25
    },
    {
      "step_number": 2,
      "observation": {
        "content": "People from that specific religion are all dirty criminals and should be thrown out of our city. We don't want them here.",
        "report_reason": "Hate Speech / Slurs",
        "user_strike_count": null,
        "policy_snippet": "Policy 101 (Hate Speech): Any language that attacks...",
        "fact_check_result": "",
        "step_number": 2
      },
      "action": {
        "action_type": "uphold_report",
        "argument": "hate_speech"
      },
      "reward": 1.0
    }
  ]
}
```

**Response:**

```json
{
  "score": 1.0
}
```

---

### 4. Run Full Episode

**Request:**

```json
POST /run
{
  "task_id": "task_easy_001"
}
```

**Response:**

```json
{
  "total_reward": 1.25,
  "evaluation_score": 0.92
}
```

---

### 5. Get State

```json
GET /state
```

---

### 6. List Tasks

```json
GET /tasks
```


## Environment Variables

-Set the following variables:

    -API_BASE_URL → LLM API endpoint
    -MODEL_NAME → model name (e.g. gpt-4o-mini)
    -HF_TOKEN → API key

## Baseline Behavior
-Supports both:
    -LLM-based reasoning
    -rule-based fallback

-Produces:
    -stable rewards
    -deterministic evaluation

## Deployment

-The project is designed to run on:
    -Docker
    -Hugging Face Spaces (containerized)

## Key Highlights
-Multi-step reasoning (not classification)
-Tool-augmented decision making
-Partial observability
-Trajectory-based evaluation
-Robust fallback system

## Novelty and Creativity
- Multi-tool dependency
  - Policy + fact-check + history
- Satire + strike interaction
  - if strikes >= 3 → escalate
  - else → dismiss
- Order-aware grading
  - Tools must come before decision

## Future Improvements
-Real policy database integration
-Better fact-checking models
-Reinforcement learning agent
-Human-in-the-loop moderation

## Conclusion
-This project demonstrates how moderation systems can move beyond classification toward reasoning-driven, policy-grounded decision making, aligning with real-world trust & safety workflows.
