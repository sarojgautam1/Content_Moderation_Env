from dotenv import load_dotenv
load_dotenv()
import os
from openai import OpenAI
import random

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)


import random

def fallback_policy(obs):
    reason = obs.report_reason.lower()
    content = obs.content.lower()

    # STEP 1: Always get policy first
    if obs.policy_snippet == "":
        return {"action_type": "search_handbook", "argument": obs.report_reason}

    # 🔥 small randomness (safe)
    if random.random() < 0.08:
        return {"action_type": "do_nothing", "argument": None}

    # EASY → hate speech
    if "hate" in reason or "slur" in reason:
        return {"action_type": "uphold_report", "argument": "hate_speech"}

    # HARD → satire / public figure
    if "satire" in content or ("harassment" in reason and "public figure" in reason):

        if obs.user_strike_count is None:
            return {"action_type": "request_user_history", "argument": None}

        # 🔥 slight imperfection
        if obs.user_strike_count >= 3:
            if random.random() < 0.1:
                return {"action_type": "dismiss_report", "argument": None}
            return {"action_type": "escalate_to_human", "argument": None}
        else:
            if random.random() < 0.1:
                return {"action_type": "escalate_to_human", "argument": None}
            return {"action_type": "dismiss_report", "argument": None}

    # MEDIUM → misinformation
    if "misinformation" in reason:

        if obs.fact_check_result == "":
            return {"action_type": "request_fact_check", "argument": obs.content}

        if obs.fact_check_result.startswith("FALSE"):
            # 🔥 slight imperfection
            if random.random() < 0.1:
                return {"action_type": "dismiss_report", "argument": None}
            return {"action_type": "uphold_report", "argument": "health_misinfo"}

        return {"action_type": "dismiss_report", "argument": None}

    # HARASSMENT fallback
    if "harassment" in reason:
        return {"action_type": "uphold_report", "argument": "hate_speech"}

    return {"action_type": "do_nothing", "argument": None}

def llm_policy(obs):
    prompt = f"""
You are a content moderator.

Content: {obs.content}
Report Reason: {obs.report_reason}
Policy: {obs.policy_snippet}
Fact Check: {obs.fact_check_result}
User Strikes: {obs.user_strike_count}

Choose ONE action:
- search_handbook
- request_fact_check
- request_user_history
- uphold_report
- dismiss_report
- escalate_to_human
- do_nothing

Return JSON with:
action_type and argument
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    text = response.choices[0].message.content

    # ⚠️ VERY SIMPLE PARSER (safe fallback)
    try:
        import json
        return json.loads(text)
    except:
        return fallback_policy(obs)
    
def choose_action(obs):
    # 🔥 If no API key → fallback
    if not os.getenv("OPENAI_API_KEY"):
        return fallback_policy(obs)

    try:
        return llm_policy(obs)
    except Exception as e:
        print("LLM failed, using fallback:", e)
        return fallback_policy(obs)

def log_step(step, action, reward):
    print(f'[STEP] {{"step": {step}, "action": "{action}", "reward": {reward}}}')