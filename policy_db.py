from typing import Dict, Any, Optional

def request_fact_check(query: str) -> str:
    base = "No fact-check available..."

    return "UNVERIFIED: conflicting sources"

    

# Mock Knowledge Base for Policy Handbook
POLICY_HANDBOOK = {
    "hate": "Policy 101 (Hate Speech): Any language that attacks a person or group based on race, religion, ethnicity, or gender is strictly prohibited. Correct action: uphold_report with violation code 'hate_speech'.",
    "misinformation": "Policy 202 (Misinformation): False claims that contradict public health guidelines or fact-checked realities. Correct action: uphold_report with violation code 'health_misinfo'. Do not remove if fact-check is UNVERIFIED.",
    "health": "Policy 202 (Misinformation): False claims that contradict public health guidelines or fact-checked realities. Correct action: uphold_report with violation code 'health_misinfo'. Do not remove if fact-check is UNVERIFIED.",
    "satire": "Policy 303 (Satire Exception): Content that is explicitly labeled or obviously parody is exempt from regular harassment/bullying rules. However, if the user has 3 or more past strikes, automatically escalate the case. Otherwise, dismiss the report.",
    "harassment": "Policy 404 (Harassment): Repeated unwanted contact or bullying. Correct action: uphold_report with violation code 'harassment'."
}

# Mock User DB (user_id -> strike_count)
USER_DB = {
    "user_clean_001": {"strikes": 0, "status": "good_standing"},
    "user_warned_002": {"strikes": 1, "status": "warned"},
    "user_repeat_003": {"strikes": 3, "status": "probation"}, # Will trigger escalation in the hard task
}

# Mock Fact Check DB (claim substring -> fact)
FACT_CHECK_DB = {
    "lemon juice cures": "FALSE: No scientific evidence supports that lemon juice cures serious illnesses. This contradicts global medical consensus.",
    "miracle drops": "FALSE: 'Miracle drops' have been debunked by the FDA as a scam.",
}

def search_handbook(query: str) -> str:
    """Returns matching policy snippets based on keyword search."""
    if not query:
        return "Please provide a search query."
    
    query = query.lower()
    results = []
    
    # Simple semantic/keyword match
    for key, text in POLICY_HANDBOOK.items():
        if key in query or any(word in query.split() for word in key.split()):
            if text not in results:
                results.append(text)
    
    if not results:
        # Fallback keyword match against the text
        for key, text in POLICY_HANDBOOK.items():
            if any(word in text.lower() for word in query.split() if len(word) > 4):
                 if text not in results:
                    results.append(text)
                    
    if not results:
        return "No matching policy found in handbook. Try checking for 'hate', 'misinformation', 'satire', or 'harassment'."
    
    return "\n---\n".join(results)

def get_user_history(user_id: str) -> Dict[str, Any]:
    """Returns the user's strike count and status."""
    return USER_DB.get(user_id, {"strikes": 0, "status": "unknown"})

def request_fact_check(query: str) -> str:
    """Returns fact check results for a given query."""
    if not query:
        return "Please provide a claim to fact-check."
        
    query = query.lower()
    for key, fact in FACT_CHECK_DB.items():
        if key in query or any(word in query.split() for word in key.split() if len(word) > 4):
            return fact
            
    return "No fact-check available for this specific query in our records."
