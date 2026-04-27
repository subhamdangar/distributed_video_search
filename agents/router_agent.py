"""
Router Agent (Minimal)
──────────────────────
Uses OpenRouter LLM to classify query into:
  - mathematics
  - computer_science
"""

import requests

# 🔴 Paste your API key here
OPENROUTER_API_KEY = (
    "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"
)

VALID_SUBJECTS = {"mathematics", "computer_science"}


def llm_route(query: str) -> str:
    prompt = f"""
You are a strict classifier.

Classify the query into EXACTLY ONE subject:
- mathematics
- computer_science

Return ONLY one word.

Query: "{query}"
"""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "meta-llama/llama-3-8b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
    )
    print(response.json())
    output = response.json()["choices"][0]["message"]["content"].strip().lower()

    print(f"[LLM ROUTER] Output: {output}")
    
    
    output = output.strip().lower()

    valid_subjects = ["computer_science", "maths", "physics"]

    if output not in valid_subjects:
        return None   # IMPORTANT: no exception

    return output
    
    
    # if output not in VALID_SUBJECTS:
    #     raise ValueError(f"Invalid subject returned: {output}")

    # return output


class RouterAgent:

    # def route(self, query: str):
    #     print(f"[SERVER] Routing query: {query}")

    #     subject = llm_route(query)

    #     print(f"[SERVER] Routing → {subject}")

    #     return [subject]
    
    # def route(self, query):
    #     output = llm_route(query).strip().lower()

    #     valid_subjects = ["computer_science", "maths"]  # your subjects

    #     if output in valid_subjects:
    #         print(f"[ROUTER] → Subject: {output}")
    #         return {"type": "youtube", "subjects": [output]}

    #     else:
    #         print(f"[ROUTER] → Fallback to WEB (LLM output: {output})")
    #         return {"type": "web", "subjects": []}
        
        
    def route(self, query):
        output = llm_route(query)

        if output is None:
            print("[ROUTER] → Web fallback triggered")
            return {"type": "web", "subjects": []}

        print(f"[ROUTER] → Subject: {output}")
        return {"type": "youtube", "subjects": [output]}
