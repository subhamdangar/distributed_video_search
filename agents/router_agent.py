"""
Router Agent (Minimal)
──────────────────────
Uses OpenRouter LLM to classify query into:
  - mathematics
  - computer_science
"""

import requests
import os
from dotenv import load_dotenv

load_dotenv()


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found")



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

    if output not in VALID_SUBJECTS:
        return None   # IMPORTANT: no exception

    return output
    


class RouterAgent:    
    def route(self, query):
        output = llm_route(query)

        if output is None:
            print("[ROUTER] → Web fallback triggered")
            return {"type": "web", "subjects": []}

        print(f"[ROUTER] → Subject: {output}")
        return {"type": "youtube", "subjects": [output]}
