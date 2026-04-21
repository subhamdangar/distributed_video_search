"""
Router Agent (Minimal)
──────────────────────
Uses OpenRouter LLM to classify query into:
  - mathematics
  - computer_science
"""

import requests

# 🔴 Paste your API key here
OPENROUTER_API_KEY = "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"

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
            "Content-Type": "application/json"
        },
        json={
            "model": "meta-llama/llama-3-8b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        }
    )

    output = response.json()["choices"][0]["message"]["content"].strip().lower()

    print(f"[LLM ROUTER] Output: {output}")

    if output not in VALID_SUBJECTS:
        raise ValueError(f"Invalid subject returned: {output}")

    return output


class RouterAgent:

    def route(self, query: str):
        print(f"[SERVER] Routing query: {query}")

        subject = llm_route(query)

        print(f"[SERVER] Routing → {subject}")

        return [subject]