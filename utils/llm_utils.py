import json
import os

import requests

from utils import timeit

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://0.0.0.0:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(MODEL_NAME)


@timeit
def generate_llm_answer(prompt: str, model_name: str = MODEL_NAME) -> str:
    """
    Generate an LLM answer using the Groq API (OpenAI-compatible endpoint).
    Requires GROQ_API_KEY to be set in the environment.
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": model_name, "messages": [{"role": "user", "content": prompt}]}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        # OpenAI format: choices[0].message.content
        return data["choices"][0]["message"]["content"].strip()
    else:
        raise RuntimeError(f"Groq API error: {response.text}")


@timeit
def generate_llm_answer_local(prompt: str, model_name: str = MODEL_NAME) -> str:
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model_name, "prompt": prompt, "stream": True},
        stream=True,
    )
    answer = ""
    if response.status_code == 200:
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    answer += data["response"]
                if data.get("done"):
                    break
            except Exception:
                continue
        return answer.strip()
    else:
        raise RuntimeError(f"Ollama API error: {response.text}")


"""
from groq import Groq

client = Groq()
completion = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
      {
        "role": "user",
        "content": "my name is ani whats ur name\n"
      },
      {
        "role": "assistant",
        "content": "Nice to meet you, Ani."
      },
      {
        "role": "user",
        "content": ""
      }
    ],
    temperature=1,
    max_completion_tokens=1024,
    top_p=1,
    stream=True,
    stop=None,
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")

"""
