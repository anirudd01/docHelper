import requests
import json


def generate_llm_answer(prompt: str, model_name: str = "llama3.2") -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
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
