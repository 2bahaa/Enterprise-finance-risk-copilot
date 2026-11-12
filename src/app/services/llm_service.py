import os

import httpx
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 400,
) -> str:
    """Send one chat request to Ollama and return the answer text."""

    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "repeat_penalty": 1.15,
                    "num_predict": max_tokens,
                },
            },
            timeout=180.0,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()

    except httpx.RequestError as exc:
        raise RuntimeError(
            "Cannot connect to Ollama. Check that Ollama is running."
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Ollama returned HTTP {exc.response.status_code}."
        ) from exc
    except (KeyError, ValueError) as exc:
        raise RuntimeError("Ollama returned an invalid response.") from exc