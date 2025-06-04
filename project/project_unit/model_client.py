from typing import List, Dict, Optional
import os, requests
from openai import OpenAI  # pip install openai>=1.3
from .config import ModelConfig


def query_model(prompt_messages: List[Dict[str, str]], cfg: ModelConfig) -> str:
    """Return raw model reply string."""
    if cfg.model_backend == "openai":
        client = OpenAI(
            api_key=cfg.openai_api_key or os.getenv("OPENAI_API_KEY"),
            base_url=cfg.openai_base_url,
        )
        resp = client.chat.completions.create(
            model=cfg.model_name,
            messages=prompt_messages,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
        )
        return resp.choices[0].message.content.strip()

    elif cfg.model_backend == "ollama":
        payload = {
            "model": cfg.model_variant or cfg.model_name,
            "prompt": "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in prompt_messages
            ),
            "stream": False,
            "options": {
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
                "num_predict": cfg.max_tokens,
            },
        }
        r = requests.post(
            "http://localhost:11434/api/generate", json=payload, timeout=120
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    else:
        raise ValueError(f"Unsupported backend: {cfg.model_backend}")
