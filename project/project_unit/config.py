"""Configuration dataclasses for data paths and model settings."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    """Dataset-related paths and parameters."""

    input_path: str = "data/geneturing.json"


@dataclass
class ModelConfig:
    """Large‑language‑model (LLM) settings."""

    model_name: str = "gpt-4"
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: Optional[int] = None
    batch_size: int = 1
    model_backend: str = "openai"  # openai | ollama
    openai_api_key: Optional[str] = None  # fallback: ENV var OPENAI_API_KEY
    openai_base_url: str = "https://api.openai.com/v1"
    model_variant: Optional[str] = None  # for Ollama local models
