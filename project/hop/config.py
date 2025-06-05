from dataclasses import dataclass
from typing import Optional


# ---------- 数据 ----------
@dataclass
class DataConfig:
    input_path: str = "data/genehop.json"
    dataset_format: str = "json"


# ---------- 模型 ----------
@dataclass
class ModelConfig:
    model_name: str = "gpt-4.1"
    model_backend: str = "azure"  # ["azure", "openai", "ollama"]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: Optional[int] = None
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None


# ---------- 评估 ----------
@dataclass
class EvalConfig:
    use_api_for_embedding: bool = True
    embedding_api_url: str = "http://198.215.61.34:8152/embed"
    lev_thr: float = 0.8
