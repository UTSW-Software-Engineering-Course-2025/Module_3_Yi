from typing import List, Dict, Optional, Type, Any
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel
from .config import ModelConfig


def query_llm(
    user_query: str,
    system_prompt: str,
    examples: List[Dict[str, str]],
    cfg: ModelConfig,
    *,
    pydantic_model: Optional[Type[BaseModel]] = None
) -> str | BaseModel:
    """统一封装 Azure / OpenAI / Ollama 调用，返回纯文本或 Pydantic 对象。"""
    messages = (
        [{"role": "system", "content": system_prompt}]
        + examples
        + [{"role": "user", "content": user_query}]
    )

    if cfg.model_backend == "azure":
        client = AzureOpenAI(
            api_key=cfg.openai_api_key,
            azure_endpoint=cfg.openai_base_url,
            api_version="2024-10-21",
        )
    elif cfg.model_backend == "openai":
        client = OpenAI(api_key=cfg.openai_api_key)
    else:
        raise ValueError("Unsupported backend")

    base_args = dict(
        model=cfg.model_name,
        messages=messages,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_tokens,
    )

    if pydantic_model:
        resp = client.beta.chat.completions.parse(
            **base_args, response_format=pydantic_model
        )
        return resp.choices[0].message.parsed

    resp = client.chat.completions.create(**base_args)
    return resp.choices[0].message.content.strip()
