from typing import List, Dict, Optional, Type, Any, Union  
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel
from .config import ModelConfig
from .tools import get_tools_definition, available_functions
import json

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

def query_llm(
    user_query: str,
    system_prompt: str,
    examples: List[Dict[str, str]],
    cfg: ModelConfig,
    *,
    pydantic_model: Optional[Type[BaseModel]] = None,
) -> str | BaseModel:
    ...
# ---------------------------------------------- #


# ============ 新增：带 Tools 的版本 ============ #
def query_llm_with_tools(
    user_query: str,
    system_prompt: str,
    examples: List[Dict[str, str]],
    cfg: ModelConfig,
    *,
    pydantic_model: Optional[Type[BaseModel]] = None,
) -> str | BaseModel:
    """与 query_llm 输入/输出完全一致，但自动处理 tool_calls。"""
    # 1) 组装消息
    messages: List[Dict[str, Union[str, None, list]]] = (
        [{"role": "system", "content": system_prompt}]
        + examples
        + [{"role": "user", "content": user_query}]
    )

    tools_schema = get_tools_definition()
    fn_map = available_functions()

    # 2) 选择后端
    client = (
        AzureOpenAI(
            api_key=cfg.openai_api_key,
            azure_endpoint=cfg.openai_base_url,
            api_version="2024-10-21",
        )
        if cfg.model_backend == "azure"
        else OpenAI(api_key=cfg.openai_api_key)
    )

    base = dict(
        model=cfg.model_name,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_tokens,
    )

    while True:
        # 3) 调用 LLM（带 tools）
        if pydantic_model:
            resp = client.beta.chat.completions.parse(
                messages=messages,
                response_format=pydantic_model,
                tools=tools_schema,
                **base,
            )
            msg = resp.choices[0].message
        else:
            resp = client.chat.completions.create(
                messages=messages,
                tools=tools_schema,
                tool_choice="auto",
                **base,
            )
            msg = resp.choices[0].message  # type: ignore[attr-defined]

        # 4) 若有 tool_calls → 执行本地函数并继续
        if msg.tool_calls:                                   # type: List[ChatCompletionMessageToolCall]
            # 先把 assistant 的 tool_call 回显添加到 messages
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.function.name,
                                "arguments": c.function.arguments,
                            },
                        }
                        for c in msg.tool_calls
                    ],
                }
            )

            # 针对每个调用执行本地函数并追加 tool 消息
            for call in msg.tool_calls:
                fn_name: str = call.function.name
                arguments: dict = json.loads(call.function.arguments or "{}")

                try:
                    result = fn_map[fn_name](**arguments)
                except Exception as e:
                    result = {"error": str(e)}

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": fn_name,
                        "content": result,          # 直接是 str 或 dict 均可
                    }
                )

            continue 
        # 5) 没有 tool 调用，直接返回
        if pydantic_model:
            return msg.parsed  # type: ignore[attr-defined]
        return msg.content.strip() if msg.content else ""