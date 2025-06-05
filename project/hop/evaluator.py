from dataclasses import dataclass, asdict
from typing import List, Dict, Union, Optional
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from .config import DataConfig, ModelConfig, EvalConfig
from .data_loader import load_genehop
from .model_client import query_llm, query_llm_with_tools
from .processing import clean_answer
from .metrics import score as score_fn
from typing import Union, Optional, List, Any



@dataclass
class Result:
    id: int
    task: str
    question: str
    gold: str
    raw_pred: Union[str, dict[str, Any], None]
    pred: Union[str, List[str], None]
    score: Optional[float]
    success: bool


def evaluate(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    eval_cfg: EvalConfig,
    system_prompt: str,
    examples: List[Dict[str, str]],
    use_tools: bool = False
) -> pd.DataFrame:
    df = load_genehop(data_cfg)
    llm_fn = query_llm_with_tools if use_tools else query_llm
    results: List[Result] = []
    task_scores: Dict[str, List[float]] = defaultdict(list)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="GeneHop"):
        try:
            raw = llm_fn(
                user_query=row["question"],
                system_prompt=system_prompt,
                examples=examples,
                cfg=model_cfg,
                pydantic_model=None,  # or GeneHopResult
            )
            pred = clean_answer(raw, row["task"])
            s = score_fn(pred, row["answer"], row["task"], eval_cfg)
            results.append(
                Result(
                    idx, row["task"], row["question"], row["answer"], raw, pred, s, True
                )
            )
            task_scores[row["task"]].append(s)
        except Exception as e:
            results.append(
                Result(
                    idx,
                    row["task"],
                    row["question"],
                    row["answer"],
                    None,
                    None,
                    None,
                    False,
                )
            )
            print(f"[Err] {idx}: {e}")

    res_df = pd.DataFrame(asdict(r) for r in results)
    res_df.to_csv("outputs/results.csv", index=False)
    print("Saved → outputs/results.csv")
    return res_df
