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
from .metrics import score_based_llm
from typing import Union, Optional, List, Any
from matplotlib import pyplot as plt


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
    use_tools: bool = False,
    use_llm_judge: bool = False,
    save_data_name: str = "results.csv",
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
            if use_llm_judge:
                judge_result = score_based_llm(
                    question=row["question"],
                    candidate=pred,
                    gold=row["answer"],
                    rubric="Correctness and completeness of biomedical information.",
                    cfg=model_cfg,
                )
                s = float(judge_result["score"])
            else:
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
    res_df.to_csv(f"outputs/{save_data_name}.csv", index=False)
    print(f"Saved → outputs/{save_data_name}.csv")
    plt.figure(figsize=(10, 6))
    plt.bar(
        task_scores.keys(),
        [sum(scores) / len(scores) for scores in task_scores.values()],
    )
    plt.xlabel("Task")
    plt.ylabel("Average Score")
    plt.title("Average Score by Task")
    plt.savefig(f"outputs/{save_data_name}.png")
    print(f"Saved → outputs/{save_data_name}.png")
    return res_df
