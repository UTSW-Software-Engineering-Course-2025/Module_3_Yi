"""Pipeline entry point: load data → run LLM → score → save."""
import pandas as pd
from tqdm import tqdm
from typing import List
from .config import DataConfig, ModelConfig
from .data_loader import load_and_flatten_gene_turing_data
from .model_client import query_model
from .evaluation import score_answer
from .utils import normalize_text

# after computing  task_avg  and  overall_avg

import mlflow, os

mlflow.set_experiment("GeneTuring")
with mlflow.start_run(run_name="v1-openai"):
    # --- params (可选) ---
    mlflow.log_param("model_name", model_config.model_name)
    mlflow.log_param("backend", model_config.model_backend)

    # --- metrics ---
    mlflow.log_metric("overall_score", overall_avg)
    for task, avg in task_avg.items():
        mlflow.log_metric(f"{task.replace(' ', '_')}_score", avg)

    # --- artifacts ---
    mlflow.log_artifact(
        "outputs/gene_turing_scores_by_task.png", artifact_path="figures"
    )
    mlflow.log_artifact("outputs/gene_turing_results.csv", artifact_path="predictions")

# ---- simple Result dataclass ----
from dataclasses import dataclass


@dataclass
class Result:
    id: int
    task: str
    question: str
    answer: str
    raw_prediction: str | None
    processed_prediction: str | None
    score: float | None
    success: bool


SYSTEM_PROMPT = (
    "You are GeneGPT, an expert genomic assistant."
    " Answer only with the final result (or 'NA')."
)

FEW_SHOTS: List[Dict[str, str]] = [
    {"role": "user", "content": "What is the official gene symbol of LMP10?"},
    {"role": "assistant", "content": "Answer: PSMB10"},
    {
        "role": "user",
        "content": "Which chromosome does SNP rs1430464868 locate on human genome?",
    },
    {"role": "assistant", "content": "Answer: chr13"},
]


def get_answer(raw_reply: str, _task: str) -> str:
    """Extract portion after 'Answer:' if present."""
    low = raw_reply.lower()
    if "answer:" in low:
        return raw_reply.split("Answer:")[-1].strip()
    return raw_reply.strip()


def evaluate_pipeline():
    data_cfg = DataConfig()
    model_cfg = ModelConfig()

    df = load_and_flatten_gene_turing_data(data_cfg)
    results: List[Result] = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        try:
            messages = (
                [{"role": "system", "content": SYSTEM_PROMPT}]
                + FEW_SHOTS
                + [{"role": "user", "content": row.question}]
            )
            raw_pred = query_model(messages, model_cfg)
            proc_pred = get_answer(raw_pred, row.task)
            sc = score_answer(proc_pred, row.answer, row.task)
            results.append(
                Result(
                    idx,
                    row.task,
                    row.question,
                    row.answer,
                    raw_pred,
                    proc_pred,
                    sc,
                    True,
                )
            )
        except Exception as e:
            results.append(
                Result(idx, row.task, row.question, row.answer, None, None, None, False)
            )
            print(f"[Error {idx}] {e}")

    df_res = pd.DataFrame([r.__dict__ for r in results])
    df_res.to_csv("outputs/results.csv", index=False)
    print("Saved results to outputs/results.csv")

    # summary
    success_rate = df_res.success.mean()
    overall = df_res.loc[df_res.success, "score"].mean()
    print(f"Success rate: {success_rate:.2%}, Overall avg score: {overall:.3f}")


if __name__ == "__main__":
    evaluate_pipeline()
