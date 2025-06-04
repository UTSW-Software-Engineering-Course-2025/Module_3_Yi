import json
import pandas as pd
from typing import Dict, List, Any
from .config import DataConfig


def load_and_flatten_gene_turing_data(cfg: DataConfig) -> pd.DataFrame:
    with open(cfg.input_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Dict[str, str]] = json.load(f)

    rows: List[Dict[str, Any]] = []
    for task, qa in raw.items():
        for q, a in qa.items():
            rows.append({"task": task, "question": q, "answer": a})
    return pd.DataFrame(rows)[["task", "question", "answer"]]
