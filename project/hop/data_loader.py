import json
import pandas as pd
from typing import Dict, Any, List
from .config import DataConfig


def load_genehop(config: DataConfig) -> pd.DataFrame:
    """Flatten genehop.json → DataFrame(task, question, answer)."""
    with open(config.input_path) as f:
        raw: Dict[str, Dict[str, str]] = json.load(f)

    rows: List[Dict[str, Any]] = [
        {"task": task, "question": q, "answer": a}
        for task, qa in raw.items()
        for q, a in qa.items()
    ]
    return pd.DataFrame(rows)
