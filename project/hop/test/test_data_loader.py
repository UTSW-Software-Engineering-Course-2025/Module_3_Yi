import json, pandas as pd, tempfile, os
from hop.data_loader import load_genehop
from hop.config import DataConfig
from pathlib import Path


def test_load_genehop(tmp_path: Path) -> None:
    # ── 构造迷你假数据 ──────────────────
    fake_json = {
        "sequence gene alias": {"Q1": "A1, A2"},
        "Disease gene location": {"Q2": "17q21.31"},
    }
    tmp_file = tmp_path / "mini.json"
    tmp_file.write_text(json.dumps(fake_json))

    # ── 调用函数 ───────────────────────
    df = load_genehop(DataConfig(input_path=str(tmp_file)))

    # ── 断言 ───────────────────────────
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"task", "question", "answer"}
    assert len(df) == 2
    assert (df["task"] == "sequence gene alias").sum() == 1
