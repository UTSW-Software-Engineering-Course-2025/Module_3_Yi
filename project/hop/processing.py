import re
from typing import Union, List
from pydantic import BaseModel


def clean_answer(ans: Union[str, BaseModel], task: str) -> Union[str, List[str]]:
    text = ans.answer if isinstance(ans, BaseModel) else str(ans)
    text = text.strip()
    if task in ("sequence gene alias", "Disease gene location"):
        return [s.strip() for s in re.split(r"[;,]\s*|\n", text) if s.strip()]
    return str(text)
