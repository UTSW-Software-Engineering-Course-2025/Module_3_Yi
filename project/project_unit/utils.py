import re
from typing import List, Set, Union


def normalize_text(text: str) -> str:
    """Lower‑case & strip surrounding whitespace."""
    return text.strip().lower()


def to_norm_set(items: Union[List[str], str]) -> Set[str]:
    """Convert list / CSV string into a deduplicated lower‑case set."""
    if isinstance(items, list):
        tokens = items
    else:
        tokens = re.split(r"[,;]+", items)
    return {normalize_text(tok) for tok in tokens if tok.strip()}
