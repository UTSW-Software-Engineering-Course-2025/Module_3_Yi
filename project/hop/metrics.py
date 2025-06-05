import re, requests, torch, Levenshtein
import torch.nn.functional as F
from typing import List, Union
from transformers import AutoTokenizer, AutoModel
from .config import EvalConfig
from typing import Callable
from torch import Tensor

EmbedFn = Callable[[List[str], str], Tensor]


# -------- 文本 utils --------
def _to_list(x: Union[str, List[str]]) -> List[str]:
    if isinstance(x, list):
        return x
    return [s.strip() for s in re.split(r"[;,]\s*|\n", x) if s.strip()]


def lev_sim(a: str, b: str) -> float:
    d: int = Levenshtein.distance(a.lower(), b.lower())
    return 1 - d / max(len(a), len(b), 1)


# -------- 句向量（本地 / API） --------
_tokenizer, _model = None, None


def _embed_local(texts: List[str], api_url: str = "") -> torch.Tensor:
    global _tokenizer, _model
    if _tokenizer is None:
        m = "FremyCompany/BioLORD-2023"
        _tokenizer, _model = AutoTokenizer.from_pretrained(
            m
        ), AutoModel.from_pretrained(m)
        _model.eval()
    encoded = _tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = _model(**encoded)[0]
    mask = encoded["attention_mask"].unsqueeze(-1).float()
    vec = (out * mask).sum(1) / mask.sum(1)
    return F.normalize(vec, p=2, dim=1)


def _embed_api(texts: List[str], url: str) -> torch.Tensor:
    resp = requests.post(url, json={"sentences": texts}, timeout=30)
    resp.raise_for_status()
    arr = resp.json()["embeddings"]
    return F.normalize(torch.tensor(arr, dtype=torch.float32), p=2, dim=1)


# -------- 主评估 --------
def score(pred: Union[str, List[str]], gold: str, task: str, cfg: EvalConfig) -> float:
    task = task.lower()
    if "alias" in task or "location" in task:
        p, g = _to_list(pred), _to_list(gold)
        sims = [max(lev_sim(pi, gi) for gi in g) for pi in p]
        sims = [s for s in sims if s >= cfg.lev_thr]
        return sum(sims) / len(p) if p else 0

    if "function" in task:
        if cfg.use_api_for_embedding:
            emb_fn: EmbedFn = _embed_api
            api_arg = cfg.embedding_api_url
        else:
            emb_fn: EmbedFn = _embed_local
            api_arg = ""

        v = emb_fn([str(pred), str(gold)], api_arg)
        return float(F.cosine_similarity(v[0], v[1], dim=0))

    return lev_sim(str(pred), str(gold))
