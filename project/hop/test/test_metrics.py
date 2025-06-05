from hop.metrics import lev_sim, score
from hop.config import EvalConfig


def test_lev_exact() -> None:
    assert lev_sim("ABC", "ABC") == 1


def test_score_alias() -> None:
    cfg = EvalConfig(use_api_for_embedding=False)
    assert score("GALNT4", "GALNT4", "sequence gene alias", cfg) == 1
