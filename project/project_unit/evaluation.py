from collections import defaultdict
from typing import Callable
from .utils import normalize_text, to_norm_set


def exact_match(pred: str, true: str) -> float:
    return float(normalize_text(pred) == normalize_text(true))


def gene_disease_association(pred, true) -> float:
    true_set = to_norm_set(true)
    pred_set = to_norm_set(pred)
    return (len(true_set & pred_set) / len(true_set)) if true_set else 1.0


def disease_gene_location(pred, true) -> float:
    return gene_disease_association(pred, true)


def human_genome_dna_alignment(pred: str, true: str) -> float:
    pred_norm, true_norm = normalize_text(pred), normalize_text(true)
    if pred_norm == true_norm:
        return 1.0
    return 0.5 if pred_norm.split(":")[0] == true_norm.split(":")[0] else 0.0


# Mapping
metric_task_map = defaultdict(
    lambda: exact_match,
    {
        "gene disease association": gene_disease_association,
        "disease gene location": disease_gene_location,
        "human genome dna alignment": human_genome_dna_alignment,
    },
)


def score_answer(pred: str, true: str, task: str) -> float:
    return metric_task_map[task.strip().lower()](pred, true)
