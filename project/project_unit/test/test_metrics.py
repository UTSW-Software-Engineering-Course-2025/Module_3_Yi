import pytest
from project.project_unit.starter_geneturing.evaluation import (
    exact_match,
    human_genome_dna_alignment,
    gene_disease_association,
)


@pytest.mark.parametrize(
    "pred,true,score",
    [
        ("Gene", "gene", 1.0),
        ("A", "B", 0.0),
    ],
)
def test_exact_match(pred, true, score):
    assert exact_match(pred, true) == score


@pytest.mark.parametrize(
    "pred,true,score",
    [
        ("chr1:1-2", "chr1:1-2", 1.0),
        ("chr1:100", "chr1:200", 0.5),
        ("chr2:100", "chr1:200", 0.0),
    ],
)
def test_alignment(pred, true, score):
    assert human_genome_dna_alignment(pred, true) == score


def test_gene_disease_multi_hit():
    pred = "BRCA1, TP53"
    true = "TP53"
    assert gene_disease_association(pred, true) == 1.0  # 1 / 1 命中
