# hop/tools/definitions.py
from __future__ import annotations
from typing import Dict, List, Collection
from .ncbi import search_gene_id, summarize_gene_details
from .blast import run_blast_job


def available_functions():
    return {
        "search_gene_id": search_gene_id,
        "summarize_gene_details": summarize_gene_details,
        "run_blast_job": run_blast_job,
    }


def get_tools_definition() -> List[Dict[str, Collection[str]]]:
    """OpenAI / Azure / Anthropic function-calling schema list."""
    return [
        {
            "type": "function",
            "function": {
                "name": "search_gene_id",
                "description": "Return NCBI gene UID for a given gene symbol / Ensembl ID etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_id": {"type": "string", "description": "Gene identifier"}
                    },
                    "required": ["query_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "summarize_gene_details",
                "description": "Get structured gene summary by UID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "uid": {"type": "string", "description": "Gene UID"}
                    },
                    "required": ["uid"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_blast_job",
                "description": "Submit a nucleotide sequence to BLAST NT and return XML result.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "seq": {"type": "string", "description": "DNA/RNA sequence"}
                    },
                    "required": ["seq"],
                },
            },
        },
    ]
