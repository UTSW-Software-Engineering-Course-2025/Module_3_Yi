# hop/tools/ncbi.py
from __future__ import annotations
import json, os
from .utils import call_api

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
API_KEY = "4877bf94b505d53e71f9abd0c1cba2c1d609"  # 记得在 .env 或 shell 中 export

# ---------- 1. 查 UID ---------- #
def search_gene_id(query_id: str) -> str:
    """
    >>> search_gene_id("LMP10")   -> '{"uid": "5699"}'
    """
    url = f"{NCBI_BASE}esearch.fcgi?db=gene&term={query_id}&retmode=json"
    ok, data = call_api(url, params={"api_key": API_KEY})
    if not ok:
        return json.dumps(data)

    ids = data.get("esearchresult", {}).get("idlist", [])
    if ids:
        return json.dumps({"uid": ids[0]})
    return json.dumps({"error": f"UID not found for {query_id}"})


# ---------- 2. Gene 摘要 ---------- #
def summarize_gene_details(uid: str) -> str:
    """
    >>> summarize_gene_details("5699") -> '{"official_symbol": "..."}'
    """
    url = f"{NCBI_BASE}esummary.fcgi?db=gene&id={uid}&retmode=json"
    ok, data = call_api(url, params={"api_key": API_KEY})
    if not ok:
        return json.dumps(data)

    summ = data.get("result", {}).get(uid)
    if not summ:
        return json.dumps({"error": f"No summary for UID {uid}", "raw": data})

    parsed = {
        "uid": summ.get("uid"),
        "official_symbol": summ.get("nomenclaturesymbol"),
        "official_full_name": summ.get("nomenclaturename"),
        "description": summ.get("description"),
        "organism": summ.get("organism", {}).get("scientificname"),
        "summary_text": summ.get("summary"),
    }
    return json.dumps(parsed)
