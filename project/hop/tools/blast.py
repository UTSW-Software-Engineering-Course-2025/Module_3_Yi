# hop/tools/blast.py
from __future__ import annotations
import time, requests, re

BLAST_URL = "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi"


def _submit(seq: str, program: str = "blastn", db: str = "nt") -> tuple[str, int]:
    resp = requests.post(
        BLAST_URL,
        data={"CMD": "Put", "PROGRAM": program, "DATABASE": db, "QUERY": seq},
        timeout=60,
    )
    resp.raise_for_status()
    rid = re.search(r"RID = (\S+)", resp.text)
    rtoe = re.search(r"RTOE = (\d+)", resp.text)
    if not rid:
        raise RuntimeError("RID not found in BLAST response")
    return rid.group(1), int(rtoe.group(1)) if rtoe else 15


def _wait(rid: str, poll: int = 10) -> None:
    while True:
        resp = requests.get(
            BLAST_URL,
            params={"CMD": "Get", "RID": rid, "FORMAT_OBJECT": "SearchInfo"},
            timeout=30,
        )
        resp.raise_for_status()
        txt = resp.text
        if "Status=WAITING" in txt:
            time.sleep(poll)
            continue
        if "Status=FAILED" in txt:
            raise RuntimeError("BLAST job failed")
        if "Status=UNKNOWN" in txt:
            raise RuntimeError("BLAST RID expired/unknown")
        break


def _fetch(rid: str, fmt: str = "XML") -> str:
    resp = requests.get(
        BLAST_URL, params={"CMD": "Get", "RID": rid, "FORMAT_TYPE": fmt}, timeout=60
    )
    resp.raise_for_status()
    return resp.text


def run_blast_job(seq: str) -> str:
    """One-shot BLAST (submit-wait-fetch). Returns XML string."""
    rid, rtoe = _submit(seq)
    time.sleep(rtoe)
    _wait(rid)
    return _fetch(rid, fmt="XML")
