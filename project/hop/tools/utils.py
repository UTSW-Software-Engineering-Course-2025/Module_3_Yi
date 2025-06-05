# hop/tools/utils.py
from __future__ import annotations

import json, random, time, logging
from typing import Any, Dict, Tuple

import requests


def call_api(
    url: str,
    params: dict | None = None,
    timeout: int = 60,
    max_retries: int = 3,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Robust HTTP GET with retries & back-off.
    Returns (success, json/err-dict)
    """
    base_delay = 1
    for attempt in range(max_retries):
        try:
            if attempt:
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                logging.warning(
                    f"API attempt {attempt+1}/{max_retries}, sleeping {delay:.1f}s"
                )
                time.sleep(delay)

            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return True, resp.json()  # type: ignore[arg-type]

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            continue  # retry
        except requests.HTTPError as e:
            if e.response and e.response.status_code in (429, 500, 502, 503, 504):
                continue
            return False, {"error": str(e)}
        except json.JSONDecodeError as e:
            return False, {"error": f"JSON decode error: {e}"}

    return False, {"error": f"Failed after {max_retries} retries"}
