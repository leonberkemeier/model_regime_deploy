"""
SEC EDGAR Fetcher
=================
Handles all communication with the SEC EDGAR public API:

  - CIK lookup by ticker symbol
  - Recent filing discovery (10-K / 10-Q)
  - Filing text download
  - Section parsing (Item 1, 1A, 7, 7A)

SEC rate limit: max 10 requests/second.
We enforce a 0.15 s delay between requests to stay well within limits.

EDGAR requires a User-Agent header in format: "Name Email"
Set EDGAR_USER_AGENT in .env, e.g.:
    EDGAR_USER_AGENT=MyFinanceApp admin@myapp.com
"""

import os
import re
import time
import logging
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
EDGAR_USER_AGENT = os.getenv("EDGAR_USER_AGENT", "FinancialDataSystem admin@localhost.com")
EDGAR_RATE_DELAY = float(os.getenv("EDGAR_RATE_DELAY", "0.15"))  # seconds between requests

HEADERS = {
    "User-Agent": EDGAR_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}

# Max characters extracted per section before being passed to the embedder.
# ~15 000 chars ≈ 3 000–4 000 tokens — safely within any local LLM context.
MAX_SECTION_CHARS = int(os.getenv("MAX_SECTION_CHARS", "15000"))

# In-process cache so we only hit the EDGAR ticker map once per run
_cik_cache: dict[str, str] = {}
_ticker_map: dict[str, str] = {}  # upper(ticker) -> zero-padded CIK


# ── CIK lookup ────────────────────────────────────────────────────────────────

def _load_ticker_map() -> None:
    """Download and cache the full SEC ticker → CIK map (called once)."""
    global _ticker_map
    if _ticker_map:
        return
    try:
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        for entry in resp.json().values():
            ticker = str(entry["ticker"]).upper()
            cik    = str(entry["cik_str"]).zfill(10)
            _ticker_map[ticker] = cik
        logger.info(f"Loaded {len(_ticker_map)} tickers from SEC EDGAR.")
    except Exception as exc:
        logger.error(f"Failed to load EDGAR ticker map: {exc}")


def get_cik(ticker: str) -> Optional[str]:
    """
    Return the zero-padded 10-digit CIK for a ticker, or None if not found.
    Results are cached for the duration of the process.
    """
    ticker_upper = ticker.upper()
    if ticker_upper in _cik_cache:
        return _cik_cache[ticker_upper]

    _load_ticker_map()
    cik = _ticker_map.get(ticker_upper)
    if cik:
        _cik_cache[ticker_upper] = cik
    else:
        logger.warning(f"No CIK found for ticker '{ticker}' — may be non-US or delisted.")
    return cik


# ── Filing discovery ──────────────────────────────────────────────────────────

def get_recent_filings(
    cik: str,
    form_types: list[str] = None,
    count: int = 4,
) -> list[dict]:
    """
    Return up to `count` recent filings of the requested form types.

    Each item:
        {
            "form":             "10-K",
            "accession_number": "0000320193-24-000123",
            "filing_date":      "2024-11-01",
            "primary_document": "aapl-20240928.htm",
        }
    """
    if form_types is None:
        form_types = ["10-K", "10-Q"]

    time.sleep(EDGAR_RATE_DELAY)
    try:
        resp = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        data     = resp.json()
        recent   = data.get("filings", {}).get("recent", {})

        forms       = recent.get("form",             [])
        acc_numbers = recent.get("accessionNumber",  [])
        dates       = recent.get("filingDate",       [])
        primary_doc = recent.get("primaryDocument",  [])

        results = []
        for form, acc, date, doc in zip(forms, acc_numbers, dates, primary_doc):
            if form in form_types:
                results.append({
                    "form":             form,
                    "accession_number": acc,
                    "filing_date":      date,
                    "primary_document": doc,
                })
            if len(results) >= count:
                break

        return results

    except Exception as exc:
        logger.error(f"Failed to fetch filings for CIK {cik}: {exc}")
        return []


# ── Filing text download ──────────────────────────────────────────────────────

def download_filing_text(
    cik: str,
    accession_number: str,
    primary_document: str,
) -> Optional[str]:
    """
    Download the primary document of a filing and return it as plain text.
    HTML tags and excess whitespace are stripped automatically.
    """
    acc_clean = accession_number.replace("-", "")
    cik_int   = int(cik)  # EDGAR URLs use unpadded int

    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_int}/{acc_clean}/{primary_document}"
    )

    time.sleep(EDGAR_RATE_DELAY)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=45)
        resp.raise_for_status()
        text = resp.text

        # Strip HTML if present
        if "<html" in text.lower() or "<HTML" in text:
            text = re.sub(r"<[^>]+>", " ", text)            # remove tags
            text = re.sub(r"&[a-zA-Z#0-9]+;", " ", text)   # html entities
            text = re.sub(r"\s{2,}", " ", text)             # collapse whitespace

        return text.strip()

    except Exception as exc:
        logger.error(f"Failed to download filing {accession_number} ({url}): {exc}")
        return None


# ── Section parsing ───────────────────────────────────────────────────────────

# Maps the Item number (from 10-K) to a stable internal key
_SECTION_MAP = {
    "1":  "business_overview",   # Item 1  — Business
    "1a": "risk_factors",        # Item 1A — Risk Factors
    "7":  "mda",                 # Item 7  — MD&A
    "7a": "quantitative_risk",   # Item 7A — Quantitative / Qualitative Disclosures
}


def parse_sections(text: str) -> dict[str, str]:
    """
    Extract the four key sections from a 10-K/10-Q text.

    Returns a dict:
        {
            "business_overview": "...",
            "risk_factors":      "...",
            "mda":               "...",
            "quantitative_risk": "...",
        }

    Sections that cannot be found are simply omitted.
    Each section is capped at MAX_SECTION_CHARS characters.
    """
    text_lower = text.lower()

    # Find positions of all "item X" markers in document order
    markers: list[tuple[int, str]] = []
    for m in re.finditer(r"\bitem\s+(\d+[a-z]?)\b", text_lower):
        markers.append((m.start(), m.group(1).lower()))

    if not markers:
        logger.warning("No 'Item X' markers found in filing text — cannot parse sections.")
        return {}

    sections: dict[str, str] = {}

    for i, (start_pos, item_num) in enumerate(markers):
        section_key = _SECTION_MAP.get(item_num)
        if section_key is None:
            continue

        # Text runs until the next marker (or end of document)
        end_pos = markers[i + 1][0] if i + 1 < len(markers) else len(text)
        snippet = text[start_pos:end_pos].strip()

        # Keep only up to MAX_SECTION_CHARS; if there are multiple occurrences
        # of the same item header, retain the longest (most likely the real body)
        snippet = snippet[:MAX_SECTION_CHARS]
        if section_key not in sections or len(snippet) > len(sections[section_key]):
            sections[section_key] = snippet

    found = list(sections.keys())
    logger.debug(f"Parsed sections: {found}")
    return sections
