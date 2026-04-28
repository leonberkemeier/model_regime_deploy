"""
LLM Evaluation Task (Collection 2)
====================================
Reads SEC filing sections from ChromaDB (sec_raw_chunks), evaluates each
section with the local Ollama model, and stores the structured output in
the sec_llm_evaluations collection.

This task runs on the AI PC (needs Ollama). The resulting evaluations are
consumed by the MCP search tool at scoring time.

Usage:
    # Evaluate all sections for all tickers that have raw chunks
    python tasks/llm_evaluation_task.py

    # Target specific tickers
    python tasks/llm_evaluation_task.py --tickers AAPL,MSFT,NVDA

    # Limit total sections processed (good for testing)
    python tasks/llm_evaluation_task.py --limit 10

    # Only process a specific section type
    python tasks/llm_evaluation_task.py --section mda

    # Use a different Ollama model
    python tasks/llm_evaluation_task.py --model gemma3:12b

Idempotent: already-evaluated sections are skipped automatically.
"""

import argparse
import json
import logging
import os
import sys
import requests
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv(Path(project_root) / ".env")

from vectordb.setup   import get_client, get_collections
from vectordb.embedder import embed_llm_evaluation

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  os.getenv("OLLAMA_URL", "http://localhost:11434"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))

logger = logging.getLogger(__name__)

# ── Section-aware prompt context ──────────────────────────────────────────────
# Each section type gets tailored guidance so the LLM focuses on what matters.

_SECTION_GUIDANCE = {
    "business_overview": (
        "Focus on: competitive moat, core revenue drivers, market position, "
        "and any notable strategic pivots or new business lines mentioned."
    ),
    "risk_factors": (
        "Focus on: the severity and novelty of risks, regulatory threats, "
        "competitive risks, and any risks that have materially worsened since prior filings."
    ),
    "mda": (
        "Focus on: revenue growth trends, margin trajectory, forward guidance, "
        "management's tone on near-term outlook, and any one-time items distorting results."
    ),
    "quantitative_risk": (
        "Focus on: interest rate sensitivity, currency exposure, commodity price risk, "
        "and whether the company's hedging posture is conservative or aggressive."
    ),
}

_SECTION_LABELS = {
    "business_overview": "Item 1 — Business Overview",
    "risk_factors":      "Item 1A — Risk Factors",
    "mda":               "Item 7 — Management Discussion & Analysis",
    "quantitative_risk": "Item 7A — Quantitative Risk Disclosures",
}


# ── Ollama interaction ────────────────────────────────────────────────────────

def _build_prompt(ticker: str, section_name: str, form_type: str, text: str) -> str:
    guidance = _SECTION_GUIDANCE.get(section_name, "Focus on the most financially material information.")
    label    = _SECTION_LABELS.get(section_name, section_name)

    return f"""You are a senior analyst at a quantitative hedge fund.
Below is the {label} section from a {form_type} SEC filing for {ticker}.

ANALYST FOCUS: {guidance}

--- BEGIN FILING SECTION ---
{text[:12000]}
--- END FILING SECTION ---

Evaluate this section and return ONLY a valid JSON object with exactly these fields:
{{
  "sentiment": "positive" or "neutral" or "negative",
  "key_themes": ["theme1", "theme2", "theme3"],
  "risk_level": <float 0.0 to 1.0>,
  "growth_signal": <float -1.0 to 1.0>,
  "notable_quote": "<single most important sentence verbatim from the text>"
}}

Rules:
- key_themes: exactly 3 short phrases (3-6 words each)
- risk_level: 0.0 = negligible risk, 1.0 = severe/existential risk
- growth_signal: -1.0 = strong decline, 0.0 = flat/neutral, 1.0 = strong growth
- notable_quote: copy one sentence verbatim, do not paraphrase
- No markdown, no explanation, only the JSON object."""


def call_ollama(prompt: str, model: str) -> dict:
    """
    Send a prompt to Ollama and parse the JSON response.
    Returns an empty dict on any failure.
    """
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "{}")
        return json.loads(raw)
    except requests.exceptions.RequestException as exc:
        logger.error(f"Ollama connection error: {exc}")
        return {}
    except json.JSONDecodeError as exc:
        logger.error(f"Ollama returned invalid JSON: {exc}")
        return {}


def _validate_and_clean(result: dict, ticker: str, section: str) -> dict | None:
    """
    Validate the LLM output has the required fields and correct types.
    Returns a cleaned dict, or None if the result is unusable.
    """
    required = ["sentiment", "key_themes", "risk_level", "growth_signal", "notable_quote"]
    if not all(k in result for k in required):
        missing = [k for k in required if k not in result]
        logger.warning(f"[{ticker}/{section}] LLM missing fields: {missing}")
        return None

    try:
        cleaned = {
            "sentiment":     str(result["sentiment"]).lower()
                             if result["sentiment"] in ("positive", "neutral", "negative")
                             else "neutral",
            "key_themes":    list(result["key_themes"])[:3],
            "risk_level":    max(0.0, min(1.0, float(result["risk_level"]))),
            "growth_signal": max(-1.0, min(1.0, float(result["growth_signal"]))),
            "notable_quote": str(result["notable_quote"])[:1000],
        }
        return cleaned
    except (TypeError, ValueError) as exc:
        logger.warning(f"[{ticker}/{section}] LLM type error: {exc}")
        return None


# ── ChromaDB helpers ──────────────────────────────────────────────────────────

def get_unique_filing_sections(raw_collection) -> list[tuple[str, str, str, str]]:
    """
    Return all unique (ticker, form_type, filing_date, section) tuples
    present in sec_raw_chunks.
    """
    results  = raw_collection.get(include=["metadatas"])
    seen     = set()
    sections = []

    for meta in results["metadatas"]:
        key = (
            meta["ticker"],
            meta["form_type"],
            meta["filing_date"],
            meta["section"],
        )
        if key not in seen:
            seen.add(key)
            sections.append(key)

    # Sort for deterministic processing order
    sections.sort()
    return sections


def reconstruct_section_text(
    raw_collection,
    ticker: str,
    form_type: str,
    filing_date: str,
    section_name: str,
) -> str:
    """
    Retrieve all chunks for a specific filing section and reassemble the text
    in original order (by chunk_index).
    """
    results = raw_collection.get(
        where={
            "$and": [
                {"ticker":      {"$eq": ticker}},
                {"form_type":   {"$eq": form_type}},
                {"filing_date": {"$eq": filing_date}},
                {"section":     {"$eq": section_name}},
            ]
        },
        include=["documents", "metadatas"],
    )

    if not results["ids"]:
        return ""

    # Sort chunks by their original index
    pairs = sorted(
        zip(results["documents"], results["metadatas"]),
        key=lambda x: x[1].get("chunk_index", 0),
    )

    return " ".join(doc for doc, _ in pairs)


def already_evaluated(llm_collection, ticker: str, form_type: str, filing_date: str, section: str) -> bool:
    """Check if an evaluation already exists in sec_llm_evaluations."""
    doc_id = f"{ticker}__{form_type}__{filing_date}__{section}__eval"
    result = llm_collection.get(ids=[doc_id])
    return len(result["ids"]) > 0


# ── Main orchestrator ─────────────────────────────────────────────────────────

def run_llm_evaluation_task(
    tickers: list[str] = None,
    limit: int = None,
    model: str = None,
    section_filter: str = None,
):
    """
    Evaluate all (or filtered) filing sections using the local Ollama model
    and store the structured results in sec_llm_evaluations.
    """
    model = model or OLLAMA_MODEL

    logger.info("=" * 60)
    logger.info(f"LLM Evaluation Task — Starting  (model: {model})")
    logger.info("=" * 60)

    # ── 1. Init collections ──
    client = get_client()
    raw_collection, llm_collection = get_collections(client)

    logger.info(f"sec_raw_chunks:      {raw_collection.count()} docs")
    logger.info(f"sec_llm_evaluations: {llm_collection.count()} docs (before run)")

    if raw_collection.count() == 0:
        logger.error("sec_raw_chunks is empty. Run tasks/sec_ingestion_task.py first.")
        return

    # ── 2. Discover all sections to evaluate ──
    all_sections = get_unique_filing_sections(raw_collection)
    logger.info(f"Found {len(all_sections)} unique filing sections in raw collection.")

    # Apply filters
    if tickers:
        ticker_set   = {t.upper() for t in tickers}
        all_sections = [s for s in all_sections if s[0] in ticker_set]
        logger.info(f"Filtered to {len(all_sections)} sections for tickers: {ticker_set}")

    if section_filter:
        all_sections = [s for s in all_sections if s[3] == section_filter]
        logger.info(f"Filtered to section type '{section_filter}': {len(all_sections)} sections")

    if limit:
        all_sections = all_sections[:limit]
        logger.info(f"Limiting to first {limit} sections")

    # ── 3. Process each section ──
    done = skipped = errors = 0

    for i, (ticker, form_type, filing_date, section_name) in enumerate(all_sections, start=1):
        label = f"{ticker} | {form_type} | {filing_date} | {section_name}"
        logger.info(f"[{i}/{len(all_sections)}] {label}")

        # Idempotency — skip if already evaluated
        if already_evaluated(llm_collection, ticker, form_type, filing_date, section_name):
            logger.info(f"  ↩ Already evaluated, skipping.")
            skipped += 1
            continue

        # Reconstruct section text from chunks
        text = reconstruct_section_text(raw_collection, ticker, form_type, filing_date, section_name)
        if not text.strip():
            logger.warning(f"  ✗ Empty text reconstructed — skipping.")
            errors += 1
            continue

        # Build prompt and call Ollama
        prompt = _build_prompt(ticker, section_name, form_type, text)
        raw_result = call_ollama(prompt, model)

        if not raw_result:
            logger.warning(f"  ✗ Ollama returned empty response — skipping.")
            errors += 1
            continue

        # Validate and clean the response
        cleaned = _validate_and_clean(raw_result, ticker, section_name)
        if not cleaned:
            logger.warning(f"  ✗ LLM output failed validation — skipping.")
            errors += 1
            continue

        # Store in Collection 2
        success = embed_llm_evaluation(
            collection=llm_collection,
            ticker=ticker,
            filing_date=filing_date,
            form_type=form_type,
            section_name=section_name,
            llm_json=cleaned,
        )

        if success:
            logger.info(
                f"  ✓ sentiment={cleaned['sentiment']} | "
                f"risk={cleaned['risk_level']:.2f} | "
                f"growth={cleaned['growth_signal']:+.2f}"
            )
            done += 1
        else:
            errors += 1

    # ── 4. Summary ──
    logger.info("=" * 60)
    logger.info("LLM Evaluation Task — Complete")
    logger.info(f"  Evaluated  : {done}")
    logger.info(f"  Skipped    : {skipped}")
    logger.info(f"  Errors     : {errors}")
    logger.info(f"  Total evals in DB: {llm_collection.count()}")
    logger.info("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Evaluate SEC filing sections with local Ollama LLM")
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated tickers to evaluate (e.g. AAPL,MSFT). Defaults to all in DB.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of sections to evaluate. Useful for testing.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Ollama model to use (default: {OLLAMA_MODEL} from .env).",
    )
    parser.add_argument(
        "--section",
        type=str,
        default=None,
        choices=["business_overview", "risk_factors", "mda", "quantitative_risk"],
        help="Only evaluate a specific section type.",
    )
    args = parser.parse_args()

    run_llm_evaluation_task(
        tickers=args.tickers.split(",") if args.tickers else None,
        limit=args.limit,
        model=args.model,
        section_filter=args.section,
    )
