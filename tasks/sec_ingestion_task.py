"""
SEC Ingestion Task
==================
Fetches SEC 10-K / 10-Q filings from EDGAR and ingests them into ChromaDB.

This script is the cron entry point. Run it directly for a full ingest,
or pass CLI flags for targeted testing:

    # Full run (reads tickers from financial_data.db)
    python tasks/sec_ingestion_task.py

    # Test with a handful of tickers
    python tasks/sec_ingestion_task.py --tickers AAPL,MSFT,NVDA

    # Limit the number of tickers processed (useful on first run)
    python tasks/sec_ingestion_task.py --limit 20

    # Only fetch 10-K filings, skip 10-Q
    python tasks/sec_ingestion_task.py --forms 10-K

Suggested cron (weekly, runs at 06:00 every Monday):
    0 6 * * 1 /path/to/venv/bin/python /path/to/tasks/sec_ingestion_task.py >> /var/log/sec_ingest.log 2>&1

The task is fully idempotent: re-running never creates duplicate records
because all ChromaDB writes use `upsert` with stable deterministic IDs.
"""

import argparse
import logging
import sqlite3
import sys
import os
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv(Path(project_root) / ".env")

from vectordb.setup   import get_client, get_collections
from vectordb.sec_fetcher import get_cik, get_recent_filings, download_filing_text, parse_sections
from vectordb.embedder    import embed_raw_chunks

# ── Config ────────────────────────────────────────────────────────────────────
FINANCIAL_DB_PATH = os.getenv("FINANCIAL_DB_PATH", str(Path(project_root) / "financial_data.db"))
DEFAULT_FORMS     = ["10-K", "10-Q"]
FILINGS_PER_TICKER = int(os.getenv("FILINGS_PER_TICKER", "4"))  # how many recent filings per ticker

logger = logging.getLogger(__name__)


# ── Ticker discovery ──────────────────────────────────────────────────────────

def get_tickers_from_db(db_path: str) -> list[str]:
    """Read the list of active tickers from financial_data.db."""
    try:
        conn   = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Filter to US-listed companies only — non-US tickers (e.g. 0700.HK, 000001.SS)
        # won't have SEC filings and will silently skip, but filtering here saves
        # thousands of unnecessary EDGAR requests on a 700-ticker universe.
        query = """
            SELECT DISTINCT ticker FROM dim_company
            WHERE ticker NOT LIKE '%.HK'
              AND ticker NOT LIKE '%.SS'
              AND ticker NOT LIKE '%.SZ'
              AND ticker NOT LIKE '%.T'
              AND ticker NOT LIKE '%.L'
              AND ticker NOT LIKE '%.PA'
              AND ticker NOT LIKE '%.DE'
              AND ticker NOT LIKE '%.AX'
              AND ticker NOT LIKE '%=%'
            ORDER BY ticker
        """
        cursor.execute(query)
        tickers = [row[0] for row in cursor.fetchall() if row[0]]
        conn.close()
        if tickers:
            logger.info(f"Found {len(tickers)} US tickers in dim_company")
            return tickers

        logger.warning("No tickers returned from dim_company.")
        return []

    except Exception as exc:
        logger.error(f"Failed to read tickers from DB ({db_path}): {exc}")
        return []


# ── Per-ticker pipeline ───────────────────────────────────────────────────────

def ingest_ticker(
    ticker: str,
    raw_collection,
    form_types: list[str],
    filings_count: int,
) -> dict:
    """
    Full ingest pipeline for one ticker.

    Returns a stats dict:
        { "filings_processed": int, "chunks_upserted": int, "skipped": bool }
    """
    stats = {"filings_processed": 0, "chunks_upserted": 0, "skipped": False}

    # 1. CIK lookup
    cik = get_cik(ticker)
    if not cik:
        logger.info(f"  [{ticker}] No CIK — skipping (likely non-US or ETF)")
        stats["skipped"] = True
        return stats

    # 2. Get recent filings
    filings = get_recent_filings(cik, form_types=form_types, count=filings_count)
    if not filings:
        logger.info(f"  [{ticker}] No {form_types} filings found — skipping")
        stats["skipped"] = True
        return stats

    logger.info(f"  [{ticker}] Found {len(filings)} filing(s) to process")

    for filing in filings:
        form        = filing["form"]
        acc_no      = filing["accession_number"]
        date        = filing["filing_date"]
        primary_doc = filing["primary_document"]

        logger.info(f"    → {form} filed {date}  ({acc_no})")

        # 3. Download filing text
        text = download_filing_text(cik, acc_no, primary_doc)
        if not text:
            logger.warning(f"    ✗ Download failed, skipping this filing")
            continue

        # 4. Parse sections
        sections = parse_sections(text)
        if not sections:
            logger.warning(f"    ✗ No sections parsed from {acc_no}, skipping")
            continue

        logger.info(f"    ✓ Parsed sections: {list(sections.keys())}")

        # 5. Embed + upsert raw chunks (Collection 1)
        n_chunks = embed_raw_chunks(
            collection=raw_collection,
            ticker=ticker,
            filing_date=date,
            form_type=form,
            sections=sections,
        )

        logger.info(f"    ✓ {n_chunks} chunks upserted")
        stats["filings_processed"] += 1
        stats["chunks_upserted"]   += n_chunks

    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def run_sec_ingestion_task(
    tickers: list[str] = None,
    limit: int = None,
    form_types: list[str] = None,
):
    """
    Orchestrate the full ingestion run.

    Args:
        tickers:    explicit list of tickers (overrides DB lookup)
        limit:      cap on number of tickers to process
        form_types: which form types to fetch (default: 10-K + 10-Q)
    """
    if form_types is None:
        form_types = DEFAULT_FORMS

    logger.info("=" * 60)
    logger.info("SEC Ingestion Task — Starting")
    logger.info(f"  Forms: {form_types}  |  Filings per ticker: {FILINGS_PER_TICKER}")
    logger.info("=" * 60)

    # ── 1. Resolve ticker list ──
    if tickers:
        ticker_list = [t.strip().upper() for t in tickers]
        logger.info(f"Using {len(ticker_list)} tickers from CLI argument")
    else:
        ticker_list = get_tickers_from_db(FINANCIAL_DB_PATH)
        if not ticker_list:
            logger.error("No tickers found. Exiting.")
            return

    if limit:
        ticker_list = ticker_list[:limit]
        logger.info(f"Limiting to first {limit} tickers")

    # ── 2. Initialise ChromaDB ──
    client = get_client()
    raw_collection, _ = get_collections(client)
    logger.info(f"ChromaDB ready. sec_raw_chunks has {raw_collection.count()} docs before run.")

    # ── 3. Process each ticker ──
    total_filings = 0
    total_chunks  = 0
    skipped       = 0
    errors        = 0

    for i, ticker in enumerate(ticker_list, start=1):
        logger.info(f"[{i}/{len(ticker_list)}] Processing {ticker} ...")
        try:
            stats = ingest_ticker(ticker, raw_collection, form_types, FILINGS_PER_TICKER)
            total_filings += stats["filings_processed"]
            total_chunks  += stats["chunks_upserted"]
            if stats["skipped"]:
                skipped += 1
        except Exception as exc:
            logger.error(f"  [{ticker}] Unexpected error: {exc}", exc_info=True)
            errors += 1

    # ── 4. Summary ──
    logger.info("=" * 60)
    logger.info("SEC Ingestion Task — Complete")
    logger.info(f"  Tickers processed : {len(ticker_list)}")
    logger.info(f"  Skipped (no CIK)  : {skipped}")
    logger.info(f"  Errors            : {errors}")
    logger.info(f"  Filings ingested  : {total_filings}")
    logger.info(f"  Chunks upserted   : {total_chunks}")
    logger.info(f"  Total docs in DB  : {raw_collection.count()}")
    logger.info("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Ingest SEC filings into ChromaDB")
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated list of tickers to process (e.g. AAPL,MSFT,NVDA). "
             "Defaults to all tickers in financial_data.db.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of tickers to process. Useful for testing.",
    )
    parser.add_argument(
        "--forms",
        type=str,
        default="10-K,10-Q",
        help="Comma-separated filing forms to fetch (default: 10-K,10-Q).",
    )
    args = parser.parse_args()

    tickers_arg    = args.tickers.split(",") if args.tickers else None
    form_types_arg = [f.strip() for f in args.forms.split(",")]

    run_sec_ingestion_task(
        tickers=tickers_arg,
        limit=args.limit,
        form_types=form_types_arg,
    )
