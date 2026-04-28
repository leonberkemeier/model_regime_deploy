"""
Embedder
========
Handles chunking, upserting and querying the two ChromaDB collections.

Collection 1: sec_raw_chunks
    Raw text split into overlapping ~500-char windows.
    Embedded with all-MiniLM-L6-v2.
    Used for: arbitrary semantic drill-down at scoring time.

Collection 2: sec_llm_evaluations  (Phase 2 — wired up later)
    LLM-distilled JSON evaluation per section, embedded.
    Used for: standardised similarity retrieval across the universe.

Document ID scheme (guarantees idempotency via upsert):
    raw:  {ticker}__{form}__{filing_date}__{section}__{chunk_idx:04d}
    llm:  {ticker}__{form}__{filing_date}__{section}__eval
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Chunk size / overlap in characters (not tokens — fast, no tokeniser needed)
CHUNK_SIZE    = int(500)
CHUNK_OVERLAP = int(50)


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split `text` into overlapping windows of approximately `chunk_size` chars.
    Tries to break on sentence boundaries ('. ') for cleaner chunks.
    """
    chunks = []
    start  = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        # Try to end on a sentence boundary within the last 20% of the window
        if end < length:
            boundary_search_start = start + int(chunk_size * 0.80)
            boundary = text.rfind(". ", boundary_search_start, end)
            if boundary != -1:
                end = boundary + 2  # include the space after '.'

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap  # slide forward with overlap

    return chunks


# ── Collection 1: raw chunks ──────────────────────────────────────────────────

def embed_raw_chunks(
    collection,
    ticker: str,
    filing_date: str,
    form_type: str,
    sections: dict[str, str],
) -> int:
    """
    Chunk every section in `sections`, embed and upsert into `collection`.

    Returns the total number of chunks upserted.
    """
    total = 0

    for section_name, text in sections.items():
        if not text or not text.strip():
            continue

        chunks = chunk_text(text)
        if not chunks:
            continue

        doc_ids   = []
        documents = []
        metadatas = []

        for idx, chunk in enumerate(chunks):
            doc_id = f"{ticker}__{form_type}__{filing_date}__{section_name}__{idx:04d}"
            doc_ids.append(doc_id)
            documents.append(chunk)
            metadatas.append({
                "ticker":       ticker,
                "form_type":    form_type,
                "filing_date":  filing_date,
                "section":      section_name,
                "chunk_index":  idx,
                "chunk_total":  len(chunks),
            })

        try:
            collection.upsert(ids=doc_ids, documents=documents, metadatas=metadatas)
            total += len(chunks)
            logger.debug(
                f"  {ticker} | {form_type} | {filing_date} | {section_name} "
                f"→ {len(chunks)} chunks upserted"
            )
        except Exception as exc:
            logger.error(f"Failed to upsert chunks for {ticker}/{section_name}: {exc}")

    return total


# ── Collection 2: LLM evaluations (Phase 2) ───────────────────────────────────

def embed_llm_evaluation(
    collection,
    ticker: str,
    filing_date: str,
    form_type: str,
    section_name: str,
    llm_json: dict,
) -> bool:
    """
    Embed and upsert a single LLM-distilled evaluation for one filing section.

    `llm_json` should contain keys like:
        sentiment, key_themes, risk_level, growth_signal, notable_quote

    The document stored in ChromaDB is a human-readable string built from
    the structured output so the embedding captures semantic meaning.

    Returns True on success.
    """
    doc_id = f"{ticker}__{form_type}__{filing_date}__{section_name}__eval"

    # Build a readable text representation for the embedding
    themes = ", ".join(llm_json.get("key_themes", []))
    quote  = llm_json.get("notable_quote", "")
    text   = (
        f"Ticker: {ticker}. Section: {section_name}. "
        f"Sentiment: {llm_json.get('sentiment', 'N/A')}. "
        f"Key themes: {themes}. "
        f"Notable: {quote}"
    )

    metadata = {
        "ticker":        ticker,
        "form_type":     form_type,
        "filing_date":   filing_date,
        "section":       section_name,
        "sentiment":     str(llm_json.get("sentiment",     "N/A")),
        "risk_level":    float(llm_json.get("risk_level",  0.5)),
        "growth_signal": float(llm_json.get("growth_signal", 0.0)),
        "notable_quote": quote[:500],  # ChromaDB metadata values have a size limit
    }

    try:
        collection.upsert(ids=[doc_id], documents=[text], metadatas=[metadata])
        logger.debug(f"LLM eval upserted: {doc_id}")
        return True
    except Exception as exc:
        logger.error(f"Failed to upsert LLM eval {doc_id}: {exc}")
        return False


# ── Query helpers (used by MCP tools later) ───────────────────────────────────

def query_raw_chunks(
    collection,
    query_text: str,
    ticker: Optional[str] = None,
    n_results: int = 3,
) -> list[dict]:
    """
    Semantic search over raw chunks.

    If `ticker` is provided, restricts results to that company.
    Returns a list of dicts: {id, text, metadata, distance}.
    """
    where = {"ticker": ticker} if ticker else None

    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({"id": doc_id, "text": doc, "metadata": meta, "distance": dist})
        return hits

    except Exception as exc:
        logger.error(f"Raw chunk query failed: {exc}")
        return []


def query_llm_evaluations(
    collection,
    query_text: str,
    ticker: Optional[str] = None,
    n_results: int = 3,
) -> list[dict]:
    """
    Semantic search over LLM-distilled evaluations.

    If `ticker` is provided, restricts results to that company.
    Returns a list of dicts: {id, text, metadata, distance}.
    """
    where = {"ticker": ticker} if ticker else None

    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({"id": doc_id, "text": doc, "metadata": meta, "distance": dist})
        return hits

    except Exception as exc:
        logger.error(f"LLM evaluation query failed: {exc}")
        return []
