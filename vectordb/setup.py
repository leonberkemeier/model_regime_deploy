"""
VectorDB Setup
==============
Initialises a persistent ChromaDB instance with two collections:

  sec_raw_chunks      — raw text chunks from SEC 10-K/10-Q filings
  sec_llm_evaluations — LLM-distilled structured evaluations per section (Phase 2)

Run directly to verify/create the DB:
    python vectordb/setup.py

When deploying on the server, add this as a cron one-shot or call it from
the ingestion task (it is fully idempotent).
"""

import os
import logging
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Config (override via .env) ────────────────────────────────────────────────
CHROMA_DB_PATH   = os.getenv("CHROMA_DB_PATH",   "./chroma_db")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL",  "all-MiniLM-L6-v2")

# Collection names — referenced by embedder and MCP tools, so keep stable
COLLECTION_RAW    = "sec_raw_chunks"
COLLECTION_LLM    = "sec_llm_evaluations"


def get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    """Return the shared SentenceTransformer embedding function."""
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def get_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client, creating the directory if needed."""
    Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


def get_collections(client: chromadb.PersistentClient = None):
    """
    Return (raw_chunks_collection, llm_evaluations_collection).
    Creates them if they don't exist yet (idempotent).
    """
    if client is None:
        client = get_client()

    ef = get_embedding_function()

    raw_chunks = client.get_or_create_collection(
        name=COLLECTION_RAW,
        embedding_function=ef,
        metadata={
            "description": "Raw text chunks from SEC 10-K/10-Q filings",
            "embedding_model": EMBEDDING_MODEL,
            "hnsw:space": "cosine",
        },
    )

    llm_evaluations = client.get_or_create_collection(
        name=COLLECTION_LLM,
        embedding_function=ef,
        metadata={
            "description": "LLM-distilled structured evaluations of SEC filing sections",
            "embedding_model": EMBEDDING_MODEL,
            "hnsw:space": "cosine",
        },
    )

    return raw_chunks, llm_evaluations


def setup_vectordb() -> chromadb.PersistentClient:
    """Initialise the DB and print a status summary. Safe to call repeatedly."""
    client = get_client()
    raw_chunks, llm_evaluations = get_collections(client)

    logger.info(f"ChromaDB initialised at: {CHROMA_DB_PATH}")
    logger.info(f"  '{COLLECTION_RAW}':    {raw_chunks.count()} documents")
    logger.info(f"  '{COLLECTION_LLM}': {llm_evaluations.count()} documents")

    return client


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    client = setup_vectordb()

    raw_chunks, llm_evaluations = get_collections(client)
    print(f"\n✅ ChromaDB ready at '{CHROMA_DB_PATH}'")
    print(f"   {COLLECTION_RAW}:    {raw_chunks.count()} docs")
    print(f"   {COLLECTION_LLM}: {llm_evaluations.count()} docs")
