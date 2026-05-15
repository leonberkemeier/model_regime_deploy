"""
Trade Log
=========
Records every portfolio decision with a human-readable rationale.
Every sell, buy, swap, rebuild, or blocked trade is written here so
the system (and a human auditor) always knows exactly why a change
was made.

Schema overview:
  action        — SELL | BUY | REBUILD | BLOCKED
  trigger_type  — RISK_VIOLATION | CONVICTION_DROP | OPPORTUNITY_COST |
                  NARRATIVE_DECAY | FULL_REBUILD | CONFIDENCE_GATE
  rationale     — Plain English explanation, e.g.:
                    "NVDA sold: conviction collapsed from 0.71 → -0.42.
                     Replaced by MSFT (conviction delta: +0.38,
                     expected return delta: +1.9%)."
  swap_ticker   — For opportunity swaps: the paired BUY/SELL ticker
  executed      — 0 if blocked by the confidence gate (confidence < threshold)

All entries are written to regimes.db so they live alongside the
quantitative data that drove them.
"""

import sqlite3
import logging
import datetime
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_base = Path(__file__).parent.parent
REGIMES_DB = os.getenv("REGIMES_DB_PATH", str(_base / "regimes.db"))


# ── Schema ────────────────────────────────────────────────────────────────────

def init_trade_log_table(conn: sqlite3.Connection) -> None:
    """Create trade_log table if it doesn't exist. Safe to call repeatedly."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trade_log (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            log_date             TEXT    NOT NULL,
            profile_id           INTEGER,
            profile_name         TEXT,
            action               TEXT    NOT NULL,  -- SELL | BUY | REBUILD | BLOCKED
            ticker               TEXT    NOT NULL,
            trigger_type         TEXT,              -- what caused this decision
            -- Metrics at decision time
            conviction_score     REAL,
            es_95                REAL,
            var_95               REAL,
            mean_expected_return REAL,
            sentiment_score      REAL,
            confidence_score     REAL,              -- LLM confidence (0-1); NULL if rule-based
            -- Swap info (populated for OPPORTUNITY_COST pairs)
            swap_ticker          TEXT,              -- paired BUY ticker (on a SELL) or paired SELL ticker (on a BUY)
            swap_conviction      REAL,
            swap_return_delta    REAL,              -- mu_swap - mu_this (positive = swap is better)
            -- Decision output
            rationale            TEXT    NOT NULL,
            executed             INTEGER DEFAULT 1, -- 0 if blocked by confidence gate
            created_at           TEXT    DEFAULT (datetime('now'))
        )
    """)
    conn.commit()


# ── Write helpers ─────────────────────────────────────────────────────────────

def log_trade(
    conn: sqlite3.Connection,
    *,
    action: str,
    ticker: str,
    trigger_type: str,
    rationale: str,
    log_date: str = None,
    profile_id: int = None,
    profile_name: str = None,
    conviction_score: float = None,
    es_95: float = None,
    var_95: float = None,
    mean_expected_return: float = None,
    sentiment_score: float = None,
    confidence_score: float = None,
    swap_ticker: str = None,
    swap_conviction: float = None,
    swap_return_delta: float = None,
    executed: int = 1,
) -> int:
    """
    Write one trade log entry. Returns the new row id.

    Args:
        action:       'SELL', 'BUY', 'REBUILD', or 'BLOCKED'
        ticker:       The asset ticker this entry is about
        trigger_type: Why this decision was made
        rationale:    Human-readable explanation (required)
        executed:     0 if blocked by the confidence gate
    """
    log_date = log_date or datetime.date.today().isoformat()

    cursor = conn.execute("""
        INSERT INTO trade_log
        (log_date, profile_id, profile_name, action, ticker, trigger_type,
         conviction_score, es_95, var_95, mean_expected_return, sentiment_score,
         confidence_score, swap_ticker, swap_conviction, swap_return_delta,
         rationale, executed)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        log_date, profile_id, profile_name, action, ticker, trigger_type,
        conviction_score, es_95, var_95, mean_expected_return, sentiment_score,
        confidence_score, swap_ticker, swap_conviction, swap_return_delta,
        rationale, executed,
    ))
    conn.commit()
    logger.info(f"[TradeLog] {action:8s} {ticker:6s} | {trigger_type} | {rationale[:80]}")
    return cursor.lastrowid


def log_swap(
    conn: sqlite3.Connection,
    *,
    log_date: str,
    profile_id: int,
    profile_name: str,
    sell_ticker: str,
    buy_ticker: str,
    sell_conviction: float,
    buy_conviction: float,
    sell_es_95: float = None,
    buy_mean_return: float = None,
    sell_mean_return: float = None,
    confidence_score: float = None,
    executed: int = 1,
) -> None:
    """
    Convenience wrapper for an opportunity-cost swap (one SELL + one BUY).
    Both entries are cross-referenced via swap_ticker.
    """
    delta = (buy_conviction or 0) - (sell_conviction or 0)
    return_delta = None
    if buy_mean_return is not None and sell_mean_return is not None:
        return_delta = round((buy_mean_return - sell_mean_return) * 100, 2)  # in %

    sell_rationale = (
        f"{sell_ticker} sold due to opportunity cost: "
        f"{buy_ticker} offers higher conviction "
        f"({buy_conviction:+.2f} vs {sell_conviction:+.2f}, delta={delta:+.2f})"
        + (f", expected return advantage: {return_delta:+.1f}%" if return_delta else "")
        + "."
    )
    buy_rationale = (
        f"{buy_ticker} bought as replacement for {sell_ticker}. "
        f"Conviction advantage: {delta:+.2f}"
        + (f", return delta: {return_delta:+.1f}%" if return_delta else "")
        + "."
    )

    log_trade(
        conn,
        action="SELL",
        ticker=sell_ticker,
        trigger_type="OPPORTUNITY_COST",
        rationale=sell_rationale,
        log_date=log_date,
        profile_id=profile_id,
        profile_name=profile_name,
        conviction_score=sell_conviction,
        es_95=sell_es_95,
        confidence_score=confidence_score,
        swap_ticker=buy_ticker,
        swap_conviction=buy_conviction,
        swap_return_delta=(-return_delta if return_delta else None),
        executed=executed,
    )
    log_trade(
        conn,
        action="BUY",
        ticker=buy_ticker,
        trigger_type="OPPORTUNITY_COST",
        rationale=buy_rationale,
        log_date=log_date,
        profile_id=profile_id,
        profile_name=profile_name,
        conviction_score=buy_conviction,
        mean_expected_return=buy_mean_return,
        confidence_score=confidence_score,
        swap_ticker=sell_ticker,
        swap_conviction=sell_conviction,
        swap_return_delta=return_delta,
        executed=executed,
    )


def log_rebuild(
    conn: sqlite3.Connection,
    *,
    log_date: str,
    profile_id: int,
    profile_name: str,
    n_violations: int,
    violation_summary: str,
) -> None:
    """Log a full portfolio reconstruction event."""
    rationale = (
        f"Full portfolio rebuild for Profile {profile_id} ({profile_name}). "
        f"Triggered by {n_violations} violation(s): {violation_summary}."
    )
    log_trade(
        conn,
        action="REBUILD",
        ticker="PORTFOLIO",
        trigger_type="FULL_REBUILD",
        rationale=rationale,
        log_date=log_date,
        profile_id=profile_id,
        profile_name=profile_name,
    )


# ── Read helpers ──────────────────────────────────────────────────────────────

def get_trade_history(
    profile_id: int = None,
    days: int = 30,
    db_path: str = None,
) -> list[dict]:
    """
    Return trade log entries for the last `days` days.
    If profile_id is None, returns entries for all profiles.
    """
    db_path = db_path or REGIMES_DB
    cutoff  = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()

    try:
        conn   = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if profile_id is not None:
            cursor.execute("""
                SELECT * FROM trade_log
                WHERE log_date >= ? AND profile_id = ?
                ORDER BY log_date DESC, id DESC
            """, (cutoff, profile_id))
        else:
            cursor.execute("""
                SELECT * FROM trade_log
                WHERE log_date >= ?
                ORDER BY log_date DESC, id DESC
            """, (cutoff,))

        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    except Exception as exc:
        logger.error(f"get_trade_history: {exc}")
        return []


def get_last_decision_for_ticker(ticker: str, db_path: str = None) -> dict | None:
    """Return the most recent trade log entry for a specific ticker."""
    db_path = db_path or REGIMES_DB
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM trade_log
            WHERE ticker = ?
            ORDER BY log_date DESC, id DESC
            LIMIT 1
        """, (ticker,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as exc:
        logger.error(f"get_last_decision_for_ticker({ticker}): {exc}")
        return None
