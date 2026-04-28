"""
Thesis-Based Realignment Task
==============================
Runs AFTER the daily LLM conviction task. Compares today's metrics against
the snapshot taken when portfolios were last built, and triggers a full
portfolio rebuild if the original investment thesis has materially decayed.

Three violation types are checked per ticker in the current portfolio:

  1. RISK VIOLATION   — ES_95 (Expected Shortfall) has worsened by more than
                        RISK_DECAY_THRESHOLD vs the snapshot. Signals that
                        tail-risk has spiked beyond the original risk envelope.

  2. CONVICTION DROP  — LLM conviction score fell below CONVICTION_FLOOR.
                        Signals that the fundamental/narrative thesis has collapsed.

  3. OPPORTUNITY COST — A ticker NOT in the portfolio has a conviction score
                        that is OPPORTUNITY_GAP higher than the held portfolio's
                        average. Signals a better alternative has emerged.

If the fraction of violated tickers crosses REBUILD_TRIGGER_RATIO, the
Greenfield portfolio task is re-run to construct fresh model portfolios.
"""

import logging
import sqlite3
import datetime
import sys
import os
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv(Path(project_root) / ".env")

from connectors.webserver_client import WebserverClient
from config.settings import WEBSERVER_URL

logger = logging.getLogger(__name__)

# ── Violation thresholds (tune via .env) ─────────────────────────────────────
RISK_DECAY_THRESHOLD  = float(os.getenv("REALIGN_RISK_DECAY",     "0.50"))  # ES worsened 50%+
CONVICTION_FLOOR      = float(os.getenv("REALIGN_CONVICTION_FLOOR", "-0.30")) # score < -0.30
OPPORTUNITY_GAP       = float(os.getenv("REALIGN_OPPORTUNITY_GAP",  "0.30"))  # challenger is 0.30+ higher
REBUILD_TRIGGER_RATIO = float(os.getenv("REALIGN_REBUILD_RATIO",    "0.30"))  # 30%+ of held tickers violated


# ── Schema helpers ────────────────────────────────────────────────────────────

def init_snapshot_tables(conn: sqlite3.Connection) -> None:
    """Create the two tables needed for realignment tracking."""
    cursor = conn.cursor()

    # Stores the LLM+risk snapshot from the most recent portfolio build
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conviction_snapshots (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date    TEXT,
            profile_id       INTEGER,
            ticker           TEXT,
            conviction_score REAL,
            es_95            REAL,
            var_95           REAL,
            mean_return      REAL,
            UNIQUE(snapshot_date, profile_id, ticker)
        )
    """)

    # Stores the list of tickers per profile from the most recent greenfield build
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_portfolio_positions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            build_date   TEXT,
            profile_id   INTEGER,
            profile_name TEXT,
            ticker       TEXT,
            weight       REAL,
            UNIQUE(build_date, profile_id, ticker)
        )
    """)

    conn.commit()


# ── Snapshot saving (called from greenfield_portfolio_task) ───────────────────

def save_conviction_snapshot(
    conn: sqlite3.Connection,
    profile_id: int,
    positions: list[dict],
    today_str: str,
) -> None:
    """
    Save today's LLM scores + risk metrics for each ticker in a portfolio.
    Called by greenfield_portfolio_task after a successful build.
    """
    cursor = conn.cursor()

    for pos in positions:
        ticker = pos.get("ticker")
        if not ticker or ticker == "CASH":
            continue

        # Fetch today's conviction + risk from regimes.db
        cursor.execute("""
            SELECT conviction_score, var_95, mean_expected_return
            FROM daily_llm_conviction
            WHERE ticker = ? AND date = ?
        """, (ticker, today_str))
        llm_row = cursor.fetchone()

        cursor.execute("""
            SELECT AVG(es_95) FROM daily_monte_carlo
            WHERE ticker = ? AND date = ? AND horizon_days = 20
        """, (ticker, today_str))
        mc_row = cursor.fetchone()

        cursor.execute("""
            INSERT OR REPLACE INTO conviction_snapshots
            (snapshot_date, profile_id, ticker, conviction_score, es_95, var_95, mean_return)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            today_str,
            profile_id,
            ticker,
            llm_row[0] if llm_row else None,
            mc_row[0]  if mc_row  else None,
            llm_row[1] if llm_row else None,
            llm_row[2] if llm_row else None,
        ))

    conn.commit()


def save_portfolio_positions(
    conn: sqlite3.Connection,
    profile_id: int,
    profile_name: str,
    positions: list[dict],
    today_str: str,
) -> None:
    """Save portfolio positions to the local DB for realignment tracking."""
    cursor = conn.cursor()
    for pos in positions:
        ticker = pos.get("ticker", "CASH")
        weight = pos.get("weight", 0.0)
        cursor.execute("""
            INSERT OR REPLACE INTO model_portfolio_positions
            (build_date, profile_id, profile_name, ticker, weight)
            VALUES (?, ?, ?, ?, ?)
        """, (today_str, profile_id, profile_name, ticker, weight))
    conn.commit()


# ── Violation detection ───────────────────────────────────────────────────────

def _get_latest_snapshot_date(cursor: sqlite3.Cursor) -> str | None:
    cursor.execute("SELECT MAX(snapshot_date) FROM conviction_snapshots")
    row = cursor.fetchone()
    return row[0] if row and row[0] else None


def _get_portfolio_tickers(cursor: sqlite3.Cursor, snapshot_date: str) -> dict[int, set[str]]:
    """Return {profile_id: {ticker, ...}} for the given snapshot date."""
    cursor.execute("""
        SELECT profile_id, ticker FROM model_portfolio_positions
        WHERE build_date = ? AND ticker != 'CASH'
    """, (snapshot_date,))
    portfolios: dict[int, set[str]] = {}
    for profile_id, ticker in cursor.fetchall():
        portfolios.setdefault(profile_id, set()).add(ticker)
    return portfolios


def check_violations(
    cursor: sqlite3.Cursor,
    snapshot_date: str,
    today_str: str,
    profile_id: int,
    held_tickers: set[str],
) -> list[dict]:
    """
    Return a list of violation dicts for tickers in this profile's portfolio.
    Each dict: { ticker, violation_type, detail }
    """
    violations = []

    if not held_tickers:
        return violations

    # Today's LLM scores
    ticker_list = ",".join(f"'{t}'" for t in held_tickers)
    cursor.execute(f"""
        SELECT ticker, conviction_score FROM daily_llm_conviction
        WHERE date = ? AND ticker IN ({ticker_list})
    """, (today_str,))
    today_conviction = {row[0]: row[1] for row in cursor.fetchall()}

    # Today's risk metrics
    cursor.execute(f"""
        SELECT ticker, AVG(es_95) FROM daily_monte_carlo
        WHERE date = ? AND horizon_days = 20 AND ticker IN ({ticker_list})
        GROUP BY ticker
    """, (today_str,))
    today_es = {row[0]: row[1] for row in cursor.fetchall()}

    # Snapshot metrics
    cursor.execute(f"""
        SELECT ticker, conviction_score, es_95 FROM conviction_snapshots
        WHERE snapshot_date = ? AND profile_id = ? AND ticker IN ({ticker_list})
    """, (snapshot_date, profile_id))
    snapshot = {row[0]: {"conviction": row[1], "es_95": row[2]} for row in cursor.fetchall()}

    for ticker in held_tickers:
        snap = snapshot.get(ticker, {})

        # 1. Risk violation: ES_95 worsened beyond threshold
        snap_es     = snap.get("es_95")
        current_es  = today_es.get(ticker)
        if snap_es is not None and current_es is not None and snap_es != 0:
            # ES_95 is negative; worsening means it gets more negative
            decay = (current_es - snap_es) / abs(snap_es)
            if decay < -RISK_DECAY_THRESHOLD:
                violations.append({
                    "ticker":         ticker,
                    "violation_type": "RISK",
                    "detail":         f"ES_95 decayed {decay:.1%} vs snapshot (threshold: -{RISK_DECAY_THRESHOLD:.0%})",
                })

        # 2. Conviction drop below floor
        current_conviction = today_conviction.get(ticker)
        if current_conviction is not None and current_conviction < CONVICTION_FLOOR:
            violations.append({
                "ticker":         ticker,
                "violation_type": "CONVICTION_DROP",
                "detail":         f"Score={current_conviction:.2f} below floor {CONVICTION_FLOOR}",
            })

    # 3. Opportunity cost: find non-held tickers with significantly higher conviction
    avg_held_conviction = (
        sum(today_conviction.get(t, 0) for t in held_tickers) / len(held_tickers)
        if held_tickers else 0
    )

    cursor.execute("""
        SELECT ticker, conviction_score FROM daily_llm_conviction
        WHERE date = ? AND conviction_score > ?
        ORDER BY conviction_score DESC LIMIT 5
    """, (today_str, avg_held_conviction + OPPORTUNITY_GAP))

    for challenger_ticker, challenger_score in cursor.fetchall():
        if challenger_ticker not in held_tickers:
            violations.append({
                "ticker":         challenger_ticker,
                "violation_type": "OPPORTUNITY",
                "detail": (
                    f"Non-held {challenger_ticker} score={challenger_score:.2f} "
                    f"vs held avg={avg_held_conviction:.2f} "
                    f"(gap={challenger_score - avg_held_conviction:.2f})"
                ),
            })
            break  # one challenger is enough to flag the profile

    return violations


# ── Main task ─────────────────────────────────────────────────────────────────

def run_realignment_task(db_path: str = "regimes.db") -> bool:
    """
    Run the daily realignment check.

    Returns True if a portfolio rebuild was triggered, False otherwise.
    """
    logger.info("=== Realignment Task — Starting ===")
    today_str = datetime.date.today().isoformat()

    conn   = sqlite3.connect(db_path)
    init_snapshot_tables(conn)
    cursor = conn.cursor()

    # Need a snapshot to compare against
    snapshot_date = _get_latest_snapshot_date(cursor)
    if not snapshot_date:
        logger.info("No snapshot found — portfolios haven't been built yet. Skipping.")
        conn.close()
        return False

    if snapshot_date == today_str:
        logger.info("Snapshot is from today — portfolios were just rebuilt. Skipping.")
        conn.close()
        return False

    logger.info(f"Comparing today ({today_str}) vs snapshot ({snapshot_date})")

    portfolios = _get_portfolio_tickers(cursor, snapshot_date)
    if not portfolios:
        logger.info("No portfolio positions found in snapshot. Skipping.")
        conn.close()
        return False

    # Check each profile
    all_violations: list[dict] = []
    rebuild_needed = False

    for profile_id, held_tickers in sorted(portfolios.items()):
        violations = check_violations(cursor, snapshot_date, today_str, profile_id, held_tickers)

        if violations:
            logger.warning(f"Profile {profile_id}: {len(violations)} violation(s):")
            for v in violations:
                logger.warning(f"  [{v['violation_type']}] {v['ticker']} — {v['detail']}")
            all_violations.extend(violations)

            # Count non-opportunity violations (RISK + CONVICTION_DROP) vs held
            hard_violations = [v for v in violations if v["violation_type"] != "OPPORTUNITY"]
            ratio = len(hard_violations) / len(held_tickers) if held_tickers else 0

            if ratio >= REBUILD_TRIGGER_RATIO or any(
                v["violation_type"] == "OPPORTUNITY" for v in violations
            ):
                logger.warning(
                    f"Profile {profile_id}: rebuild threshold reached "
                    f"({ratio:.0%} hard violations, opportunity={any(v['violation_type']=='OPPORTUNITY' for v in violations)})"
                )
                rebuild_needed = True
        else:
            logger.info(f"Profile {profile_id}: No violations. Thesis intact.")

    conn.close()

    # Trigger rebuild if any profile exceeded thresholds
    if rebuild_needed:
        logger.warning("=== THESIS DECAY DETECTED — Triggering portfolio rebuild ===")
        try:
            from tasks.greenfield_portfolio_task import run_greenfield_models_task
            api_client = WebserverClient(WEBSERVER_URL)
            run_greenfield_models_task(api_client)
            logger.info("Portfolio rebuild completed.")
        except Exception as exc:
            logger.error(f"Portfolio rebuild failed: {exc}", exc_info=True)
    else:
        logger.info("=== All portfolio theses intact. No rebuild needed. ===")

    return rebuild_needed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_realignment_task()
