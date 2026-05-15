"""
Realignment Orchestrator (Task 3)
===================================
LLM-driven portfolio review. For each risk profile:

  1. Calls get_realignment_candidates() to fetch Weakest Links vs Challengers
  2. Presents the comparison to Ollama with a structured decision prompt
  3. Ollama produces a trade_plan: per-position SWAP / HOLD decisions + rationale
  4. Applies the confidence gate (confidence < threshold → BLOCKED)
  5. Logs every decision to trade_log
  6. Pushes the approved plan to the webserver

Run standalone or add to the daemon after the daily pipeline:
    python tasks/realignment_orchestrator.py
    python tasks/realignment_orchestrator.py --profiles 1 2 3

This is distinct from realignment_task.py (which is rule-based).
The orchestrator adds an LLM reasoning layer on top, explaining WHY
each swap is recommended in natural language.
"""

import argparse
import datetime
import json
import logging
import os
import requests
import sqlite3
import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv(Path(project_root) / ".env")

from mcp_server.tools import get_realignment_candidates
from tasks.trade_log import init_trade_log_table, log_trade, log_swap
from config.settings import WEBSERVER_URL
from connectors.webserver_client import WebserverClient

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))
CONFIDENCE_THRESHOLD = float(os.getenv("AGENT_CONFIDENCE_THRESHOLD", "0.80"))

_base = Path(__file__).parent.parent
REGIMES_DB = os.getenv("REGIMES_DB_PATH", str(_base / "regimes.db"))

RISK_PROFILES = {
    1: "Conservative",
    2: "Moderately Conservative",
    3: "Moderate",
    4: "Moderately Aggressive",
    5: "Aggressive",
}


# ── LLM interaction ───────────────────────────────────────────────────────────

def _build_orchestrator_prompt(profile_id: int, candidates: dict) -> str:
    """Build the structured decision prompt for Ollama."""
    profile_name  = candidates.get("profile_name", f"Profile {profile_id}")
    eval_date     = candidates.get("evaluation_date", "today")
    weakest       = candidates.get("weakest_links", [])
    challengers   = candidates.get("challengers", [])

    # Format weakest links
    weak_lines = []
    for w in weakest[:5]:
        sentiment_str = f", sentiment={w['sentiment']:.3f}" if w.get("sentiment") else ""
        weak_lines.append(
            f"  - {w['ticker']}: conviction={w['conviction']:+.2f}, "
            f"win_prob={w['prob_positive']:.1%}, "
            f"ES_95={w['es_95']:.2%}{sentiment_str} "
            f"[weakness={w['weakness_score']:.2f}]"
        )

    # Format challengers
    chall_lines = []
    for c in challengers:
        ret_str = f", expected_return={c['expected_return']:.2%}" if c.get("expected_return") else ""
        chall_lines.append(
            f"  - {c['ticker']}: conviction={c['conviction']:+.2f}, "
            f"win_prob={c.get('prob_positive', 'N/A')}"
            f"{ret_str}"
        )

    return f"""You are a portfolio manager for the {profile_name} risk profile (as of {eval_date}).

CURRENT WEAKEST HELD POSITIONS (ranked worst first):
{chr(10).join(weak_lines) if weak_lines else '  None found.'}

TOP MARKET CHALLENGERS (not currently held):
{chr(10).join(chall_lines) if chall_lines else '  None found.'}

Apply these capital rotation rules:
  Rule 1: If a held position has ES_95 < -15%, it MUST be exited (risk violation).
  Rule 2: If a held position's sentiment is significantly negative, exit even if math is stable.
  Rule 3: Only swap a held position for a challenger if the challenger's expected return
          is at least 1.5% higher. Marginal improvement is not worth the disruption.
  Rule 4: Do NOT exit a held position just because a challenger looks slightly better
          if the held position's win_probability and sentiment remain healthy.

For EACH of the weakest held positions, output a SWAP or HOLD decision with rationale.
Then output your overall confidence in these recommendations.

Respond ONLY with this exact JSON structure:
{{
  "decisions": [
    {{
      "ticker": "<held ticker>",
      "action": "SWAP" or "HOLD",
      "swap_with": "<challenger ticker or null>",
      "rationale": "<one clear sentence explaining the decision>",
      "rule_triggered": "RISK_VIOLATION | NARRATIVE_DECAY | OPPORTUNITY_COST | PROTECT_MOMENTUM | null"
    }}
  ],
  "confidence_score": <float 0.0 to 1.0>,
  "overall_summary": "<one sentence summarising the portfolio health>"
}}"""


def call_ollama_for_plan(prompt: str) -> dict:
    """Call Ollama and parse the trade plan JSON."""
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"},
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


# ── Per-profile orchestration ─────────────────────────────────────────────────

def orchestrate_profile(
    profile_id: int,
    today_str: str,
    conn: sqlite3.Connection,
    api_client: WebserverClient,
) -> dict | None:
    """
    Run the full LLM realignment review for one risk profile.
    Returns the trade_plan dict, or None if blocked/failed.
    """
    profile_name = RISK_PROFILES.get(profile_id, f"Profile {profile_id}")
    logger.info(f"[Profile {profile_id} — {profile_name}] Fetching candidates...")

    candidates = get_realignment_candidates(profile_id)
    if "error" in candidates:
        logger.warning(f"  Skipping: {candidates['error']}")
        return None

    n_weak  = len(candidates.get("weakest_links", []))
    n_chall = len(candidates.get("challengers", []))
    logger.info(f"  {n_weak} weakest links, {n_chall} challengers identified.")

    if n_weak == 0:
        logger.info(f"  No weak positions found. Skipping.")
        return None

    # Ask Ollama for a trade plan
    prompt = _build_orchestrator_prompt(profile_id, candidates)
    raw    = call_ollama_for_plan(prompt)

    if not raw:
        logger.warning(f"  [Profile {profile_id}] Ollama returned empty response.")
        return None

    confidence    = float(raw.get("confidence_score", 0.0))
    decisions     = raw.get("decisions", [])
    summary       = raw.get("overall_summary", "")

    logger.info(f"  LLM confidence: {confidence:.2f} | Summary: {summary}")

    # Confidence gate
    if confidence < CONFIDENCE_THRESHOLD:
        logger.warning(
            f"  [Profile {profile_id}] Confidence {confidence:.2f} < {CONFIDENCE_THRESHOLD}. "
            f"Logging as BLOCKED."
        )
        log_trade(
            conn,
            action="BLOCKED",
            ticker="PORTFOLIO",
            trigger_type="CONFIDENCE_GATE",
            rationale=(
                f"Realignment plan for Profile {profile_id} ({profile_name}) blocked: "
                f"LLM confidence {confidence:.2f} below threshold {CONFIDENCE_THRESHOLD}. "
                f"Summary: {summary}"
            ),
            log_date=today_str,
            profile_id=profile_id,
            profile_name=profile_name,
            confidence_score=confidence,
            executed=0,
        )
        return None

    # Build challenger lookup for enrichment
    challengers_by_ticker = {
        c["ticker"]: c for c in candidates.get("challengers", [])
    }
    weakest_by_ticker = {
        w["ticker"]: w for w in candidates.get("weakest_links", [])
    }

    trade_plan = []

    for decision in decisions:
        ticker       = decision.get("ticker", "")
        action       = decision.get("action", "HOLD").upper()
        swap_with    = decision.get("swap_with")
        rationale    = decision.get("rationale", "No rationale provided.")
        rule_hit     = decision.get("rule_triggered")
        weak_metrics = weakest_by_ticker.get(ticker, {})
        chall_metrics = challengers_by_ticker.get(swap_with, {}) if swap_with else {}

        logger.info(f"  {action:6s} {ticker:6s} {('→ ' + swap_with) if swap_with else ''} | {rationale[:70]}")

        if action == "SWAP" and swap_with:
            log_swap(
                conn,
                log_date=today_str,
                profile_id=profile_id,
                profile_name=profile_name,
                sell_ticker=ticker,
                buy_ticker=swap_with,
                sell_conviction=weak_metrics.get("conviction", 0.0),
                buy_conviction=chall_metrics.get("conviction", 0.0),
                sell_es_95=weak_metrics.get("es_95"),
                sell_mean_return=None,
                buy_mean_return=chall_metrics.get("expected_return"),
                confidence_score=confidence,
                executed=1,
            )
        elif action == "HOLD":
            log_trade(
                conn,
                action="HOLD",
                ticker=ticker,
                trigger_type=rule_hit or "PROTECT_MOMENTUM",
                rationale=rationale,
                log_date=today_str,
                profile_id=profile_id,
                profile_name=profile_name,
                conviction_score=weak_metrics.get("conviction"),
                es_95=weak_metrics.get("es_95"),
                confidence_score=confidence,
                executed=1,
            )

        trade_plan.append({
            "ticker":        ticker,
            "action":        action,
            "swap_with":     swap_with,
            "rationale":     rationale,
            "rule_triggered": rule_hit,
        })

    return {
        "profile_id":       profile_id,
        "profile_name":     profile_name,
        "evaluation_date":  today_str,
        "confidence_score": confidence,
        "overall_summary":  summary,
        "trade_plan":       trade_plan,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_realignment_orchestrator(profiles: list[int] = None) -> list[dict]:
    """
    Run the LLM realignment review for the specified profiles (default: all 5).
    Returns a list of approved trade plans.
    """
    if profiles is None:
        profiles = list(RISK_PROFILES.keys())

    today_str  = datetime.date.today().isoformat()
    api_client = WebserverClient(WEBSERVER_URL)

    conn = sqlite3.connect(REGIMES_DB)
    init_trade_log_table(conn)

    logger.info("=" * 60)
    logger.info(f"Realignment Orchestrator — {today_str}")
    logger.info(f"Profiles: {profiles} | Model: {OLLAMA_MODEL}")
    logger.info("=" * 60)

    approved_plans = []

    for pid in profiles:
        plan = orchestrate_profile(pid, today_str, conn, api_client)
        if plan:
            approved_plans.append(plan)

    conn.close()

    if approved_plans:
        # Push all approved plans to webserver
        try:
            payload = {"evaluation_date": today_str, "plans": approved_plans}
            api_client.post_realignment_plans(payload)
            logger.info(f"✅ {len(approved_plans)} realignment plan(s) pushed to webserver.")
        except Exception as exc:
            logger.warning(f"Webserver push failed (non-critical): {exc}")

    logger.info("=" * 60)
    logger.info(f"Orchestrator complete. {len(approved_plans)} plan(s) approved.")
    logger.info("=" * 60)

    return approved_plans


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="LLM-driven portfolio realignment orchestrator")
    parser.add_argument(
        "--profiles",
        type=int,
        nargs="+",
        default=None,
        help="Profile IDs to review (1-5). Defaults to all.",
    )
    args = parser.parse_args()

    run_realignment_orchestrator(profiles=args.profiles)
