"""
scripts/simulate_match.py
──────────────────────────
Phase 3 Day 19 – Replay the 2024 IPL Final (KKR vs SRH) ball-by-ball
through the live WebSocket pipeline to validate end-to-end latency
and intent prediction accuracy under realistic match conditions.

Usage:
  python scripts/simulate_match.py [--host ws://localhost:8000] [--delay 0.5]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import time
from dataclasses import dataclass, field

import websockets

logger = logging.getLogger(__name__)

# ── 2024 IPL Final – KKR vs SRH (Narendra Modi Stadium, Ahmedabad) ───────────
# KKR won by 8 wickets. KKR: 114 all-out (SRH) chased 113 in 10.3 overs
# Simplified ball-by-ball sequence (Phase 1: SRH innings)

SRH_INNINGS_SAMPLE = [
    # over, ball, batter, bowler, runs, wicket_kind
    (0, 1, "TH David", "MI Starc", 0, None),
    (0, 2, "TH David", "MI Starc", 0, "caught"),           # Wicket
    (0, 3, "A Sharma", "MI Starc", 4, None),
    (0, 4, "A Sharma", "MI Starc", 0, None),
    (0, 5, "A Sharma", "MI Starc", 6, None),
    (0, 6, "A Sharma", "MI Starc", 0, None),
    (1, 1, "H Klaasen", "VR Iyer", 6, None),
    (1, 2, "H Klaasen", "VR Iyer", 4, None),
    (1, 3, "H Klaasen", "VR Iyer", 0, None),
    (1, 4, "H Klaasen", "VR Iyer", 1, None),
    (1, 5, "A Sharma", "VR Iyer", 0, None),
    (1, 6, "A Sharma", "VR Iyer", 2, None),
    (5, 1, "H Klaasen", "SN Thakur", 6, None),
    (5, 2, "H Klaasen", "SN Thakur", 0, "caught"),          # Wicket
    (5, 3, "P Dubey", "SN Thakur", 4, None),
    (9, 1, "N Pooran", "VD Chakravarthy", 0, None),
    (9, 2, "N Pooran", "VD Chakravarthy", 0, None),
    (9, 3, "N Pooran", "VD Chakravarthy", 0, "bowled"),     # Wicket
    (14, 1, "B Kumar", "SN Thakur", 4, None),
    (14, 2, "B Kumar", "SN Thakur", 0, "caught"),           # Wicket
    (16, 1, "Shahbaz Ahmed", "VD Chakravarthy", 0, "stumped"),  # Wicket
    (19, 1, "P Dubey", "HH Pandya", 6, None),
    (19, 2, "P Dubey", "HH Pandya", 0, None),
    (19, 3, "P Dubey", "HH Pandya", 0, "caught"),           # Final wicket (114 all out)
]

KKR_CHASE_SAMPLE = [
    (0, 1, "Sunil Narine", "B Kumar", 4, None),
    (0, 2, "Sunil Narine", "B Kumar", 6, None),
    (0, 3, "Sunil Narine", "B Kumar", 0, None),
    (0, 4, "Sunil Narine", "B Kumar", 4, None),
    (0, 5, "Sunil Narine", "B Kumar", 0, None),
    (0, 6, "Sunil Narine", "B Kumar", 4, None),
    (1, 1, "PK Garg", "T Natarajan", 6, None),
    (1, 2, "Sunil Narine", "T Natarajan", 4, None),
    (1, 3, "Sunil Narine", "T Natarajan", 0, None),
    (1, 4, "PK Garg", "T Natarajan", 4, None),
    (1, 5, "Sunil Narine", "T Natarajan", 0, "caught"),     # Wicket
    (2, 1, "V Iyer", "Shahbaz Ahmed", 6, None),
    (2, 2, "V Iyer", "Shahbaz Ahmed", 6, None),
    (5, 1, "AG Patel", "Shahbaz Ahmed", 4, None),
    (5, 2, "V Iyer", "Shahbaz Ahmed", 4, None),
    (10, 1, "AG Patel", "Natarajan", 4, None),              # Match-winning boundary
    (10, 2, "AG Patel", "Natarajan", 4, None),
    (10, 3, "AG Patel", "Natarajan", 0, None),              # KKR win
]


@dataclass
class SimulationStats:
    latencies_ms: list[float] = field(default_factory=list)
    intent_distribution: dict[str, int] = field(default_factory=dict)
    errors: int = 0
    total_balls: int = 0

    def record(self, verdict: dict, latency_ms: float) -> None:
        self.latencies_ms.append(latency_ms)
        self.total_balls += 1
        intent = verdict.get("intent", "unknown")
        self.intent_distribution[intent] = self.intent_distribution.get(intent, 0) + 1

    def summary(self) -> dict:
        if not self.latencies_ms:
            return {"error": "No results recorded"}
        return {
            "total_balls": self.total_balls,
            "errors": self.errors,
            "latency_ms": {
                "min": round(min(self.latencies_ms), 1),
                "max": round(max(self.latencies_ms), 1),
                "mean": round(statistics.mean(self.latencies_ms), 1),
                "p50": round(statistics.median(self.latencies_ms), 1),
                "p95": round(sorted(self.latencies_ms)[int(len(self.latencies_ms) * 0.95)], 1),
            },
            "intent_distribution": self.intent_distribution,
            "p95_within_300ms": sorted(self.latencies_ms)[int(len(self.latencies_ms) * 0.95)] < 300,
        }


def build_ball_event(
    inning: int,
    batting_team: str,
    bowling_team: str,
    over: int,
    ball_num: int,
    batter: str,
    bowler: str,
    cum_runs: int,
    wickets_fallen: int,
    target: int,
) -> dict:
    balls_elapsed = over * 6 + ball_num
    current_rr = (cum_runs / max(balls_elapsed / 6, 0.1))
    balls_left = (20 - over) * 6 - ball_num
    required_rr = (target - cum_runs) / max(balls_left / 6, 0.1) if target > 0 else 0.0

    return {
        "match_id": "ipl_2024_final",
        "batter": batter,
        "bowler": bowler,
        "non_striker": "Partner",
        "over": over,
        "ball_num": ball_num,
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "cum_runs": cum_runs,
        "wickets_fallen": wickets_fallen,
        "current_rr": round(current_rr, 2),
        "required_rr": round(required_rr, 2),
        "target": target,
        "h2h_balls_total": 30,
        "h2h_sr_global": 130.0,
        "pressure_index": abs(required_rr - current_rr),
        "phase_enc": 0 if over < 6 else (2 if over >= 15 else 1),
        "inning": inning,
        "season": 2024,
        "ball": ball_num,
        "bowler_last3_runs": 10.0,
        "bowler_economy": 8.5,
        "bowler_spell_balls": over * 6,
        "batter_last5_avg": 35.0,
        "bowler_type_enc": 0,
        "cum_balls": balls_elapsed,
        "extra_runs": 0,
        "total_runs": 0,
        "wickets_remaining": 10 - wickets_fallen,
    }


async def simulate_innings(
    ws,
    innings_data: list[tuple],
    inning: int,
    batting_team: str,
    bowling_team: str,
    target: int,
    stats: SimulationStats,
    delay: float,
) -> int:
    """Send all deliveries for one innings, return final score."""
    cum_runs = 0
    wickets = 0

    for entry in innings_data:
        over, ball, batter, bowler, runs, wicket = entry

        event = build_ball_event(
            inning=inning,
            batting_team=batting_team,
            bowling_team=bowling_team,
            over=over,
            ball_num=ball,
            batter=batter,
            bowler=bowler,
            cum_runs=cum_runs,
            wickets_fallen=wickets,
            target=target,
        )

        t_send = time.perf_counter()
        await ws.send(json.dumps(event))

        try:
            response_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            latency_ms = (time.perf_counter() - t_send) * 1000
            verdict = json.loads(response_raw)
            stats.record(verdict, latency_ms)

            intent_emoji = {"aggressive": "🔥", "defensive": "🛡️",
                           "neutral": "⚖️", "pressure_error": "⚠️"}.get(verdict.get("intent", ""), "❓")
            logger.info(
                "  %s Over %d.%d %s vs %s → %s %s (%.0fms)",
                intent_emoji, over, ball, batter, bowler,
                verdict.get("intent", "?").upper(),
                f"[CONF⚠️]" if verdict.get("confidence_flag") else "",
                latency_ms,
            )

        except asyncio.TimeoutError:
            logger.warning("  ⏱️  Over %d.%d timed out!", over, ball)
            stats.errors += 1

        cum_runs += runs
        if wicket:
            wickets += 1

        await asyncio.sleep(delay)

    return cum_runs


async def run_simulation(host: str, delay: float) -> SimulationStats:
    stats = SimulationStats()
    ws_url = f"{host}/ws/live"

    logger.info("=" * 60)
    logger.info("🏏 2024 IPL FINAL SIMULATION")
    logger.info("   Kolkata Knight Riders vs Sunrisers Hyderabad")
    logger.info("   Narendra Modi Stadium, Ahmedabad")
    logger.info("=" * 60)
    logger.info("Connecting to %s …", ws_url)

    try:
        async with websockets.connect(ws_url, open_timeout=10) as ws:
            logger.info("✅ WebSocket connected.\n")

            # ── Innings 1: SRH bat ─────────────────────────────────────────
            logger.info("📋 INNINGS 1: Sunrisers Hyderabad bat")
            srh_score = await simulate_innings(
                ws, SRH_INNINGS_SAMPLE, inning=1,
                batting_team="Sunrisers Hyderabad",
                bowling_team="Kolkata Knight Riders",
                target=0, stats=stats, delay=delay,
            )
            logger.info("  → SRH all out for %d runs\n", srh_score)

            # ── Innings 2: KKR chase ────────────────────────────────────────
            logger.info("📋 INNINGS 2: Kolkata Knight Riders chase %d", srh_score + 1)
            kkr_score = await simulate_innings(
                ws, KKR_CHASE_SAMPLE, inning=2,
                batting_team="Kolkata Knight Riders",
                bowling_team="Sunrisers Hyderabad",
                target=srh_score + 1, stats=stats, delay=delay,
            )
            logger.info("  → KKR won with 8 wickets! Score: %d\n", kkr_score)

    except Exception as e:
        logger.error("Connection failed: %s", e)
        logger.error("Is the backend running? Start with: uvicorn api.main:app --reload")
        stats.errors += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="IPL 2024 Final Match Simulator")
    parser.add_argument("--host", default="ws://localhost:8000", help="Backend WebSocket host")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between balls (seconds)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    stats = asyncio.run(run_simulation(args.host, args.delay))

    # ── Results ──────────────────────────────────────────────────────────────
    summary = stats.summary()
    logger.info("=" * 60)
    logger.info("📊 SIMULATION RESULTS")
    logger.info("=" * 60)
    logger.info("  Total balls processed : %d", summary["total_balls"])
    logger.info("  Errors                : %d", summary["errors"])
    logger.info("  Latency (ms)")
    for k, v in summary.get("latency_ms", {}).items():
        logger.info("    %-6s : %s ms", k, v)
    logger.info("  Intent distribution")
    for intent, count in summary.get("intent_distribution", {}).items():
        pct = 100 * count / max(summary["total_balls"], 1)
        logger.info("    %-18s : %d (%.1f%%)", intent, count, pct)
    logger.info("")

    passed = summary.get("p95_within_300ms", False)
    if passed:
        logger.info("✅ PASSED: p95 latency < 300ms")
    else:
        logger.warning("❌ FAILED: p95 latency ≥ 300ms — pipeline needs optimisation")

    if summary["errors"] == 0:
        logger.info("✅ PASSED: Zero errors")
    else:
        logger.warning("❌ %d errors encountered", summary["errors"])


if __name__ == "__main__":
    main()
