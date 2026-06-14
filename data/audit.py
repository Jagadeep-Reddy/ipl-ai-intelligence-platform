"""
data/audit.py
─────────────
Phase 1 Day 1 – Automated data quality audit using ydata-profiling.
Detects nulls, outliers, team-name inconsistencies, and player name
variants before any downstream processing.

Run: python data/audit.py
Outputs: data/reports/matches_profile.html, deliveries_profile.html
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from ydata_profiling import ProfileReport

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
REPORT_DIR = Path("data/reports")

# Columns we care most about for quality
CRITICAL_MATCH_COLS = ["id", "season", "team1", "team2", "winner", "venue", "date"]
CRITICAL_DELIVERY_COLS = [
    "match_id", "inning", "over", "ball", "batter", "bowler",
    "batsman_runs", "total_runs", "dismissal_kind",
]

KNOWN_TEAM_VARIANTS = {
    "Deccan Chargers", "SunRisers Hyderabad", "Delhi Daredevils",
    "Kings XI Punjab", "Rising Pune Supergiant", "Rising Pune Supergiants",
}


def run_audit(
    matches_path: Path | None = None,
    deliveries_path: Path | None = None,
) -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Auto-detect paths
    if matches_path is None:
        candidates = list(RAW_DIR.glob("*matches*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No matches CSV found in {RAW_DIR}")
        matches_path = candidates[0]

    if deliveries_path is None:
        candidates = list(RAW_DIR.glob("*deliveries*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No deliveries CSV found in {RAW_DIR}")
        deliveries_path = candidates[0]

    logger.info("Loading %s …", matches_path)
    matches = pd.read_csv(matches_path)
    logger.info("Loading %s …", deliveries_path)
    deliveries = pd.read_csv(deliveries_path, low_memory=False)

    issues = {}

    # ── Matches audit ────────────────────────────────────────────────────────
    logger.info("Auditing matches.csv …")

    # Null check on critical columns
    null_counts = matches[CRITICAL_MATCH_COLS].isnull().sum()
    issues["matches_nulls"] = null_counts[null_counts > 0].to_dict()
    if issues["matches_nulls"]:
        logger.warning("Null values in matches: %s", issues["matches_nulls"])

    # Duplicate match IDs
    dup_ids = matches["id"].duplicated().sum()
    issues["matches_duplicate_ids"] = int(dup_ids)
    if dup_ids:
        logger.warning("Duplicate match IDs: %d", dup_ids)

    # Team name variants
    all_teams = set(matches["team1"].dropna()) | set(matches["team2"].dropna())
    team_variants = all_teams & KNOWN_TEAM_VARIANTS
    issues["team_name_variants"] = sorted(team_variants)
    logger.info("Team name variants requiring normalisation: %s", team_variants)

    # Season coverage
    seasons = sorted(matches["season"].dropna().unique().tolist()) if "season" in matches.columns else []
    if not seasons and "date" in matches.columns:
        matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
        seasons = sorted(matches["date"].dt.year.dropna().unique().tolist())
    issues["seasons_covered"] = seasons
    logger.info("Seasons in dataset: %s", seasons)

    # ── Deliveries audit ─────────────────────────────────────────────────────
    logger.info("Auditing deliveries.csv …")

    null_del = deliveries[
        [c for c in CRITICAL_DELIVERY_COLS if c in deliveries.columns]
    ].isnull().sum()
    issues["deliveries_nulls"] = null_del[null_del > 0].to_dict()
    if issues["deliveries_nulls"]:
        logger.warning("Null values in deliveries: %s", issues["deliveries_nulls"])

    total_deliveries = len(deliveries)
    issues["total_deliveries"] = total_deliveries
    logger.info("Total deliveries: %s", f"{total_deliveries:,}")

    # Detect outlier run values
    if "total_runs" in deliveries.columns:
        outlier_runs = deliveries[deliveries["total_runs"] > 7]
        issues["outlier_deliveries_runs_gt7"] = int(len(outlier_runs))
        if len(outlier_runs):
            logger.warning("Deliveries with total_runs > 7: %d", len(outlier_runs))

    # Player name checks
    batters = set(deliveries["batter"].dropna() if "batter" in deliveries.columns
                  else deliveries["batsman"].dropna() if "batsman" in deliveries.columns else [])
    issues["unique_batters"] = len(batters)
    logger.info("Unique batters: %d", len(batters))

    # ── Generate HTML profiles ────────────────────────────────────────────────
    logger.info("Generating ydata-profiling report for matches.csv …")
    matches_profile = ProfileReport(
        matches,
        title="IPL Matches – Data Profile",
        minimal=True,
        progress_bar=False,
    )
    matches_report_path = REPORT_DIR / "matches_profile.html"
    matches_profile.to_file(matches_report_path)
    logger.info("Saved: %s", matches_report_path)

    logger.info("Generating ydata-profiling report for deliveries.csv (sample 50k) …")
    deliveries_profile = ProfileReport(
        deliveries.sample(min(50_000, len(deliveries)), random_state=42),
        title="IPL Deliveries – Data Profile",
        minimal=True,
        progress_bar=False,
    )
    deliveries_report_path = REPORT_DIR / "deliveries_profile.html"
    deliveries_profile.to_file(deliveries_report_path)
    logger.info("Saved: %s", deliveries_report_path)

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("AUDIT SUMMARY")
    logger.info("  Matches:              %d", len(matches))
    logger.info("  Deliveries:           %s", f"{total_deliveries:,}")
    logger.info("  Seasons:              %s", seasons)
    logger.info("  Team variants:        %s", issues["team_name_variants"])
    logger.info("  Match null columns:   %s", list(issues["matches_nulls"].keys()))
    logger.info("  Delivery null cols:   %s", list(issues["deliveries_nulls"].keys()))
    logger.info("=" * 50)
    logger.info("ACTION REQUIRED: Apply TEAM_ALIASES in data/ingest.py before proceeding.")

    return issues


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    issues = run_audit()
