from __future__ import annotations
# data/ingest.py — Phase 1 Day 1: ETL, normalisation, intent labelling
# Sources: Kaggle 2008-2020 + Cricsheet 2021-2024 (no login needed)
 
import logging
import subprocess
import urllib.request
import zipfile
from pathlib import Path
 
import pandas as pd
 
logger = logging.getLogger(__name__)
 
CRICSHEET_IPL_URL = "https://cricsheet.org/downloads/ipl_csv2.zip"
 
# ── Canonical team-name map ───────────────────────────────────────────────────
TEAM_ALIASES: dict[str, str] = {
    "Deccan Chargers":            "Sunrisers Hyderabad",
    "SunRisers Hyderabad":        "Sunrisers Hyderabad",
    "Delhi Daredevils":           "Delhi Capitals",
    "Delhi Capitals":             "Delhi Capitals",
    "Kings XI Punjab":            "Punjab Kings",
    "Punjab Kings":               "Punjab Kings",
    "Rising Pune Supergiant":     "Rising Pune Supergiants",
    "Rising Pune Supergiants":    "Rising Pune Supergiants",
    "Mumbai Indians":             "Mumbai Indians",
    "Chennai Super Kings":        "Chennai Super Kings",
    "Kolkata Knight Riders":      "Kolkata Knight Riders",
    "Royal Challengers Bangalore":"Royal Challengers Bengaluru",
    "Royal Challengers Bengaluru":"Royal Challengers Bengaluru",
    "Rajasthan Royals":           "Rajasthan Royals",
    "Gujarat Titans":             "Gujarat Titans",
    "Lucknow Super Giants":       "Lucknow Super Giants",
    "Kochi Tuskers Kerala":       "Kochi Tuskers Kerala",
    "Pune Warriors":              "Pune Warriors",
}
 
 
def normalise_teams(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].map(TEAM_ALIASES).fillna(df[col])
    return df
 
 
def label_intent(row: pd.Series) -> str:
    """
    aggressive     – batsman scores >= 4 runs
    defensive      – dot ball (0 batsman runs, 0 extras)
    pressure_error – dismissal on this ball
    neutral        – everything else
    """
    if pd.notna(row.get("dismissal_kind")):
        return "pressure_error"
    batsman_runs = row.get("batsman_runs", row.get("runs_off_bat", 0))
    extras = row.get("extras", row.get("extra_runs", 0))
    if batsman_runs >= 4:
        return "aggressive"
    if batsman_runs == 0 and extras == 0:
        return "defensive"
    return "neutral"
 
 
# ── Cricsheet helpers ─────────────────────────────────────────────────────────
 
def _parse_info_file(raw_bytes: bytes) -> dict[str, str]:
    meta = {}
    lines = raw_bytes.decode("utf-8").splitlines()
    for line in lines:
        if line.startswith("info,"):
            parts = line.split(",", 2)
            if len(parts) == 3:
                meta[parts[1].strip()] = parts[2].strip()
    return meta


def _parse_cricsheet_file(raw_bytes: bytes, filename: str, info_bytes: bytes | None = None) -> pd.DataFrame | None:
    try:
        import io
        lines = raw_bytes.decode("utf-8").splitlines()
        
        meta = {}
        if info_bytes is not None:
            meta = _parse_info_file(info_bytes)
        else:
            # Fallback for legacy format with inline info lines
            data_lines = []
            for line in lines:
                if line.startswith("info,"):
                    parts = line.split(",", 2)
                    if len(parts) == 3:
                        meta[parts[1].strip()] = parts[2].strip()
                else:
                    data_lines.append(line)
            lines = data_lines
        
        df = pd.read_csv(io.StringIO("\n".join(lines)), low_memory=False)
        required = {"innings", "ball", "batting_team", "bowling_team", "striker"}
        if not required.issubset(df.columns):
            logger.debug("Skipping %s — missing columns", filename)
            return None
        
        match_id = Path(filename).stem
        if "match_id" not in df.columns:
            df["match_id"] = match_id
        
        # Attach metadata to every row
        for key in ("winner", "venue", "city", "toss_winner", "toss_decision",
                    "player_of_match", "date"):
            if key in meta and key not in df.columns:
                df[key] = meta[key]
        
        return df
    except Exception as exc:
        logger.debug("Skipping %s: %s", filename, exc)
        return None
 
 
def _build_match_index(deliveries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for match_id, g in deliveries.groupby("match_id"):
        teams = g["batting_team"].dropna().unique().tolist()
        rows.append({
            "id":            match_id,
            "date":          g["start_date"].iloc[0] if "start_date" in g.columns else None,
            "venue":         g["venue"].iloc[0]      if "venue"      in g.columns else None,
            "team1":         teams[0] if len(teams) > 0 else None,
            "team2":         teams[1] if len(teams) > 1 else None,
            "winner":        g["winner"].iloc[0]     if "winner"     in g.columns else None,
            "toss_winner":   g["toss_winner"].iloc[0]   if "toss_winner"   in g.columns else None,
            "toss_decision": g["toss_decision"].iloc[0] if "toss_decision" in g.columns else None,
            "win_by_runs":   0,
            "win_by_wickets":0,
            "target_runs":   0,
        })
    return pd.DataFrame(rows)
 
 
def parse_cricsheet_zip(zip_path: Path, data_dir: Path) -> None:
    if not zip_path.exists():
        logger.warning("Cricsheet ZIP file not found at %s", zip_path)
        return

    all_deliveries: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as zf:
        namelist = zf.namelist()
        match_files = [n for n in namelist if n.endswith(".csv") and not n.endswith("_info.csv")]
        logger.info("Cricsheet archive: %d match files", len(match_files))
        for name in match_files:
            info_name = name.replace(".csv", "_info.csv")
            info_bytes = None
            if info_name in namelist:
                info_bytes = zf.read(info_name)
            
            df = _parse_cricsheet_file(zf.read(name), name, info_bytes)
            if df is not None:
                all_deliveries.append(df)
 
    if not all_deliveries:
        logger.warning("No Cricsheet files parsed — skipping.")
        return
 
    deliveries_df = pd.concat(all_deliveries, ignore_index=True)
 
    # Keep only 2021+ to avoid overlap with Kaggle 2008-2020
    if "start_date" in deliveries_df.columns:
        deliveries_df["start_date"] = pd.to_datetime(
            deliveries_df["start_date"], errors="coerce"
        )
        deliveries_df = deliveries_df[
            deliveries_df["start_date"].dt.year >= 2021
        ].copy()
 
    matches_df = _build_match_index(deliveries_df)
    matches_df.to_csv(data_dir / "matches_cricsheet.csv", index=False)
    deliveries_df.to_csv(data_dir / "deliveries_cricsheet.csv", index=False)
    logger.info(
        "Cricsheet: %d matches, %d deliveries (2021-2024)",
        len(matches_df), len(deliveries_df),
    )


def download_cricsheet(data_dir: Path) -> None:
    zip_path = data_dir / "ipl_cricsheet.zip"
    data_dir.mkdir(parents=True, exist_ok=True)
 
    logger.info("Downloading Cricsheet from %s ...", CRICSHEET_IPL_URL)
    urllib.request.urlretrieve(CRICSHEET_IPL_URL, zip_path)
    logger.info("Downloaded %.1f MB", zip_path.stat().st_size / 1_048_576)
 
    parse_cricsheet_zip(zip_path, data_dir)
 
 
def download_datasets(data_dir: Path = Path("data/raw")) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
 
    logger.info("Downloading Kaggle 2008-2020 dataset ...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download",
             "-d", "patrickb1912/ipl-complete-dataset-20082020",
             "-p", str(data_dir), "--unzip"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.warning(
            "Kaggle download failed (exit %s) — continuing with existing files.",
            e.returncode,
        )
 
    download_cricsheet(data_dir)
 
 
def load_and_merge_raw(
    data_dir: Path = Path("data/raw"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    match_files    = sorted(data_dir.glob("*matches*.csv"))
    delivery_files = sorted(data_dir.glob("*deliveries*.csv"))
 
    if not match_files:
        raise FileNotFoundError(
            f"No matches CSV in {data_dir}. "
            "Run without --skip-download, or place CSVs in data/raw/ manually."
        )
    if not delivery_files:
        raise FileNotFoundError(f"No deliveries CSV in {data_dir}.")
 
    matches    = pd.concat([pd.read_csv(f) for f in match_files],    ignore_index=True)
    deliveries = pd.concat([pd.read_csv(f) for f in delivery_files], ignore_index=True)
    logger.info("Loaded %d matches, %d deliveries", len(matches), len(deliveries))
    return matches, deliveries
 
 
def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    if "id" not in df.columns and "match_id" in df.columns:
        df = df.rename(columns={"match_id": "id"})
    elif "match_id" in df.columns:
        df = df.drop(columns=["match_id"])
 
    df = normalise_teams(df, ["team1", "team2", "winner", "toss_winner"])
    df["date"]   = pd.to_datetime(df["date"], errors="coerce")
    df["season"] = df["date"].dt.year.astype("Int64")
 
    defaults = {
        "target_runs": 0, "win_by_runs": 0, "win_by_wickets": 0,
        "player_of_match": None, "dl_applied": False,
        "city": None, "venue": None, "toss_decision": None,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
 
    for col in ("target_runs", "win_by_runs", "win_by_wickets"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
 
    return df.drop_duplicates(subset=["id"], keep="last").reset_index(drop=True)
 
 
def clean_deliveries(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "runs_off_bat": "batsman_runs",
        "extras":       "extra_runs",
        "wicket_type":  "dismissal_kind",
        "striker":      "batter",     # Cricsheet uses 'striker'
        "batsman":      "batter",     # old Kaggle format
        "innings":      "inning",     # Cricsheet uses plural
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df = df.loc[:, ~df.columns.duplicated()]
    if "match_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "match_id"})
 
    if "total_runs" not in df.columns:
        bat = pd.to_numeric(df.get("batsman_runs", 0), errors="coerce").fillna(0)
        ext = pd.to_numeric(df.get("extra_runs",   0), errors="coerce").fillna(0)
        df["total_runs"] = (bat + ext).astype(int)
 
    df = normalise_teams(df, ["batting_team", "bowling_team"])
 
    for col in ["batsman_runs", "extra_runs", "total_runs"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
 
    for col in ["over", "ball", "inning"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
 
    return df
 
 
def build_pipeline(
    data_dir: Path = Path("data/raw"),
    out_dir:  Path = Path("data/processed"),
    download: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
 
    if download:
        download_datasets(data_dir)
    else:
        # Re-parse existing local Cricsheet ZIP to apply updated parsing logic
        zip_path = data_dir / "ipl_cricsheet.zip"
        if zip_path.exists():
            logger.info("Re-parsing existing local Cricsheet ZIP to apply parser updates...")
            parse_cricsheet_zip(zip_path, data_dir)
        else:
            logger.warning("Local ipl_cricsheet.zip not found.")
 
    matches, deliveries = load_and_merge_raw(data_dir)
    matches    = clean_matches(matches)
    deliveries = clean_deliveries(deliveries)
 
    logger.info("Applying intent labels to %d deliveries ...", len(deliveries))
    deliveries["intent"] = deliveries.apply(label_intent, axis=1)
 
    matches.to_csv(out_dir    / "matches_clean.csv",       index=False)
    deliveries.to_csv(out_dir / "deliveries_labelled.csv", index=False)
 
    logger.info(
        "Saved matches_clean.csv (%d rows), deliveries_labelled.csv (%d rows)",
        len(matches), len(deliveries),
    )
    logger.info("Intent distribution:\n%s", deliveries["intent"].value_counts())
    return matches, deliveries
 
 
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_pipeline(download=False)