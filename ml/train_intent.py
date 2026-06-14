"""
ml/train_intent.py
──────────────────
Phase 3 Days 15-16 – Train XGBoost Model B: per-ball batsman intent classifier.
20 contextual features → 4-class classification
  aggressive | defensive | neutral | pressure_error
Target: ~71% accuracy on 2024 holdout.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

INTENT_CLASSES = ["aggressive", "defensive", "neutral", "pressure_error"]

# ─────────────────────────────────────────────────────────────────────────
# Single source of truth for the train/serve feature contract.
#
# `engineer_features()` produces columns using names that are natural for
# the raw deliveries dataframe (e.g. "ball", "wickets_remaining",
# "cum_balls", "h2h_sr_global"). The live BallEvent schema (api/schemas.py)
# uses different names for the same concepts (e.g. "ball_num",
# "wickets_fallen", ...).
#
# Historically `predict_intent()` looked up `feature_cols` directly against
# the live BallEvent dict, so any name mismatch silently fell back to 0 —
# this caused the model to always land in the "pressure_error" region of
# feature space regardless of the real match situation.
#
# Fix: rename the engineered-feature columns to their BallEvent-canonical
# names *before* they are persisted as `feature_cols` in the model
# artifact, so training and serving always speak the same vocabulary.
# `predict_intent()` then needs no special-casing at all.
TRAIN_TO_SERVING_NAME = {
    "ball": "ball_num",
    "wickets_remaining": "wickets_remaining",  # kept, but derived from wickets_fallen at serve time
    "cum_balls": "cum_balls",
    "h2h_sr_global": "h2h_sr",
}

# Features that exist at training time (computed from ball-by-ball outcomes)
# but are NOT knowable before a live ball is bowled. They are intentionally
# excluded from the live feature contract; predict_intent fills them with
# neutral defaults.
LIVE_UNAVAILABLE_DEFAULTS = {
    "extra_runs": 0.0,
    "total_runs": 1.0,   # average outcome of a ball (≈1 run)
    "cum_balls": 0.0,    # overwritten below from over/ball_num if missing
    "wickets_remaining": 10.0,  # overwritten below from wickets_fallen if missing
    "h2h_sr": 100.0,
}


def engineer_features(deliveries: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """
    Build the 20-feature matrix for every delivery.
    Features mirror the domain specification exactly.
    """
    # Merge season from matches
    # Drop any existing season column from deliveries so the merge doesn't
    # produce season_x / season_y suffixes.
    deliveries = deliveries.drop(columns=["season"], errors="ignore")
    df = deliveries.merge(matches[["id", "season"]], left_on="match_id", right_on="id", how="left")
    df = df.rename(columns={"id_y": "match_season_id"})

    # Phase encoding
    df["phase_enc"] = df["phase"].map({"powerplay": 0, "middle": 1, "death": 2}).fillna(1)

    # Running totals per innings
    df = df.sort_values(["match_id", "inning", "over", "ball"]).reset_index(drop=True)
    df["cum_runs"] = df.groupby(["match_id", "inning"])["total_runs"].cumsum()
    df["cum_balls"] = df.groupby(["match_id", "inning"]).cumcount() + 1
    df["cum_wickets"] = df.groupby(["match_id", "inning"])["is_wicket"].cumsum()

    # Current run rate
    df["current_rr"] = (df["cum_runs"] / df["cum_balls"] * 6).round(2).fillna(0)

    # Wickets remaining
    df["wickets_remaining"] = 10 - df["cum_wickets"]

    # Required run rate (for 2nd innings)
    # Approximate: target from match not easily available; use a proxy
    df["required_rr"] = df["current_rr"] * 1.1  # placeholder; replace with actual target

    # Pressure index: |required_rr - current_rr|
    df["pressure_index"] = (df["required_rr"] - df["current_rr"]).abs().round(2)

    # Bowler last-3-balls runs conceded
    df["bowler_last3_runs"] = (
        df.groupby(["match_id", "bowler"])["total_runs"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
        .fillna(0)
    )

    # Bowler economy this spell (approx: session = same match)
    df["bowler_spell_balls"] = df.groupby(["match_id", "bowler"]).cumcount() + 1
    df["bowler_spell_runs"] = df.groupby(["match_id", "bowler"])["total_runs"].cumsum()
    df["bowler_economy"] = (df["bowler_spell_runs"] / df["bowler_spell_balls"] * 6).round(2)

    # Historical H2H strike rate – need a summary table
    # Pre-compute batter vs bowler global SR
    h2h_sr = (
        df.groupby(["batter", "bowler"])
        .apply(lambda g: 100.0 * g["batsman_runs"].sum() / max(len(g), 1))
        .reset_index(name="h2h_sr_global")
    )
    df = df.merge(h2h_sr, on=["batter", "bowler"], how="left")
    df["h2h_sr_global"] = df["h2h_sr_global"].fillna(100.0)

    # Batter's recent form (last 5 innings average)
    batter_inn_avg = (
        df.groupby(["batter", "match_id"])["batsman_runs"]
        .sum()
        .reset_index()
        .groupby("batter")["batsman_runs"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )
    df["batter_last5_avg"] = (
        df.merge(
            df.groupby(["batter", "match_id"])["batsman_runs"].sum().rename("inn_runs").reset_index(),
            on=["batter", "match_id"],
            how="left",
        )["inn_runs"]
        .fillna(20.0)
    )

    # Bowler type (pace/spin proxy by name – simplified)
    # In production: join a player metadata table
    df["bowler_type_enc"] = 0  # 0=pace, 1=spin (populated downstream)

    # Select final 20 features
    feature_cols = [
        "over",                    # 1
        "ball",                    # 2
        "phase_enc",               # 3
        "current_rr",              # 4
        "required_rr",             # 5
        "pressure_index",          # 6
        "wickets_remaining",       # 7
        "cum_runs",                # 8
        "cum_balls",               # 9
        "h2h_sr_global",           # 10
        "bowler_last3_runs",       # 11
        "bowler_economy",          # 12
        "bowler_spell_balls",      # 13
        "batter_last5_avg",        # 14
        "bowler_type_enc",         # 15
        "inning",                  # 16
        "batsman_runs",            # 17  (won't be available live; used for validation only)
        "extra_runs",              # 18
        "total_runs",              # 19
        "season",                  # 20
    ]
    # Drop columns that leak the label in live mode (batsman_runs used only for training)
    TRAIN_FEATURES = [c for c in feature_cols if c not in ("batsman_runs",)]
    df["season"] = df["season"].fillna(2023).astype(int)

    # Rename engineered columns to their canonical BallEvent/serving names so
    # the persisted `feature_cols` list (and X matrix below) line up exactly
    # with what `predict_intent()` receives at inference time. See
    # TRAIN_TO_SERVING_NAME for the full rationale.
    df = df.rename(columns=TRAIN_TO_SERVING_NAME)
    TRAIN_FEATURES = [TRAIN_TO_SERVING_NAME.get(c, c) for c in TRAIN_FEATURES]

    return df, TRAIN_FEATURES


def train_intent_classifier(
    deliveries_path: Path = Path("data/processed/deliveries_labelled.csv"),
    matches_path: Path = Path("data/processed/matches_clean.csv"),
    model_path: Path = Path("ml/models/intent_classifier.joblib"),
    n_optuna_trials: int = 50,
) -> dict:
    logger.info("Loading deliveries …")
    deliveries = pd.read_csv(deliveries_path, low_memory=False)
    matches = pd.read_csv(matches_path, low_memory=False)
    matches = matches.rename(columns={"id": "id"})

    # Assign phase if missing
    if "phase" not in deliveries.columns:
        deliveries["phase"] = deliveries["over"].apply(
            lambda o: "powerplay" if o < 6 else ("death" if o >= 15 else "middle")
        )
    if "is_wicket" not in deliveries.columns:
        deliveries["is_wicket"] = deliveries["dismissal_kind"].notna()

    df, feature_cols = engineer_features(deliveries, matches)
    df = df.dropna(subset=feature_cols + ["intent"])

    le = LabelEncoder()
    le.fit(INTENT_CLASSES)
    df["label"] = le.transform(df["intent"].str.strip())

    X = df[feature_cols].fillna(0).astype(float)
    y = df["label"]

    train_mask = df["season"] < 2024
    test_mask = df["season"] == 2024

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    logger.info("Train: %d, Test: %d", len(X_train), len(X_test))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "max_depth": trial.suggest_int("max_depth", 4, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "num_class": len(INTENT_CLASSES),
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "random_state": 42,
            "tree_method": "hist",
        }
        model = XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        return scores.mean()

    logger.info("Running Optuna for intent model …")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_optuna_trials, show_progress_bar=True)

    best_params = study.best_params | {
        "num_class": len(INTENT_CLASSES),
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "random_state": 42,
        "tree_method": "hist",
    }
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    acc = float((y_pred == y_test.values).mean())
    logger.info("Intent classifier accuracy on 2024 holdout: %.4f", acc)
    logger.info("\n%s", classification_report(y_test, y_pred, target_names=INTENT_CLASSES))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "label_encoder": le,
            "feature_cols": feature_cols,
            "classes": INTENT_CLASSES,
        },
        model_path,
    )
    logger.info("Intent model saved to %s", model_path)
    return {"accuracy": acc, "best_params": best_params}


def derive_serving_features(ball_event: dict[str, Any]) -> dict[str, Any]:
    """
    Compute the small set of model features that cannot be read directly off
    BallEvent and must be derived from other BallEvent fields.

    This is the ONLY place where train/serve feature derivation logic lives.
    If `engineer_features()` is ever changed, this function (and
    TRAIN_TO_SERVING_NAME / LIVE_UNAVAILABLE_DEFAULTS) must be updated to match.
    """
    derived = dict(ball_event)

    # wickets_remaining = 10 - wickets_fallen
    if "wickets_remaining" not in derived or derived.get("wickets_remaining") is None:
        derived["wickets_remaining"] = 10 - ball_event.get("wickets_fallen", 0)

    # cum_balls = balls bowled so far this innings
    if "cum_balls" not in derived or derived.get("cum_balls") in (None, 0):
        derived["cum_balls"] = ball_event.get("over", 0) * 6 + ball_event.get("ball_num", 0)

    # h2h_sr -> already named correctly in BallEvent, default handled below

    # Features genuinely unknowable before the ball is bowled live.
    for feat, default in LIVE_UNAVAILABLE_DEFAULTS.items():
        derived.setdefault(feat, default)

    return derived


def predict_intent(
    ball_event: dict[str, Any],
    model_path: Path = Path("ml/models/intent_classifier.joblib"),
) -> dict[str, Any]:
    """
    Run inference on a single ball event dict.
    Returns: {intent, probabilities, confidence_flag}

    `ball_event` is expected to be a `BallEvent.model_dump()` (api/schemas.py).
    `feature_cols` stored in the model artifact are in BallEvent-canonical
    naming (see TRAIN_TO_SERVING_NAME in engineer_features), so no per-call
    name remapping is required here.
    """
    artifact = joblib.load(model_path)
    model: XGBClassifier = artifact["model"]
    le: LabelEncoder = artifact["label_encoder"]
    feature_cols: list[str] = artifact["feature_cols"]

    serving_event = derive_serving_features(ball_event)

    # Fail loudly (in logs) if the model expects a feature that neither
    # BallEvent nor derive_serving_features can supply, instead of silently
    # zero-filling and producing a garbage/constant prediction.
    missing = [c for c in feature_cols if c not in serving_event]
    if missing:
        logger.warning(
            "predict_intent: feature(s) %s not found in ball_event; "
            "defaulting to 0. This usually indicates a train/serve schema "
            "drift — check TRAIN_TO_SERVING_NAME and BallEvent.",
            missing,
        )

    X = pd.DataFrame([{col: serving_event.get(col, 0) for col in feature_cols}])
    proba = model.predict_proba(X)[0]
    pred_class = int(proba.argmax())
    intent_label = le.inverse_transform([pred_class])[0]

    h2h_balls = ball_event.get("h2h_balls_total", 100)
    confidence_flag = h2h_balls < 30

    return {
        "intent": intent_label,
        "probabilities": {cls: round(float(p), 4) for cls, p in zip(INTENT_CLASSES, proba)},
        "confidence_flag": confidence_flag,
        "h2h_balls": h2h_balls,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = train_intent_classifier()
    print(result)