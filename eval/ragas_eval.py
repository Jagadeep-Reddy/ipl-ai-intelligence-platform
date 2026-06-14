"""
ml/train_win_prob.py
────────────────────
Phase 1 Days 6-7 – Train XGBoost Model A: pre-match win probability.
Features: h2h win%, venue win%, avg first-inn score, last-5 form,
          toss factor, squad strength index.
Target:   match winner (binary per team)
Metric:   AUC > 0.72 on 2024 holdout
"""
 
from __future__ import annotations
 
import logging
import os
from pathlib import Path
 
import joblib
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
 
logger = logging.getLogger(__name__)
 
 
def build_match_features(engine) -> pd.DataFrame:
    """
    Merge matches with pre-computed aggregates to build one row per match
    with all features needed for win probability prediction.
    """
    sql = """
    SELECT
        m.id, m.season, m.venue, m.team1, m.team2,
        m.toss_winner, m.toss_decision, m.winner,
        -- H2H win percentage (team1 perspective)
        COALESCE(h.team1_wins::float / NULLIF(h.total, 0), 0.5) AS h2h_win_pct,
        -- Venue win-batting-first
        COALESCE(v.win_bat_first_pct, 50.0) / 100.0             AS venue_bat_first_pct,
        -- Avg first innings at venue
        COALESCE(v.avg_first_inn, 160.0)                         AS venue_avg_score,
        -- Team1 form rating
        COALESCE(f1.form_rating, 50.0) / 100.0                   AS team1_form,
        -- Team2 form rating
        COALESCE(f2.form_rating, 50.0) / 100.0                   AS team2_form,
        -- Toss advantage: 1 if team1 won toss else 0
        CASE WHEN m.toss_winner = m.team1 THEN 1 ELSE 0 END     AS team1_toss,
        -- Squad strength
        COALESCE(s1.strength_index, 50.0) / 100.0               AS team1_strength,
        COALESCE(s2.strength_index, 50.0) / 100.0               AS team2_strength
    FROM matches m
    LEFT JOIN (
        SELECT team1, team2,
               SUM(CASE WHEN winner=team1 THEN 1 ELSE 0 END) AS team1_wins,
               COUNT(*) AS total
        FROM matches GROUP BY team1, team2
    ) h ON h.team1=m.team1 AND h.team2=m.team2
    LEFT JOIN venue_stats v ON v.venue=m.venue AND v.season=m.season
    LEFT JOIN team_form f1 ON f1.team=m.team1 AND f1.season=m.season
    LEFT JOIN team_form f2 ON f2.team=m.team2 AND f2.season=m.season
    LEFT JOIN squad_strength s1 ON s1.team=m.team1 AND s1.season=m.season
    LEFT JOIN squad_strength s2 ON s2.team=m.team2 AND s2.season=m.season
    WHERE m.winner IS NOT NULL
    """
    df = pd.read_sql(sql, engine)
    df["target"] = (df["winner"] == df["team1"]).astype(int)
    return df
 
 
FEATURE_COLS = [
    "h2h_win_pct",
    "venue_bat_first_pct",
    "venue_avg_score",
    "team1_form",
    "team2_form",
    "team1_toss",
    "team1_strength",
    "team2_strength",
]
 
 
def train_win_probability_model(
    db_url: str,
    model_path: Path = Path("ml/models/win_probability.joblib"),
    n_optuna_trials: int = 100,
) -> dict:
    from sqlalchemy import create_engine
    engine = create_engine(db_url)
    df = build_match_features(engine)
 
    X = df[FEATURE_COLS].fillna(0.5)
    y = df["target"]
 
    train_mask = df["season"] < 2024
    test_mask = df["season"] == 2024
 
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
 
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "eval_metric": "auc",
            "random_state": 42,
        }
        model = XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        return scores.mean()
 
    logger.info("Running Optuna with %d trials …", n_optuna_trials)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_optuna_trials, show_progress_bar=True)
 
    best_params = study.best_params | {"eval_metric": "auc", "random_state": 42}
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)
 
    # Calibrate probabilities
    cal_model = CalibratedClassifierCV(model, cv=5, method="isotonic")
    cal_model.fit(X_train, y_train)
 
    from sklearn.metrics import roc_auc_score
    y_pred_proba = cal_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    logger.info("Win Probability Model AUC on 2024 holdout: %.4f", auc)
 
    # SHAP explainability
    explainer = shap.TreeExplainer(model)
 
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": cal_model, "explainer": explainer, "feature_cols": FEATURE_COLS},
        model_path,
    )
    logger.info("Model saved to %s", model_path)
 
    return {"auc": auc, "best_params": best_params}
 
 
def predict_win_probability(
    team1: str,
    team2: str,
    venue: str,
    season: int,
    toss_winner: str,
    toss_decision: str,
    db_url: str,
    model_path: Path = Path("ml/models/win_probability.joblib"),
) -> dict:
    """Return win probability + SHAP-based explanation for team1."""
    from sqlalchemy import create_engine
    engine = create_engine(db_url)
    artifact = joblib.load(model_path)
    model = artifact["model"]
    explainer = artifact["explainer"]
 
    # Build single-row feature vector
    venue_row = pd.read_sql(
        f"SELECT * FROM venue_stats WHERE venue='{venue}' AND season={season}", engine
    )
    form_t1 = pd.read_sql(
        f"SELECT form_rating FROM team_form WHERE team='{team1}' AND season={season}", engine
    )
    form_t2 = pd.read_sql(
        f"SELECT form_rating FROM team_form WHERE team='{team2}' AND season={season}", engine
    )
 
    features = {
        "h2h_win_pct": 0.5,
        "venue_bat_first_pct": float(venue_row["win_bat_first_pct"].iloc[0]) / 100 if len(venue_row) else 0.5,
        "venue_avg_score": float(venue_row["avg_first_inn"].iloc[0]) if len(venue_row) else 160.0,
        "team1_form": float(form_t1["form_rating"].iloc[0]) / 100 if len(form_t1) else 0.5,
        "team2_form": float(form_t2["form_rating"].iloc[0]) / 100 if len(form_t2) else 0.5,
        "team1_toss": 1 if toss_winner == team1 else 0,
        "team1_strength": 0.5,
        "team2_strength": 0.5,
    }
    X = pd.DataFrame([features])[FEATURE_COLS]
    prob = float(model.predict_proba(X)[0][1])
 
    # SHAP top-3 for natural language explanation
    base_model = model.estimator if hasattr(model, "estimator") else model
    try:
        sv = explainer.shap_values(X)[0]
        top3 = sorted(zip(FEATURE_COLS, sv), key=lambda x: abs(x[1]), reverse=True)[:3]
        shap_explanation = [
            {"feature": f, "direction": "increases" if v > 0 else "decreases", "magnitude": round(float(abs(v)), 3)}
            for f, v in top3
        ]
    except Exception:
        shap_explanation = []
 
    return {
        "team1": team1,
        "team2": team2,
        "team1_win_prob": round(prob, 4),
        "team2_win_prob": round(1 - prob, 4),
        "shap_explanation": shap_explanation,
    }
 
 
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = train_win_probability_model(os.environ["DATABASE_URL"])
    print(result)