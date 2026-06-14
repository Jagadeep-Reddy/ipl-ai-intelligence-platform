"""
tests/conftest.py — Shared pytest fixtures
"""
import pandas as pd
import pytest


@pytest.fixture
def sample_matches_df():
    """Minimal matches dataframe for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3],
        "season": [2022, 2023, 2024],
        "date": ["2022-04-02", "2023-04-01", "2024-03-22"],
        "team1": ["Mumbai Indians", "Deccan Chargers", "Delhi Daredevils"],
        "team2": ["Chennai Super Kings", "Mumbai Indians", "Chennai Super Kings"],
        "winner": ["Mumbai Indians", "Mumbai Indians", "Chennai Super Kings"],
        "toss_winner": ["Mumbai Indians", "Deccan Chargers", "Delhi Daredevils"],
        "toss_decision": ["bat", "field", "bat"],
        "venue": ["Wankhede Stadium", "Wankhede Stadium", "Arun Jaitley Stadium"],
    })


@pytest.fixture
def sample_deliveries_df():
    """Minimal deliveries dataframe for testing."""
    return pd.DataFrame({
        "match_id": [1, 1, 1, 1],
        "inning": [1, 1, 1, 1],
        "over": [0, 0, 5, 18],
        "ball": [1, 2, 3, 4],
        "batter": ["V Kohli", "V Kohli", "V Kohli", "MS Dhoni"],
        "bowler": ["JJ Bumrah", "JJ Bumrah", "JJ Bumrah", "JJ Bumrah"],
        "non_striker": ["RG Sharma", "RG Sharma", "RG Sharma", "SK Raina"],
        "batsman_runs": [4, 0, 1, 6],
        "extra_runs": [0, 0, 0, 0],
        "total_runs": [4, 0, 1, 6],
        "dismissal_kind": [None, None, None, None],
        "player_dismissed": [None, None, None, None],
        "batting_team": ["MI", "MI", "MI", "CSK"],
        "bowling_team": ["CSK", "CSK", "CSK", "MI"],
    })


@pytest.fixture
def sample_ball_event():
    """Sample BallEvent dict for ML inference tests."""
    return {
        "match_id": "test_001",
        "batter": "V Kohli",
        "bowler": "JJ Bumrah",
        "non_striker": "AB de Villiers",
        "over": 18,
        "ball_num": 3,
        "batting_team": "Royal Challengers Bengaluru",
        "bowling_team": "Mumbai Indians",
        "cum_runs": 156,
        "wickets_fallen": 2,
        "current_rr": 9.75,
        "required_rr": 14.0,
        "target": 185,
        "h2h_balls_total": 45,
        "h2h_sr_global": 128.5,
        "pressure_index": 4.25,
        "phase_enc": 2,
        "inning": 2,
        "season": 2025,
        "over": 18,
        "ball": 3,
        "bowler_last3_runs": 18.0,
        "bowler_economy": 10.5,
        "bowler_spell_balls": 18,
        "batter_last5_avg": 48.0,
        "bowler_type_enc": 0,
        "cum_balls": 111,
        "extra_runs": 0,
        "total_runs": 0,
        "wickets_remaining": 8,
        "required_rr": 14.0,
    }
