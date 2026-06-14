"""
tests/test_ingest.py — Unit tests for data ingestion pipeline
"""
import pandas as pd
import pytest
from data.ingest import (
    TEAM_ALIASES,
    label_intent,
    normalise_teams,
)


class TestTeamNormalisation:
    def test_deccan_chargers_mapped(self):
        df = pd.DataFrame({"team1": ["Deccan Chargers"], "team2": ["Mumbai Indians"]})
        result = normalise_teams(df, ["team1"])
        assert result["team1"].iloc[0] == "Sunrisers Hyderabad"

    def test_delhi_daredevils_mapped(self):
        df = pd.DataFrame({"team1": ["Delhi Daredevils"]})
        result = normalise_teams(df, ["team1"])
        assert result["team1"].iloc[0] == "Delhi Capitals"

    def test_kings_xi_punjab_mapped(self):
        df = pd.DataFrame({"team1": ["Kings XI Punjab"]})
        result = normalise_teams(df, ["team1"])
        assert result["team1"].iloc[0] == "Punjab Kings"

    def test_rcb_new_name_mapped(self):
        df = pd.DataFrame({"team1": ["Royal Challengers Bangalore"]})
        result = normalise_teams(df, ["team1"])
        assert result["team1"].iloc[0] == "Royal Challengers Bengaluru"

    def test_unknown_team_preserved(self):
        df = pd.DataFrame({"team1": ["Unknown XI"]})
        result = normalise_teams(df, ["team1"])
        assert result["team1"].iloc[0] == "Unknown XI"

    def test_multiple_columns_normalised(self):
        df = pd.DataFrame({
            "team1": ["Deccan Chargers"],
            "team2": ["Delhi Daredevils"],
            "winner": ["Deccan Chargers"],
        })
        result = normalise_teams(df, ["team1", "team2", "winner"])
        assert result["team1"].iloc[0] == "Sunrisers Hyderabad"
        assert result["team2"].iloc[0] == "Delhi Capitals"
        assert result["winner"].iloc[0] == "Sunrisers Hyderabad"


class TestIntentLabelling:
    def _row(self, batsman_runs=0, extras=0, dismissal_kind=None):
        return pd.Series({
            "batsman_runs": batsman_runs,
            "extras": extras,
            "dismissal_kind": dismissal_kind,
        })

    def test_dismissal_is_pressure_error(self):
        row = self._row(batsman_runs=0, dismissal_kind="bowled")
        assert label_intent(row) == "pressure_error"

    def test_four_is_aggressive(self):
        row = self._row(batsman_runs=4)
        assert label_intent(row) == "aggressive"

    def test_six_is_aggressive(self):
        row = self._row(batsman_runs=6)
        assert label_intent(row) == "aggressive"

    def test_dot_ball_is_defensive(self):
        row = self._row(batsman_runs=0, extras=0)
        assert label_intent(row) == "defensive"

    def test_single_is_neutral(self):
        row = self._row(batsman_runs=1)
        assert label_intent(row) == "neutral"

    def test_three_is_neutral(self):
        row = self._row(batsman_runs=3)
        assert label_intent(row) == "neutral"

    def test_extra_dot_is_defensive(self):
        # extra run but no batsman runs → still defensive by runs_off_bat logic
        # (extras go under 'extras' col, not batsman_runs)
        row = self._row(batsman_runs=0, extras=1)
        assert label_intent(row) == "neutral"  # has extra so not a dot

    def test_dismissal_overrides_runs(self):
        row = self._row(batsman_runs=6, dismissal_kind="caught")
        assert label_intent(row) == "pressure_error"
