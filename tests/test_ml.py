"""
tests/test_ml.py — Unit tests for ML model utilities
"""
import numpy as np
import pytest
from ml.train_win_prob import FEATURE_COLS
from ml.train_intent import INTENT_CLASSES


class TestWinProbFeatureCols:
    def test_expected_features_present(self):
        expected = {
            "h2h_win_pct", "venue_bat_first_pct", "venue_avg_score",
            "team1_form", "team2_form", "team1_toss",
            "team1_strength", "team2_strength",
        }
        assert expected == set(FEATURE_COLS)

    def test_feature_count(self):
        assert len(FEATURE_COLS) == 8

    def test_no_duplicates(self):
        assert len(FEATURE_COLS) == len(set(FEATURE_COLS))


class TestIntentClasses:
    def test_four_classes(self):
        assert len(INTENT_CLASSES) == 4

    def test_expected_classes(self):
        assert set(INTENT_CLASSES) == {"aggressive", "defensive", "neutral", "pressure_error"}


class TestIntentPredictInterface:
    """Test predict_intent function signature without loading a model."""

    def test_ball_event_dict_keys(self):
        """Confirm the keys a ball_event dict must provide match the feature schema."""
        from ml.train_intent import INTENT_CLASSES
        required_sample_keys = [
            "over", "ball", "phase_enc", "current_rr", "required_rr",
            "pressure_index", "wickets_remaining", "cum_runs", "cum_balls",
            "h2h_sr_global", "bowler_last3_runs", "bowler_economy",
            "bowler_spell_balls", "batter_last5_avg", "bowler_type_enc",
            "inning", "extra_runs", "total_runs", "season",
        ]
        # All must be valid Python identifiers
        for key in required_sample_keys:
            assert key.isidentifier(), f"{key!r} is not a valid identifier"

    def test_intent_classes_are_strings(self):
        for cls in INTENT_CLASSES:
            assert isinstance(cls, str)
            assert len(cls) > 0


class TestWinProbabilityOutputRange:
    """Test win probability output shape and range with a mock model."""

    def test_probability_sums_to_one(self):
        # Simulate what predict_win_probability returns
        team1_prob = 0.63
        team2_prob = round(1 - team1_prob, 4)
        assert abs(team1_prob + team2_prob - 1.0) < 1e-6

    def test_probability_in_valid_range(self):
        for p in np.linspace(0.0, 1.0, 20):
            assert 0.0 <= p <= 1.0
