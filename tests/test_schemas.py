"""
tests/test_schemas.py — Unit tests for Pydantic API schemas
"""
import pytest
from pydantic import ValidationError
from api.schemas import BallEvent, IntentVerdict, QueryRequest, MetricsResponse


class TestQueryRequest:
    def test_valid(self):
        req = QueryRequest(message="Who won the 2016 IPL?")
        assert req.message == "Who won the 2016 IPL?"
        assert req.session_id == "default"

    def test_custom_session(self):
        req = QueryRequest(message="test", session_id="sess-123")
        assert req.session_id == "sess-123"

    def test_empty_message_allowed(self):
        req = QueryRequest(message="")
        assert req.message == ""


class TestBallEvent:
    def _minimal(self):
        return dict(
            match_id="m001",
            batter="V Kohli",
            bowler="JJ Bumrah",
            non_striker="KL Rahul",
            over=18,
            ball_num=3,
            batting_team="RCB",
            bowling_team="MI",
        )

    def test_valid_minimal(self):
        be = BallEvent(**self._minimal())
        assert be.batter == "V Kohli"
        assert be.over == 18

    def test_defaults(self):
        be = BallEvent(**self._minimal())
        assert be.cum_runs == 0
        assert be.current_rr == 0.0
        assert be.phase_enc == 1

    def test_full_payload(self):
        data = self._minimal()
        data.update(cum_runs=145, wickets_fallen=3, current_rr=9.5,
                    required_rr=12.0, target=180)
        be = BallEvent(**data)
        assert be.cum_runs == 145
        assert be.required_rr == 12.0

    def test_missing_required_field_raises(self):
        data = self._minimal()
        del data["batter"]
        with pytest.raises(ValidationError):
            BallEvent(**data)


class TestIntentVerdict:
    def test_valid(self):
        v = IntentVerdict(
            batter="V Kohli",
            bowler="JJ Bumrah",
            over=18,
            ball_num=3,
            intent="aggressive",
            probabilities={"aggressive": 0.7, "defensive": 0.1, "neutral": 0.1, "pressure_error": 0.1},
            verdict_text="Kohli looks to attack the short ball.",
            latency_ms=245,
        )
        assert v.intent == "aggressive"
        assert v.confidence_flag is False

    def test_confidence_flag_default_false(self):
        v = IntentVerdict(
            batter="X", bowler="Y", over=1, ball_num=1,
            intent="neutral", probabilities={},
            verdict_text="test", latency_ms=100,
        )
        assert v.confidence_flag is False


class TestMetricsResponse:
    def test_defaults_zero(self):
        m = MetricsResponse()
        assert m.faithfulness == 0.0
        assert m.answer_relevancy == 0.0

    def test_set_values(self):
        m = MetricsResponse(faithfulness=0.92, answer_relevancy=0.85,
                            context_precision=0.78, context_recall=0.73,
                            last_evaluated="2025-01-01T00:00:00Z",
                            total_questions=200)
        assert m.faithfulness == 0.92
        assert m.total_questions == 200
