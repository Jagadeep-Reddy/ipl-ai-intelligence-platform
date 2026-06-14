"""
locust_test.py
──────────────
Phase 2 Day 14 – Load testing with Locust.
Simulates 200 concurrent users hitting the platform's endpoints.

Run:
  locust -f locust_test.py --headless -u 200 -r 20 -t 60s \
         --host http://localhost:8000

Success criteria (from spec):
  p95 latency < 4s for /query
  p95 latency < 300ms for /predict_intent
  Error rate < 1%
"""

from __future__ import annotations

import json
import random

from locust import HttpUser, between, task


# ── Sample queries distributed across all 5 agent types ─────────────────────

STATS_QUERIES = [
    "Top 5 wicket-takers in powerplay 2023",
    "Rohit Sharma average at Wankhede Stadium",
    "Best economy bowlers in death overs 2024",
    "Total runs scored by Virat Kohli in 2016",
    "Number of centuries in IPL 2022 season",
    "KKR win percentage at Eden Gardens",
    "Which team scored most runs in 2023 powerplays",
]

NARRATIVE_QUERIES = [
    "Tell me about the 2016 IPL season",
    "Describe MS Dhoni's captaincy legacy",
    "What makes Wankhede Stadium special",
    "History of Mumbai Indians vs Chennai Super Kings rivalry",
    "Tell me about Jasprit Bumrah's bowling style",
    "Describe the 2024 IPL final",
]

PREDICTION_QUERIES = [
    "Who will win MI vs CSK at Wankhede?",
    "Predict the outcome of RCB vs GT in Bengaluru",
    "Win probability for SRH vs KKR at Eden Gardens",
    "Which team is likely to win DC vs RR?",
]

MATCHUP_QUERIES = [
    "How does Bumrah bowl to Virat Kohli in death overs?",
    "Rohit Sharma vs Rashid Khan head to head",
    "MS Dhoni vs Bhuvneshwar Kumar matchup",
    "KL Rahul vs Jasprit Bumrah powerplay record",
]

TEAMVSTEAM_QUERIES = [
    "CSK vs MI all-time head-to-head record",
    "GT vs LSG comparison 2023 season",
    "RCB vs RR historical record",
    "KKR vs SRH rivalry statistics",
]

ALL_QUERIES = STATS_QUERIES + NARRATIVE_QUERIES + PREDICTION_QUERIES + MATCHUP_QUERIES + TEAMVSTEAM_QUERIES


# ── Ball events for /predict_intent ─────────────────────────────────────────

BATTERS = ["V Kohli", "RG Sharma", "MS Dhoni", "KL Rahul", "DA Warner", "AB de Villiers"]
BOWLERS = ["JJ Bumrah", "YS Chahal", "RA Jadeja", "Rashid Khan", "B Kumar", "PP Chawla"]


def random_ball_event() -> dict:
    over = random.randint(0, 19)
    return {
        "match_id": f"load_test_{random.randint(1, 100)}",
        "batter": random.choice(BATTERS),
        "bowler": random.choice(BOWLERS),
        "non_striker": random.choice(BATTERS),
        "over": over,
        "ball_num": random.randint(1, 6),
        "batting_team": "Mumbai Indians",
        "bowling_team": "Chennai Super Kings",
        "cum_runs": random.randint(0, 200),
        "wickets_fallen": random.randint(0, 9),
        "current_rr": round(random.uniform(6.0, 14.0), 2),
        "required_rr": round(random.uniform(6.0, 18.0), 2),
        "target": random.randint(140, 220),
        "h2h_balls_total": random.randint(5, 120),
        "h2h_sr_global": round(random.uniform(80.0, 180.0), 1),
        "pressure_index": round(random.uniform(0.0, 8.0), 2),
        "phase_enc": 0 if over < 6 else (2 if over >= 15 else 1),
        "inning": random.choice([1, 2]),
        "season": 2025,
        "ball": random.randint(1, 6),
        "bowler_last3_runs": round(random.uniform(0.0, 24.0), 1),
        "bowler_economy": round(random.uniform(5.0, 14.0), 2),
        "bowler_spell_balls": random.randint(1, 24),
        "batter_last5_avg": round(random.uniform(10.0, 60.0), 1),
        "bowler_type_enc": random.randint(0, 1),
        "cum_balls": random.randint(1, 120),
        "extra_runs": 0,
        "total_runs": random.randint(0, 6),
        "wickets_remaining": random.randint(1, 10),
    }


# ── Locust User ───────────────────────────────────────────────────────────────

class IPLPlatformUser(HttpUser):
    """
    Simulates a mixed-load user:
      60% — historical Q&A queries (SSE endpoint)
      30% — single-ball intent predictions (REST)
      10% — health checks
    """
    wait_time = between(1, 3)  # 1-3 second think time between tasks

    @task(6)
    def query_chat(self):
        """POST /query — historical Q&A with SSE streaming."""
        query = random.choice(ALL_QUERIES)
        payload = {
            "message": query,
            "session_id": f"load_user_{self.environment.runner.user_count}",
        }
        # Use stream=False to measure full response time
        with self.client.post(
            "/query",
            json=payload,
            headers={"Accept": "text/event-stream"},
            stream=True,
            catch_response=True,
            timeout=10,
        ) as response:
            if response.status_code == 200:
                # Read and discard the stream
                content = b""
                for chunk in response.iter_content(chunk_size=1024):
                    content += chunk
                if len(content) > 0:
                    response.success()
                else:
                    response.failure("Empty SSE stream received")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(3)
    def predict_intent(self):
        """POST /predict_intent — per-ball intent prediction."""
        payload = random_ball_event()
        with self.client.post(
            "/predict_intent",
            json=payload,
            catch_response=True,
            timeout=2,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "intent" in data and data["intent"] in (
                    "aggressive", "defensive", "neutral", "pressure_error"
                ):
                    response.success()
                else:
                    response.failure("Invalid intent in response")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def health_check(self):
        """GET /health — liveness probe."""
        with self.client.get("/health", catch_response=True, timeout=2) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok":
                    response.success()
                else:
                    response.failure("Status not ok")
            else:
                response.failure(f"HTTP {response.status_code}")
