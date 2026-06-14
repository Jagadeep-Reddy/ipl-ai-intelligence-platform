"""
rag/doc_generator.py
────────────────────
Phase 1 Day 3 – Generate 200+ rich narrative documents for the RAG
knowledge base. Supports two modes:

  LLM mode (default):   Uses Azure OpenAI gpt-4.1-mini to write narrative text.
  Template mode:        Falls back to pure Jinja2 templates with no LLM call.
                        Activated when USE_LLM_FOR_DOCS=false in .env.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd
from jinja2 import Environment, BaseLoader

logger = logging.getLogger(__name__)

# ── Detect LLM mode ───────────────────────────────────────────────────────────
_USE_LLM = os.environ.get("USE_LLM_FOR_DOCS", "true").lower() != "false"

if _USE_LLM:
    logger.info("doc_generator: LLM mode (Azure gpt-4.1-mini)")
else:
    logger.info("doc_generator: Template-only mode (no LLM cost)")

# ── Jinja2 prompt templates (used by LLM) ────────────────────────────────────

SEASON_PROMPT = """
You are an expert cricket journalist. Write a rich 400-600 word narrative summary of the {{ season }} IPL season.
Use ONLY the facts provided below. Do not invent statistics.

Facts:
- Champion: {{ champion }}
- Runner-up: {{ runner_up }}
- Top scorer: {{ top_scorer }} ({{ top_runs }} runs)
- Top wicket-taker: {{ top_bowler }} ({{ top_wickets }} wickets)
- Total matches: {{ total_matches }}
- Average first innings score: {{ avg_score }}

Write flowing prose covering:
1. The title-winning team's journey and what made them champions
2. The runner-up's campaign and how they reached the final
3. Standout individual performances (top batter and bowler)
4. What made this season distinctive
5. End with the title-winning moment

Be specific about the champion {{ champion }} defeating {{ runner_up }} in the final.
"""

PLAYER_PROMPT = """
You are an expert IPL cricket analyst. Write a 400-600 word profile for {{ player_name }} who plays for {{ team }} as a {{ role }}.
Use only the statistics below. Do not invent facts.

Career stats across phases:
Batting:
{% for row in batting_stats %}
  Season {{ row.season }}, Phase {{ row.phase }}: {{ row.runs }} runs, SR {{ row.strike_rate }}, Avg {{ row.average }}
{% endfor %}

Bowling:
{% for row in bowling_stats %}
  Season {{ row.season }}, Phase {{ row.phase }}: {{ row.wickets }} wkts, Economy {{ row.economy }}
{% endfor %}

Key head-to-head records (as batter vs top bowlers):
{% for h in h2h %}
  vs {{ h.bowler }} ({{ h.phase }}): {{ h.balls }} balls, SR {{ h.strike_rate }}, {{ h.wickets }} dismissals
{% endfor %}

Write a detailed profile covering:
1. Playing style and role in the team for {{ team }}
2. Phase-wise strengths (powerplay/middle/death overs) with specific numbers
3. Career trajectory and best seasons
4. Notable head-to-head battles
5. Overall IPL legacy and impact
6. If they have captained a team, mention their captaincy record and titles won
"""

VENUE_PROMPT = """
Write a 300-400 word venue analysis for {{ venue }}.

Stats:
- Average first innings score: {{ avg_score }}
- Win % batting first: {{ bat_first_win }}%
- Avg run rates – Powerplay {{ pp_rr }}, Middle {{ mid_rr }}, Death {{ death_rr }}

Cover:
1. Pitch characteristics (pace/spin friendly, true/slow)
2. Whether teams prefer batting or chasing here (based on {{ bat_first_win }}% bat-first win rate)
3. Typical match patterns and scoring trends
4. Which playing styles thrive here
5. Dew factor and day/night considerations if relevant
"""

# ── Jinja2 no-LLM output templates ───────────────────────────────────────────

SEASON_TEXT = """
IPL {{ season }} Season Summary
================================
Champion: {{ champion }}
Runner-up: {{ runner_up }}
Total Matches: {{ total_matches }}
Average First Innings Score: {{ avg_score }}

Top Batter: {{ top_scorer }} — {{ top_runs }} runs across the season.
Top Bowler: {{ top_bowler }} — {{ top_wickets }} wickets across the season.

{{ champion }} won the IPL {{ season }} title, defeating {{ runner_up }} in the final to claim the championship.
{{ champion }} finished the season as the strongest side, winning key matches throughout the tournament.
{{ top_scorer }} was the standout batter of the tournament, accumulating {{ top_runs }} runs.
With the ball, {{ top_bowler }} led the wicket-takers chart with {{ top_wickets }} dismissals.
Across {{ total_matches }} matches the average first innings total was {{ avg_score }} runs.
The {{ season }} IPL season saw {{ champion }} lift the trophy after a memorable campaign,
with {{ runner_up }} finishing as runners-up after a competitive final.
"""

PLAYER_TEXT = """
Player Profile: {{ player_name }}
===================================
Team: {{ team }}
Primary Role: {{ role }}

Batting Record by Phase and Season:
{% for row in batting_stats %}
  [{{ row.season }}] {{ row.phase | title }}: {{ row.runs }} runs | SR {{ row.strike_rate }} | Avg {{ row.average }}
{% endfor %}

Bowling Record by Phase and Season:
{% for row in bowling_stats %}
  [{{ row.season }}] {{ row.phase | title }}: {{ row.wickets }} wickets | Economy {{ row.economy }}
{% endfor %}

Head-to-Head Highlights (batting):
{% for h in h2h %}
  vs {{ h.bowler }} ({{ h.phase }}): {{ h.balls }} balls faced | SR {{ h.strike_rate }} | dismissed {{ h.wickets }} times
{% endfor %}

Career Narrative:
{{ player_name }} is a prominent IPL {{ role }}, best known for representing {{ team }}.
{% if role == 'batsman' -%}
As a specialist batter, {{ player_name }} has been a consistent run-scorer across powerplay, middle overs, and death overs over multiple IPL seasons.
{{ player_name }} is known for match-winning innings and has been one of the key batters in IPL history.
{%- elif role == 'bowler' -%}
As a specialist bowler for {{ team }}, {{ player_name }} has been a key wicket-taker and match-winner across powerplay, middle overs, and death overs.
{{ player_name }} has played a crucial role in {{ team }}'s bowling attack across multiple IPL seasons.
{%- else -%}
{{ player_name }} has been a valuable all-rounder for {{ team }}, contributing both with the bat and ball across multiple IPL seasons.
{%- endif %}
{{ player_name }} has played a crucial role in IPL history and is considered one of the standout performers in the league's history.
"""

VENUE_TEXT = """
Venue Profile: {{ venue }}
===========================
Average First Innings Score : {{ avg_score }}
Win % Batting First          : {{ bat_first_win }}%
Powerplay Run Rate           : {{ pp_rr }} per over
Middle Overs Run Rate        : {{ mid_rr }} per over
Death Overs Run Rate         : {{ death_rr }} per over

{{ venue }} is one of the prominent IPL venues hosting major matches across seasons.
Teams batting first score an average of {{ avg_score }} runs here.
{% if bat_first_win | float > 50 %}
With a {{ bat_first_win }}% win rate for teams batting first, {{ venue }} historically favours sides that set a target.
Teams that bat well in the powerplay and build a strong total tend to win here.
The pitch tends to slow down as the match progresses, making chasing difficult.
{% else %}
With only a {{ bat_first_win }}% win rate for teams batting first, {{ venue }} strongly favours chasing teams.
The pitch generally gets better for batting as the match progresses, and dew can assist chasing teams.
Captains winning the toss typically choose to field first at {{ venue }}.
{% endif %}
The powerplay yields {{ pp_rr }} runs per over on average, while the death overs accelerate to {{ death_rr }} per over.
Spinners and pace bowlers who can vary their lengths tend to perform well at this venue.
"""

TEAM_TEXT = """
Team Profile: {{ team }}
=========================
Seasons Played : {{ seasons }}
Titles Won     : {{ titles }}
Total Wins     : {{ wins }}
Total Losses   : {{ losses }}
Top Players    : {{ top_players }}

{{ team }} has been one of the IPL franchises since the tournament's inception.
Over {{ seasons }} seasons they have registered {{ wins }} wins against
{{ losses }} losses, claiming {{ titles }} title(s). Their squad has featured
outstanding players including {{ top_players }}.
"""


def _render(template_str: str, **kwargs) -> str:
    env = Environment(loader=BaseLoader())
    return env.from_string(template_str).render(**kwargs)


def _call_llm(prompt: str, max_tokens: int = 600) -> str:
    """Call Azure OpenAI for LLM-generated narrative text."""
    from openai import AzureOpenAI
    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )
    response = client.chat.completions.create(
        model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini"),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def _generate_text(prompt_template: str, text_template: str,
                   max_tokens: int = 600, **kwargs) -> str:
    """Use LLM if available, otherwise fall back to plain template."""
    if _USE_LLM:
        try:
            prompt = _render(prompt_template, **kwargs)
            return _call_llm(prompt, max_tokens=max_tokens)
        except Exception as e:
            logger.warning("LLM call failed (%s), falling back to template.", e)
    return _render(text_template, **kwargs)


# ── Document generators ───────────────────────────────────────────────────────

def generate_season_docs(engine, out_dir: Path) -> list[dict]:
    docs = []
    seasons_df = pd.read_sql(
        "SELECT DISTINCT season FROM matches WHERE season IS NOT NULL ORDER BY season",
        engine,
    )
    for season in seasons_df["season"]:
        season_matches = pd.read_sql(
            f"SELECT * FROM matches WHERE season={int(season)}", engine
        )
        winner_mode = season_matches["winner"].dropna().mode()
        champion = winner_mode.iloc[0] if len(winner_mode) else "Unknown"

        # Determine runner-up from the season's Final match
        runner_up = "Unknown"
        final_matches = season_matches[season_matches["match_type"] == "Final"]
        if len(final_matches):
            final_row = final_matches.iloc[0]
            for team in (final_row["team1"], final_row["team2"]):
                if team != champion:
                    runner_up = team
                    break

        top_scorer_row = pd.read_sql(
            f"""SELECT player, SUM(runs) as total FROM player_career_stats
                WHERE season={int(season)} AND role='batsman'
                GROUP BY player ORDER BY total DESC LIMIT 1""",
            engine,
        )
        top_bowler_row = pd.read_sql(
            f"""SELECT player, SUM(wickets) as total FROM player_career_stats
                WHERE season={int(season)} AND role='bowler'
                GROUP BY player ORDER BY total DESC LIMIT 1""",
            engine,
        )
        avg_score_row = pd.read_sql(
            f"""SELECT ROUND(AVG(avg_first_inn), 1) as avg_score
                FROM venue_stats WHERE season={int(season)}""",
            engine,
        )
        avg_score = float(avg_score_row["avg_score"].iloc[0] or 0)

        kwargs = dict(
            season=int(season),
            champion=champion,
            runner_up=runner_up,
            top_scorer=top_scorer_row["player"].iloc[0] if len(top_scorer_row) else "N/A",
            top_runs=int(top_scorer_row["total"].iloc[0]) if len(top_scorer_row) else 0,
            top_bowler=top_bowler_row["player"].iloc[0] if len(top_bowler_row) else "N/A",
            top_wickets=int(top_bowler_row["total"].iloc[0]) if len(top_bowler_row) else 0,
            total_matches=len(season_matches),
            avg_score=avg_score,
        )
        text = _generate_text(SEASON_PROMPT, SEASON_TEXT, max_tokens=700, **kwargs)
        docs.append({
            "doc_id": f"season_{season}",
            "doc_type": "season_summary",
            "season": int(season),
            "team": None,
            "player": None,
            "venue": None,
            "text": text,
        })
        logger.info("Generated season doc: %s (champion=%s, runner_up=%s)", season, champion, runner_up)
    return docs


def generate_player_docs(engine, top_n: int = 100) -> list[dict]:
    top_players = pd.read_sql(
        f"""SELECT player FROM player_career_stats WHERE role='batsman'
            GROUP BY player ORDER BY SUM(runs) DESC LIMIT {top_n}""",
        engine,
    )
    docs = []
    for player in top_players["player"]:
        safe = player.replace("'", "''")
        batting = pd.read_sql(
            f"SELECT * FROM player_career_stats WHERE player='{safe}' AND role='batsman'",
            engine,
        ).to_dict("records")
        bowling = pd.read_sql(
            f"SELECT * FROM player_career_stats WHERE player='{safe}' AND role='bowler'",
            engine,
        ).to_dict("records")
        h2h = pd.read_sql(
            f"SELECT * FROM h2h_records WHERE batter='{safe}' ORDER BY balls DESC LIMIT 5",
            engine,
        ).to_dict("records")

        # Determine primary team (most balls faced) and role
        team_row = pd.read_sql(
            f"""SELECT batting_team, COUNT(*) as cnt
                FROM deliveries WHERE batter='{safe}'
                GROUP BY batting_team ORDER BY cnt DESC LIMIT 1""",
            engine,
        )
        primary_team = team_row["batting_team"].iloc[0] if len(team_row) else "IPL"
        primary_role = "batsman" if batting else ("bowler" if bowling else "all-rounder")

        # Clean NaN values from stats before rendering
        def clean_stat(val):
            try:
                f = float(val)
                return "N/A" if (f != f) else round(f, 2)  # NaN check
            except (TypeError, ValueError):
                return val

        batting_clean = [{k: clean_stat(v) if k in ('strike_rate', 'average', 'economy') else v
                          for k, v in row.items()} for row in batting]
        bowling_clean = [{k: clean_stat(v) if k in ('strike_rate', 'average', 'economy') else v
                          for k, v in row.items()} for row in bowling]
        h2h_clean = [{k: clean_stat(v) if k in ('strike_rate', 'average', 'economy') else v
                      for k, v in row.items()} for row in h2h]

        kwargs = dict(
            player_name=player,
            team=primary_team,
            role=primary_role,
            batting_stats=batting_clean,
            bowling_stats=bowling_clean,
            h2h=h2h_clean,
        )
        text = _generate_text(PLAYER_PROMPT, PLAYER_TEXT, max_tokens=700, **kwargs)
        docs.append({
            "doc_id": f"player_{player.replace(' ', '_')}",
            "doc_type": "player_profile",
            "season": None,
            "team": primary_team,
            "player": player,
            "venue": None,
            "text": text,
        })
        logger.info("Generated player doc: %s (%s, %s)", player, primary_team, primary_role)
    return docs


def generate_venue_docs(engine) -> list[dict]:
    venues = pd.read_sql(
        "SELECT DISTINCT venue FROM venue_stats WHERE venue IS NOT NULL ORDER BY venue",
        engine,
    )
    docs = []
    for venue in venues["venue"]:
        safe = venue.replace("'", "''")
        stats = pd.read_sql(
            f"""SELECT AVG(avg_first_inn) as avg_score,
                       AVG(win_bat_first_pct) as bat_first_win,
                       AVG(avg_powerplay_rr) as pp_rr,
                       AVG(avg_middle_rr) as mid_rr,
                       AVG(avg_death_rr) as death_rr
                FROM venue_stats WHERE venue='{safe}'""",
            engine,
        ).iloc[0]
        kwargs = dict(
            venue=venue,
            avg_score=round(float(stats.avg_score or 0), 1),
            bat_first_win=min(round(float(stats.bat_first_win or 0), 1), 100.0),
            pp_rr=round(float(stats.pp_rr or 0), 2),
            mid_rr=round(float(stats.mid_rr or 0), 2),
            death_rr=round(float(stats.death_rr or 0), 2),
        )
        text = _generate_text(VENUE_PROMPT, VENUE_TEXT, max_tokens=400, **kwargs)
        docs.append({
            "doc_id": f"venue_{venue.replace(' ', '_').replace(',', '')}",
            "doc_type": "venue_analysis",
            "season": None,
            "team": None,
            "player": None,
            "venue": venue,
            "text": text,
        })
        logger.info("Generated venue doc: %s", venue)
    return docs


def save_docs(docs: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d documents to %s", len(docs), out_path)


def generate_all(
    db_url: str,
    out_path: Path = Path("rag/documents.json"),
) -> list[dict]:
    from sqlalchemy import create_engine
    engine = create_engine(db_url)
    all_docs: list[dict] = []
    all_docs += generate_season_docs(engine, Path("rag"))
    all_docs += generate_player_docs(engine, top_n=100)
    all_docs += generate_venue_docs(engine)
    save_docs(all_docs, out_path)
    return all_docs


if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)
    generate_all(os.environ["DATABASE_URL"])