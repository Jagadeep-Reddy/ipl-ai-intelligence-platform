"""
agents/router.py
────────────────
Phase 1 Days 5-7 – LangGraph conditional query router + all 5 specialist agents.
  StatsQA → Text-to-SQL via LangChain SQL agent
  NarrativeQA → Hybrid RAG pipeline
  Prediction → XGBoost win probability + SHAP + LLM narrative
  Matchup → H2H SQL + RAG enrichment
  TeamVsTeam → Team comparison SQL + form rating
"""
 
from __future__ import annotations
 
import json
import logging
import os
from typing import Any, Literal, TypedDict
 
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
 
logger = logging.getLogger(__name__)
 
QueryType = Literal["STATS", "NARRATIVE", "PREDICTION", "MATCHUP", "TEAMVSTEAM"]


def _to_native(obj: Any) -> Any:
    """Convert numpy scalar types (float32, int64, etc.) to native Python types for json.dumps."""
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
 
 
# ── Graph state ───────────────────────────────────────────────────────────────
 
class AgentState(TypedDict):
    query: str
    query_type: QueryType | None
    context: list[dict]            # retrieved RAG docs
    sql_result: str | None
    ml_result: dict | None
    answer: str | None
    confidence: float
    backend_used: str | None
    session_id: str
    history: list[dict]            # last 5 turns
 
 
# ── LLM instances ─────────────────────────────────────────────────────────────
 
def get_llm(temperature: float = 0.1) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini"),
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        temperature=temperature,
        streaming=True,
    )
 
 
# ── Router node ───────────────────────────────────────────────────────────────
 
ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query classifier for an IPL cricket analytics platform.
Classify the user query into exactly one of these categories:
 
STATS      – Requires precise numbers, aggregations, rankings, or filter on structured data.
             Examples: "Top 5 wicket-takers in powerplay 2023", "Rohit Sharma's average at Wankhede"
 
NARRATIVE  – Requires story, history, biography, or contextual narrative about a season, player, team, or venue.
             Examples: "Tell me about IPL 2016", "Describe MS Dhoni's captaincy", "What makes Wankhede special"
 
PREDICTION – Requires win probability or match outcome prediction.
             Examples: "Who will win MI vs CSK?", "Predict the outcome of RCB vs GT"
 
MATCHUP    – Requires head-to-head analysis between a specific batsman and bowler.
             Examples: "How does Bumrah bowl to Kohli?", "Rohit vs Rashid Khan in death overs"
 
TEAMVSTEAM – Requires all-time or season-specific comparison between two teams.
             Examples: "CSK vs MI head-to-head", "GT vs LSG comparison 2023"
 
Reply with ONLY the category word. No explanation."""),
    ("human", "Query: {query}")
])
 
 
def route_query(state: AgentState) -> AgentState:
    llm = get_llm()
    chain = ROUTER_PROMPT | llm
    result = chain.invoke({"query": state["query"]})
    raw = result.content.strip().upper()
 
    valid = {"STATS", "NARRATIVE", "PREDICTION", "MATCHUP", "TEAMVSTEAM"}
    query_type = raw if raw in valid else "NARRATIVE"  # safe fallback
    logger.info("Query type: %s for '%s'", query_type, state["query"][:60])
    return {**state, "query_type": query_type}
 
 
def decide_next(state: AgentState) -> str:
    return state["query_type"].lower()
 
 
# ── Agent: StatsQA (Text-to-SQL) ─────────────────────────────────────────────
 
SQL_EXAMPLES = """
-- Example 1: Top 5 wicket-takers in powerplay phase in 2023
SELECT bowler, SUM(wickets) as total_wickets
FROM player_career_stats
WHERE phase = 'powerplay' AND season = 2023 AND role = 'bowler'
GROUP BY bowler ORDER BY total_wickets DESC LIMIT 5;
 
-- Example 2: Player strike rate at a specific venue
SELECT p.player, p.strike_rate
FROM player_career_stats p
JOIN matches m ON m.season = p.season
WHERE m.venue = 'Wankhede Stadium' AND p.player = 'RG Sharma' AND p.role = 'batsman'
GROUP BY p.player, p.strike_rate LIMIT 1;
 
-- Example 3: H2H between batsman and bowler across all phases
SELECT batter, bowler, phase, balls, strike_rate, wickets
FROM h2h_records WHERE batter = 'V Kohli' AND bowler = 'JJ Bumrah'
ORDER BY balls DESC;
 
-- Example 4: Team win percentage at a venue
SELECT v.venue, v.win_bat_first_pct FROM venue_stats v
WHERE v.venue = 'MA Chidambaram Stadium' ORDER BY v.season DESC LIMIT 1;
 
-- Example 5: Best economy bowlers in death overs 2024
SELECT player, economy FROM player_career_stats
WHERE phase = 'death' AND season = 2024 AND role = 'bowler' AND innings >= 5
ORDER BY economy ASC LIMIT 10;
"""
 
 
def stats_agent_node(state: AgentState, db_url: str) -> AgentState:
    db = SQLDatabase.from_uri(db_url)
    llm = get_llm()
    agent = create_sql_agent(
        llm=llm,
        db=db,
        verbose=False,
        agent_type="tool-calling",
        handle_parsing_errors=True,
        prefix=(
            "You are an expert IPL cricket statistician. Use the provided SQL examples to guide "
            "accurate query generation. Always execute the query and include the actual returned "
            "rows/values in your final answer, followed by a brief plain-English explanation. "
            "If the query returns zero rows, say so explicitly. "
            f"\n\nSQL EXAMPLES:\n{SQL_EXAMPLES}"
        ),
    )
    try:
        result = agent.invoke({"input": state["query"]})
        answer = result.get("output", "No result found.")
        return {**state, "answer": answer, "backend_used": "SQL", "confidence": 0.92}
    except Exception as e:
        logger.exception("SQL agent error for query: %s", state["query"])
        return {**state, "answer": f"Query could not be answered via SQL: {e}", "backend_used": "SQL", "confidence": 0.3}
 
 
# ── Agent: NarrativeQA (RAG) ──────────────────────────────────────────────────
 
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert IPL cricket analyst with deep knowledge of 18 seasons of IPL history.
Answer the question using ONLY the context documents provided below.
Be specific, factual, and cite which document supports each claim.
If the context does not contain enough information, say so explicitly.
 
Context documents:
{context}"""),
    ("human", "{query}")
])
 
 
def narrative_agent_node(state: AgentState, retriever) -> AgentState:
    docs = retriever.retrieve(state["query"])
    context_text = "\n\n---\n\n".join(
        f"[{d.get('doc_type','doc').upper()} – {d.get('doc_id','')}]\n{d['text']}"
        for d in docs
    )
    llm = get_llm()
    chain = RAG_PROMPT | llm
    result = chain.invoke({"context": context_text, "query": state["query"]})
    return {
        **state,
        "context": docs,
        "answer": result.content,
        "backend_used": "RAG",
        "confidence": 0.85,
    }
 
 
# ── Agent: Prediction ─────────────────────────────────────────────────────────
 
PREDICTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an IPL win probability analyst. You have been given a machine learning
prediction result with SHAP feature explanations. Generate a concise, insightful match preview
in 150-200 words covering:
1. The predicted winner and probability
2. The key factors driving the prediction (from SHAP)
3. One risk factor that could change the outcome
Use cricket-specific language. Be confident but acknowledge uncertainty."""),
    ("human", "Prediction data:\n{prediction_json}\n\nUser query: {query}")
])
 
 
def prediction_agent_node(state: AgentState, db_url: str) -> AgentState:
    from ml.train_win_prob import predict_win_probability
 
    # Extract teams from query using LLM
    extract_llm = get_llm()
    extract_prompt = f"Extract team1 and team2 from: '{state['query']}'. Reply as JSON: {{\"team1\": \"...\", \"team2\": \"...\", \"venue\": \"...\"}}"
    teams_raw = extract_llm.invoke(extract_prompt).content
    try:
        teams = json.loads(teams_raw.strip().strip("```json").strip("```"))
    except Exception:
        teams = {"team1": "Mumbai Indians", "team2": "Chennai Super Kings", "venue": "Wankhede Stadium"}
 
    pred = predict_win_probability(
        team1=teams.get("team1", "Mumbai Indians"),
        team2=teams.get("team2", "Chennai Super Kings"),
        venue=teams.get("venue", "Wankhede Stadium"),
        season=2025,
        toss_winner=teams.get("team1", ""),
        toss_decision="bat",
        db_url=db_url,
    )
    # Convert numpy scalar types (float32/int64 from XGBoost/sklearn) to
    # native Python types so json.dumps and the LangGraph msgpack
    # checkpointer can serialize this dict.
    pred = json.loads(json.dumps(pred, default=_to_native))

    llm = get_llm()
    chain = PREDICTION_PROMPT | llm
    narrative = chain.invoke({"prediction_json": json.dumps(pred, indent=2), "query": state["query"]})
    answer = (
        f"**Win Probability: {pred['team1']} {pred['team1_win_prob']*100:.1f}% | "
        f"{pred['team2']} {pred['team2_win_prob']*100:.1f}%**\n\n"
        + narrative.content
    )
    return {**state, "ml_result": pred, "answer": answer, "backend_used": "ML", "confidence": 0.78}
 
 
# ── Agent: Matchup ────────────────────────────────────────────────────────────
 
MATCHUP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an IPL bowling tactics analyst. Given SQL-computed head-to-head statistics
and RAG-retrieved recent form context, generate a detailed matchup verdict (200-250 words):
1. Historical dominance (who has the edge and why)
2. Phase-by-phase breakdown (powerplay, middle, death)
3. Key dismissal patterns or scoring zones
4. Recent form adjustment
5. A one-line verdict for the upcoming match
Context: {context}
H2H Stats: {h2h_stats}"""),
    ("human", "{query}")
])
 
 
def matchup_agent_node(state: AgentState, db_url: str, retriever) -> AgentState:
    from sqlalchemy import create_engine, text
    engine = create_engine(db_url)
 
    # Extract names from query
    llm = get_llm()
    extract = llm.invoke(
        f"Extract batsman and bowler from: '{state['query']}'. Reply as JSON: {{\"batter\": \"...\", \"bowler\": \"...\"}}"
    ).content
    try:
        names = json.loads(extract.strip().strip("```json").strip("```"))
    except Exception:
        names = {"batter": "V Kohli", "bowler": "JJ Bumrah"}
 
    batter = names.get("batter", "V Kohli")
    bowler = names.get("bowler", "JJ Bumrah")
 
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT * FROM h2h_records WHERE batter = :b AND bowler = :bw ORDER BY balls DESC"),
            {"b": batter, "bw": bowler},
        ).fetchall()
 
    h2h_stats = [dict(r._mapping) for r in rows]
    confidence_flag = sum(r.get("balls", 0) for r in h2h_stats) < 30
 
    # RAG enrichment
    rag_docs = retriever.retrieve(f"{batter} vs {bowler} head to head matchup", rerank=False)
    context = "\n".join(d["text"][:400] for d in rag_docs[:3])
 
    chain = MATCHUP_PROMPT | llm
    answer = chain.invoke({
        "context": context,
        "h2h_stats": json.dumps(h2h_stats, indent=2),
        "query": state["query"],
    })
    confidence = 0.6 if confidence_flag else 0.88
    answer_text = answer.content
    if confidence_flag:
        answer_text += "\n\n⚠️ *Confidence flag: fewer than 30 balls of H2H data available.*"
 
    return {**state, "answer": answer_text, "backend_used": "MATCHUP", "confidence": confidence}
 
 
# ── Agent: TeamVsTeam ─────────────────────────────────────────────────────────
 
TEAM_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an IPL team historian. Using the head-to-head SQL data provided,
write a 200-word all-time rivalry overview:
1. Overall H2H record
2. Which team performs better in playoffs vs league
3. Current form differential
4. Prediction for their next encounter
H2H Data: {h2h_data}"""),
    ("human", "{query}")
])
 
 
def teamvsteam_agent_node(state: AgentState, db_url: str) -> AgentState:
    from sqlalchemy import create_engine, text
    engine = create_engine(db_url)
 
    llm = get_llm()
    extract = llm.invoke(
        f"Extract team1 and team2 from: '{state['query']}'. Reply as JSON: {{\"team1\": \"...\", \"team2\": \"...\"}}"
    ).content
    try:
        teams = json.loads(extract.strip().strip("```json").strip("```"))
    except Exception:
        teams = {"team1": "Mumbai Indians", "team2": "Chennai Super Kings"}
 
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
            SELECT season,
                   SUM(CASE WHEN winner = :t1 THEN 1 ELSE 0 END) AS t1_wins,
                   SUM(CASE WHEN winner = :t2 THEN 1 ELSE 0 END) AS t2_wins,
                   COUNT(*) as matches
            FROM matches
            WHERE (team1 = :t1 AND team2 = :t2) OR (team1 = :t2 AND team2 = :t1)
            GROUP BY season ORDER BY season
            """),
            {"t1": teams["team1"], "t2": teams["team2"]},
        ).fetchall()
 
    h2h_data = [dict(r._mapping) for r in rows]
    chain = TEAM_PROMPT | llm
    answer = chain.invoke({"h2h_data": json.dumps(h2h_data, indent=2), "query": state["query"]})
    return {**state, "answer": answer.content, "backend_used": "TEAMVSTEAM", "confidence": 0.90}
 
 
# ── Graph builder ─────────────────────────────────────────────────────────────
 
def build_graph(db_url: str, retriever) -> Any:
    from functools import partial
 
    builder = StateGraph(AgentState)
 
    builder.add_node("router", route_query)
    builder.add_node("stats", partial(stats_agent_node, db_url=db_url))
    builder.add_node("narrative", partial(narrative_agent_node, retriever=retriever))
    builder.add_node("prediction", partial(prediction_agent_node, db_url=db_url))
    builder.add_node("matchup", partial(matchup_agent_node, db_url=db_url, retriever=retriever))
    builder.add_node("teamvsteam", partial(teamvsteam_agent_node, db_url=db_url))
 
    builder.set_entry_point("router")
    builder.add_conditional_edges(
        "router",
        decide_next,
        {
            "stats": "stats",
            "narrative": "narrative",
            "prediction": "prediction",
            "matchup": "matchup",
            "teamvsteam": "teamvsteam",
        },
    )
    for node in ("stats", "narrative", "prediction", "matchup", "teamvsteam"):
        builder.add_edge(node, END)
 
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)