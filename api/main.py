"""
api/main.py
───────────
Phase 2 Day 8 – Production FastAPI backend.
  POST /query        → LangGraph multi-agent pipeline (SSE streaming)
  WS   /ws/live      → Real-time batsman intent pipeline
  GET  /health       → Service status
  GET  /metrics      → Latest RAGAS scores
"""
 
from __future__ import annotations
 
import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

import joblib
import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
 
from api.schemas import (
    BallEvent,
    IntentVerdict,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
)
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)
 
# ── Startup state ─────────────────────────────────────────────────────────────
 
class AppState:
    graph = None
    retriever = None
    redis_client = None
    intent_artifact = None
    ragas_scores: dict = {}
    model_load_times: dict = {}
 
 
app_state = AppState()
 
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and connections at startup."""
    t0 = time.perf_counter()
 
    # Redis
    app_state.redis_client = aioredis.from_url(
        os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        decode_responses=True,
    )
 
    # RAG retriever
    from rag.retriever import IPLRetriever
    try:
        app_state.retriever = IPLRetriever(
            qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            bm25_path=Path("rag/bm25_index.pkl"),
        )
    except Exception as e:
        logger.warning("Retriever load failed (index may not exist yet): %s", e)
 
    # LangGraph
    if app_state.retriever:
        from agents.router import build_graph
        app_state.graph = build_graph(
            db_url=os.environ["DATABASE_URL"],
            retriever=app_state.retriever,
        )
 
    # Intent classifier
    intent_path = Path(os.environ.get("INTENT_MODEL_PATH", "ml/models/intent_classifier.joblib"))
    if intent_path.exists():
        app_state.intent_artifact = joblib.load(intent_path)
        app_state.model_load_times["intent"] = round(time.perf_counter() - t0, 3)
 
    # Load last RAGAS scores
    ragas_file = Path("eval/last_ragas_scores.json")
    if ragas_file.exists():
        app_state.ragas_scores = json.loads(ragas_file.read_text())
 
    logger.info("Startup complete in %.2fs", time.perf_counter() - t0)
    yield
 
    # Teardown
    if app_state.redis_client:
        await app_state.redis_client.aclose()
 
 
# ── App ───────────────────────────────────────────────────────────────────────
 
app = FastAPI(
    title="IPL AI Intelligence Platform",
    version="1.0.0",
    description="Hybrid RAG + ML + Real-Time IPL Analytics",
    lifespan=lifespan,
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
 
# ── /health ───────────────────────────────────────────────────────────────────
 
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "graph_ready": app_state.graph is not None,
        "retriever_ready": app_state.retriever is not None,
        "intent_model_ready": app_state.intent_artifact is not None,
        "model_load_times": app_state.model_load_times,
    }
 
 
# ── /metrics ──────────────────────────────────────────────────────────────────
 
@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    ragas_file = Path("eval/last_ragas_scores.json")
    if ragas_file.exists():
        try:
            scores = json.loads(ragas_file.read_text())
            return MetricsResponse(**scores)
        except Exception:
            logger.warning("Failed to read %s; falling back to startup snapshot", ragas_file)
    return MetricsResponse(**app_state.ragas_scores) if app_state.ragas_scores else MetricsResponse()
 
 
# ── /query (SSE streaming) ────────────────────────────────────────────────────
 
CACHE_TTL = int(os.environ.get("CACHE_TTL_SECONDS", 3600))
 
 
async def _stream_answer(query: str, session_id: str):
    """Core streaming generator – checks Redis cache first, then runs the graph."""
    cache_key = f"query:{hash(query)}"
 
    # Cache hit
    cached = await app_state.redis_client.get(cache_key)
    if cached:
        data = json.loads(cached)
        yield {"event": "token", "data": data["answer"]}
        yield {"event": "metadata", "data": json.dumps({"backend": data["backend"], "confidence": data["confidence"], "cached": True})}
        return
 
    if app_state.graph is None:
        yield {"event": "error", "data": "System not ready – models loading"}
        return
 
    state = {
        "query": query,
        "query_type": None,
        "context": [],
        "sql_result": None,
        "ml_result": None,
        "answer": None,
        "confidence": 0.0,
        "backend_used": None,
        "session_id": session_id,
        "history": [],
    }
 
    config = {"configurable": {"thread_id": session_id}}
    try:
        result = await asyncio.to_thread(
            app_state.graph.invoke, state, config
        )
    except Exception:
        import traceback
        traceback.print_exc()
        logger.exception("Graph invocation failed for query: %s", query)
        yield {"event": "error", "data": "Internal error while processing query (see server logs)."}
        return
 
    answer = result.get("answer", "No answer generated.")
    backend = result.get("backend_used", "unknown")
    confidence = result.get("confidence", 0.0)
 
    # Stream tokens word-by-word for natural feel
    for word in answer.split(" "):
        yield {"event": "token", "data": word + " "}
        await asyncio.sleep(0.01)
 
    yield {
        "event": "metadata",
        "data": json.dumps({
            "backend": backend,
            "confidence": confidence,
            "sources": [d.get("doc_id", "") for d in result.get("context", [])[:3]],
            "cached": False,
        }),
    }
 
    # Cache response
    await app_state.redis_client.setex(
        cache_key,
        CACHE_TTL,
        json.dumps({"answer": answer, "backend": backend, "confidence": confidence}),
    )
 
 
@app.post("/query")
async def query_endpoint(req: QueryRequest):
    return EventSourceResponse(_stream_answer(req.message, req.session_id))
 
 
# ── /ws/live (WebSocket – real-time intent) ───────────────────────────────────
 
@app.websocket("/ws/live")
async def live_intent_websocket(websocket: WebSocket):
    """
    Receives BallEvent JSON, runs the 5-step real-time pipeline,
    and pushes IntentVerdict back within 300ms.
    """
    await websocket.accept()
    logger.info("WebSocket connection established.")
 
    try:
        while True:
            raw = await websocket.receive_text()
            ball_data = json.loads(raw)
            ball = BallEvent(**ball_data)
 
            t_start = time.perf_counter()
 
            # Steps 1-2: feature extraction + XGBoost inference (parallel with RAG)
            intent_task = asyncio.create_task(_run_intent(ball))
            rag_task = asyncio.create_task(_run_live_rag(ball))
 
            intent_result, rag_context = await asyncio.gather(intent_task, rag_task)
 
            # Step 3: LLM verdict generation
            verdict_text = await asyncio.to_thread(
                _generate_verdict, ball, intent_result, rag_context
            )
 
            elapsed_ms = int((time.perf_counter() - t_start) * 1000)
 
            verdict = IntentVerdict(
                batter=ball.batter,
                bowler=ball.bowler,
                over=ball.over,
                ball_num=ball.ball_num,
                intent=intent_result["intent"],
                probabilities=intent_result["probabilities"],
                confidence_flag=intent_result["confidence_flag"],
                verdict_text=verdict_text,
                rag_context_summary=rag_context[:200] if rag_context else "",
                latency_ms=elapsed_ms,
            )
            await websocket.send_text(verdict.model_dump_json())
 
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        await websocket.close(code=1011)
 
 
# ── /predict_intent (REST – single-ball, used by frontend + load tests) ───────
 
@app.post("/predict_intent", response_model=IntentVerdict)
async def predict_intent_endpoint(ball: BallEvent):
    """
    REST alternative to the WebSocket live pipeline.
    Runs XGBoost intent inference + RAG context fetch + LLM commentary
    and returns a complete IntentVerdict within 300ms.
    """
    t_start = time.perf_counter()
 
    intent_task = asyncio.create_task(_run_intent(ball))
    rag_task = asyncio.create_task(_run_live_rag(ball))
    intent_result, rag_context = await asyncio.gather(intent_task, rag_task)
 
    verdict_text = await asyncio.to_thread(
        _generate_verdict, ball, intent_result, rag_context
    )
    elapsed_ms = int((time.perf_counter() - t_start) * 1000)
 
    return IntentVerdict(
        batter=ball.batter,
        bowler=ball.bowler,
        over=ball.over,
        ball_num=ball.ball_num,
        intent=intent_result["intent"],
        probabilities=intent_result["probabilities"],
        confidence_flag=intent_result["confidence_flag"],
        verdict_text=verdict_text,
        rag_context_summary=rag_context[:200] if rag_context else "",
        latency_ms=elapsed_ms,
    )
 
 
async def _run_intent(ball: BallEvent) -> dict:
    if app_state.intent_artifact is None:
        return {"intent": "neutral", "probabilities": {}, "confidence_flag": True}
 
    from ml.train_intent import predict_intent
    ball_dict = ball.model_dump()
    return await asyncio.to_thread(predict_intent, ball_dict)
 
 
async def _run_live_rag(ball: BallEvent) -> str:
    if app_state.retriever is None:
        return ""
 
    # Check Redis for cached matchup context
    cache_key = f"live:{ball.batter}:{ball.bowler}"
    if app_state.redis_client:
        cached = await app_state.redis_client.get(cache_key)
        if cached:
            return cached
 
    from rag.retriever import LiveContextRetriever
    lc = LiveContextRetriever(app_state.retriever)
    docs = await asyncio.to_thread(lc.fetch_matchup_context, ball.batter, ball.bowler)
    context = " | ".join(d["text"][:200] for d in docs)
 
    if app_state.redis_client:
        await app_state.redis_client.setex(cache_key, 3600, context)
    return context
 
 
VERDICT_PROMPT = """
Ball context: Over {over}.{ball}, {batter} facing {bowler}.
Score: {score}/{wickets}, RRR: {rrr:.1f}, CRR: {crr:.1f}
Intent prediction: {intent} (confidence: {conf:.0%})
Historical context: {context}
 
Write a crisp 1-2 sentence live commentary verdict explaining the predicted intent.
Be specific about the match situation driving this prediction.
"""
 
 
def _generate_verdict(ball: BallEvent, intent_result: dict, rag_context: str) -> str:
    from langchain_openai import AzureChatOpenAI
 
    prompt = VERDICT_PROMPT.format(
        over=ball.over,
        ball=ball.ball_num,
        batter=ball.batter,
        bowler=ball.bowler,
        score=ball.cum_runs,
        wickets=ball.wickets_fallen,
        rrr=ball.required_rr,
        crr=ball.current_rr,
        intent=intent_result["intent"],
        conf=max(intent_result.get("probabilities", {}).values(), default=0.5),
        context=rag_context[:300],
    )
 
    try:
        llm = AzureChatOpenAI(
            azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini"),
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            temperature=0.2,
            max_tokens=80,
        )
        resp = llm.invoke(prompt)
        return resp.content.strip()
    except Exception as e:
        logger.error("Error generating verdict: %s", e)
        return "Verdict generation failed."