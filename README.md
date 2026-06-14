# 🏏 IPL AI Intelligence Platform

> **Production-grade multi-agent AI system** combining Foundry IQ agentic retrieval, Hybrid RAG, XGBoost ML, and LangGraph orchestration over 18 seasons of IPL data (2008–2026).
>
> Built for the **Microsoft Agents League Hackathon 2026** — Enterprise Agents track.

[![CI](https://github.com/Jagadeep-Reddy/ipl-ai-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/Jagadeep-Reddy/ipl-ai-platform/actions)
[![RAGAS Faithfulness](https://img.shields.io/badge/RAGAS%20Faithfulness-≥0.88-green)](eval/last_ragas_scores.json)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)](https://fastapi.tiangolo.com)
[![Foundry IQ](https://img.shields.io/badge/Microsoft-Foundry%20IQ-0078D4)](https://azure.microsoft.com/products/ai-foundry/iq/)

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         User Query (Natural Language)                        │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend  (SSE Streaming + WebSocket)               │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                      LangGraph Multi-Agent Router                      │  │
│  │                                                                        │  │
│  │   STATS │ NARRATIVE │ PREDICTION │ MATCHUP │ TEAMVSTEAM                │  │
│  └──────────┬───────────────┬──────────────────────────────────────────── ┘  │
│             │               │                                               │
│  ┌──────────▼────────┐  ┌───▼──────────────────────────────────────────┐  │
│  │  StatsQA Agent    │  │           NarrativeQA Agent                  │  │
│  │  Text-to-SQL      │  │                                              │  │
│  │  PostgreSQL       │  │  ┌─────────────────────────────────────────┐│  │
│  │  LangChain SQL    │  │  │  PRIMARY: Foundry IQ (Azure AI Search)  ││  │
│  └───────────────────┘  │  │                                         ││  │
│                          │  │  1. Query decomposition (LLM)           ││  │
│  ┌───────────────────┐  │  │  2. Parallel hybrid search              ││  │
│  │  Prediction Agent │  │  │     keyword + vector + semantic         ││  │
│  │  XGBoost Win Prob │  │  │  3. RRF merge + deduplication           ││  │
│  │  SHAP Explainer   │  │  │  4. Grounded answer + [SOURCE N] cite   ││  │
│  └───────────────────┘  │  └─────────────────────────────────────────┘│  │
│                          │                    │ on failure              │  │
│  ┌───────────────────┐  │  ┌─────────────────▼───────────────────────┐│  │
│  │  Matchup Agent    │  │  │  FALLBACK: Local Qdrant RAG             ││  │
│  │  H2H SQL + RAG    │  │  │  BGE-M3 + BM25 + RRF + cross-encoder   ││  │
│  └───────────────────┘  │  └─────────────────────────────────────────┘│  │
│                          └──────────────────────────────────────────────┘  │
│  ┌───────────────────┐                                                      │
│  │  TeamVsTeam Agent │                                                      │
│  │  All-time H2H SQL │                                                      │
│  └───────────────────┘                                                      │
└──────────────────────────────────────────────────────────────────────────────┘
        │                                          │
        ▼                                          ▼
┌────────────────────┐                  ┌──────────────────────┐
│  Streamlit Frontend│                  │  WebSocket Live Feed  │
│  Historical Q&A    │                  │  Per-Ball Intent      │
│  Live Match Intent │                  │  < 300ms latency      │
└────────────────────┘                  └──────────────────────┘
```

---

## 🔷 Foundry IQ Integration

This platform integrates **Microsoft Foundry IQ** — the Microsoft intelligence layer for enterprise AI agents — as the primary knowledge retrieval engine for all narrative queries.

### What is Foundry IQ?

Foundry IQ is Microsoft's managed knowledge layer built on Azure AI Search. Unlike traditional RAG (one query → one vector lookup), Foundry IQ treats retrieval as a **reasoning task**:

```
Traditional RAG                     Foundry IQ Agentic Retrieval
──────────────                      ─────────────────────────────
User Query                          User Query
    │                                   │
    ▼                                   ▼
Single vector search            LLM Query Decomposition
    │                           (breaks into 2-3 subqueries)
    ▼                                   │
Top-K docs                             ▼
    │                           Parallel Hybrid Search
    ▼                           (keyword + vector + semantic)
LLM Answer                            │  │  │
                                       ▼  ▼  ▼
                                 RRF Merge + Deduplication
                                       │
                                       ▼
                               Grounded Answer + Citations
                               [SOURCE 1], [SOURCE 2] ...
```

### How Foundry IQ is used in this platform

The `agents/foundry_iq.py` module implements the full agentic retrieval pipeline:

**1. Index Setup** — All 179 IPL knowledge documents (seasons, players, venues) are indexed into Azure AI Search with:
- `text-embedding-3-small` vector embeddings (1536 dims)
- Full-text search with `en.microsoft` analyzer
- Semantic configuration for reranking
- Metadata filters (doc_type, season, team, player, venue)

**2. Query Decomposition** — When a narrative query arrives (e.g. "Describe MS Dhoni's captaincy across seasons"), the LLM decomposes it into focused subqueries:
```json
["MS Dhoni captaincy record CSK titles",
 "MS Dhoni leadership style IPL finals",
 "Chennai Super Kings title wins seasons"]
```

**3. Parallel Hybrid Search** — All subqueries execute simultaneously against Azure AI Search using hybrid mode (keyword + vector + semantic reranking).

**4. RRF Merge** — Results from all subqueries are merged using Reciprocal Rank Fusion (k=60), promoting documents that appear highly ranked across multiple subqueries.

**5. Grounded Answer** — GPT-4.1-mini generates a factual answer citing specific sources as `[SOURCE N]`, ensuring full traceability.

**6. Graceful Fallback** — If Foundry IQ is unavailable (missing credentials, network issue), the pipeline falls back automatically to the local Qdrant RAG (BGE-M3 + BM25 + cross-encoder) without any user-visible degradation.

```python
# In agents/router.py — narrative_agent_node
foundry_result = foundry_iq_narrative_node(state)

if foundry_result.get("backend_used") == "FOUNDRY_IQ":
    return foundry_result  # Foundry IQ answered

# Fallback: local Qdrant RAG
docs = retriever.retrieve(state["query"])
...
```

---

## ✨ Key Features

| Feature | Detail |
|---|---|
| **Foundry IQ Agentic Retrieval** | Microsoft intelligence layer — query decomposition → parallel hybrid search → RRF merge → cited answers |
| **5 Specialist AI Agents** | StatsQA (Text-to-SQL), NarrativeQA (Foundry IQ + RAG), Prediction (ML+LLM), Matchup (H2H), TeamVsTeam |
| **Hybrid RAG Fallback** | BGE-M3 dense + BM25 sparse → RRF fusion → cross-encoder reranking (local Qdrant) |
| **Win Probability** | XGBoost + Optuna (AUC > 0.72) with SHAP explanations |
| **Intent Classifier** | 20-feature XGBoost → 4-class per-ball intent (~96% accuracy) |
| **Real-time Pipeline** | WebSocket per-ball updates < 300ms (XGBoost + RAG + LLM parallel) |
| **LLM** | Azure OpenAI GPT-4.1-mini |
| **RAGAS Evaluation** | CI gate: faithfulness ≥ 0.88 required before deployment |
| **LLM-generated Docs** | GPT-4.1-mini writes rich narrative docs for each player/season/venue |

---

## 📁 Project Structure

```
ipl-ai-platform/
├── agents/
│   ├── router.py           # LangGraph: router + 5 specialist agent nodes
│   └── foundry_iq.py       # ★ Foundry IQ: Azure AI Search indexing + agentic retrieval
├── data/
│   ├── ingest.py           # ETL: load, clean, normalise, intent-label 465K deliveries
│   └── precompute.py       # PostgreSQL schema + 7 pre-aggregated tables
├── rag/
│   ├── doc_generator.py    # GPT-4.1-mini → 179 rich narrative IPL documents
│   └── retriever.py        # BGE-M3 + BM25 + RRF + cross-encoder (local fallback)
├── ml/
│   ├── train_win_prob.py   # XGBoost Model A: win probability + SHAP
│   └── train_intent.py     # XGBoost Model B: per-ball batsman intent (4 classes)
├── api/
│   ├── main.py             # FastAPI: /query SSE stream, /ws/live WebSocket, /metrics
│   └── schemas.py          # Pydantic v2 request/response schemas
├── frontend/
│   └── app.py              # Streamlit: Historical Q&A chat + Live Match Intent dashboard
├── eval/
│   ├── ragas_eval.py       # RAGAS evaluation: 4 metrics + CI gate (faithfulness ≥ 0.88)
│   └── golden_qa.json      # 139 ground-truth Q&A pairs
├── scripts/
│   ├── setup.py            # Master bootstrap: Phase 1a→1b→1c→1d→2→3→4
│   └── simulate_match.py   # 2024 IPL Final WebSocket replay
├── tests/                  # pytest unit tests
├── locust_test.py          # Load test: 200 concurrent users
├── docker-compose.yml      # PostgreSQL + Qdrant + Redis + API + UI
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## 🚀 Quickstart

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Kaggle account (dataset download)
- Azure subscription with:
  - Azure OpenAI resource (GPT-4.1-mini + text-embedding-3-small deployed)
  - Azure AI Search resource (for Foundry IQ)

### 1. Clone & Configure

```bash
git clone https://github.com/Jagadeep-Reddy/ipl-ai-platform.git
cd ipl-ai-platform
cp .env.example .env
```

Edit `.env` and fill in:

```env
# Azure OpenAI (required)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small

# Foundry IQ – Azure AI Search (required for agentic retrieval)
AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_API_KEY=your-admin-key
AZURE_SEARCH_INDEX_NAME=ipl-knowledge
FOUNDRY_IQ_ENABLED=true

# PostgreSQL
DATABASE_URL=postgresql://ipl:ipl_secret@127.0.0.1:5433/ipl_db

# Qdrant (local RAG fallback)
QDRANT_URL=http://localhost:6333

# Kaggle
KAGGLE_USERNAME=your-username
KAGGLE_KEY=your-key
```

### 2. Start Infrastructure

```bash
docker-compose up postgres qdrant redis -d
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Bootstrap the Platform

```bash
python scripts/setup.py
```

This runs all 4 phases automatically:

```
Phase 1a – Data Ingestion          (~1 min)   ETL 465K deliveries, intent labels
Phase 1b – PostgreSQL Database     (~5 min)   Schema + 7 pre-aggregated tables
Phase 1c – RAG Document Generation (~40 min)  GPT-4.1-mini writes 179 IPL docs
Phase 1d – Foundry IQ Indexing     (~10 min)  Index docs into Azure AI Search
Phase 2  – Win Probability Model   (~3 min)   XGBoost + Optuna + SHAP
Phase 3  – Intent Classifier       (~90 min)  XGBoost multi-class
Phase 4  – RAGAS Evaluation        (~40 min)  CI gate: faithfulness ≥ 0.88
```

**Skip flags:**
```bash
python scripts/setup.py --skip-models          # Skip Phase 2 & 3
python scripts/setup.py --skip-docs            # Skip Phase 1c
python scripts/setup.py --skip-foundry-iq      # Skip Phase 1d
python scripts/setup.py --skip-eval            # Skip Phase 4
```

### 5. Start the Application

```bash
# Full stack
docker-compose up

# Or individually
uvicorn api.main:app --reload        # http://localhost:8000
streamlit run frontend/app.py        # http://localhost:8501
```

---

## 🌐 API Reference

### `POST /query` — SSE Streaming Chat

```bash
curl -N -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"message": "Describe MS Dhoni captaincy across seasons", "session_id": "abc"}'
```

Response: Server-Sent Events stream
- `event: token` — streaming answer tokens
- `event: metadata` — `backend_used` (FOUNDRY_IQ / RAG / SQL / ML), confidence, sources

### `WS /ws/live` — Real-Time Intent

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/live");
ws.send(JSON.stringify({
  match_id: "m001", batter: "V Kohli", bowler: "JJ Bumrah",
  over: 18, ball_num: 2, runs_scored: 0, ...
}));
// Returns IntentVerdict JSON within 300ms
```

### `GET /metrics` — RAGAS Scores + System Health

```bash
curl http://localhost:8000/metrics
# Returns: faithfulness, answer_relevancy, context_precision, context_recall
```

### `GET /health` — Service Status

```bash
curl http://localhost:8000/health
```

---

## 🤖 The 5 AI Agents

| Agent | Query Type | Primary Backend | Latency |
|---|---|---|---|
| **StatsQA** | Numbers, rankings, aggregations | PostgreSQL Text-to-SQL | ~3s |
| **NarrativeQA** | History, biography, season stories | Foundry IQ → Qdrant RAG | ~4s |
| **Prediction** | Win probability, match outcomes | XGBoost + SHAP + LLM | ~3s |
| **Matchup** | Batter vs bowler H2H analysis | SQL + RAG enrichment | ~4s |
| **TeamVsTeam** | All-time team rivalry records | PostgreSQL aggregation | ~2s |

The router uses GPT-4.1-mini to classify incoming queries into one of 5 types, then dispatches to the appropriate specialist agent. All agents share the same `AgentState` TypedDict flowing through the LangGraph DAG.

---

## 🧠 ML Models

### Model A — Win Probability (XGBoost)

- **Training data:** 1,243 IPL matches (2008–2024)
- **Features:** H2H win%, venue win%, avg 1st-innings score, last-5 form rating, toss factor, squad strength index
- **Target:** Binary match winner
- **Metric:** AUC > 0.72 on 2024 holdout
- **Tuning:** Optuna Bayesian hyperparameter search (190 seconds, ~50 trials)
- **Extras:** SHAP TreeExplainer generates per-prediction feature importance narrative

### Model B — Batsman Intent Classifier (XGBoost)

- **Training data:** 465,150 deliveries with rule-based intent labels
- **Features:** 20 contextual features including phase, required run rate, current run rate, pressure index, H2H strike rate, bowler economy
- **Classes:** `aggressive` | `defensive` | `neutral` | `pressure_error`
- **Metric:** 95.6% overall accuracy (class imbalance: neutral dominates)
- **Note:** `pressure_error` class has lower recall (0.30) due to rarity — future improvement with SMOTE/focal loss

---

## 📊 Database Schema

```sql
matches              -- 1,243 rows: season, teams, venue, winner, toss, match_type
deliveries           -- 465,150 rows: ball-by-ball with intent label + phase tag
player_career_stats  -- pre-agg: runs/SR/avg/wickets/economy per player·season·phase·role
h2h_records          -- batter vs bowler: SR/dots/boundaries per phase (all history)
venue_stats          -- avg score, bat-first win%, powerplay/middle/death RR per venue·season
team_form            -- last-5-match win/loss rating per team·season
squad_strength       -- composite squad strength index per team·season
```

---

## 🔬 Dual Retrieval Architecture

### Primary: Foundry IQ (Azure AI Search)

```
User Query
    │
    ▼
LLM Query Decomposition
(GPT-4.1-mini breaks into 2-3 subqueries)
    │
    ├──────────────┬──────────────┐
    ▼              ▼              ▼
Subquery 1     Subquery 2    Subquery 3
    │              │              │
    └──────────────┴──────────────┘
              (parallel)
                  │
                  ▼
      Azure AI Search Hybrid Search
      keyword (en.microsoft) + vector (1536-dim)
      + semantic reranking (ipl-semantic config)
                  │
                  ▼
      RRF Merge (k=60) across all subqueries
      Deduplicate → top-8 docs
                  │
                  ▼
      GPT-4.1-mini Grounded Answer
      with [SOURCE N] citations
```

### Fallback: Local Qdrant RAG

```
User Query
    │
    ├──► BGE-M3 Embedding → Qdrant HNSW (INT8) ── top-80 dense hits
    │
    ├──► BM25 (rank-bm25) ─────────────────────── top-80 sparse hits
    │
    └──► RRF Fusion (k=60) ─────────────────────── top-80 fused
              │
              └──► Cross-Encoder (ms-marco-MiniLM-L6) ── top-12 final docs
                          │
                          └──► GPT-4.1-mini → Answer
```

**Document corpus (179 docs, LLM-generated):**
- 19 season narrative summaries (2008–2026) with champion, runner-up, top performers
- 100 player profiles with batting/bowling stats by phase and head-to-head records
- 60 venue analyses with scoring patterns, pitch characteristics, bat-first/chase stats

---

## 📏 RAGAS Evaluation

```bash
python eval/ragas_eval.py
```

| Metric | Target | What it measures |
|---|---|---|
| **Faithfulness** | ≥ 0.88 | Are answer claims grounded in retrieved context? |
| **Answer Relevancy** | ≥ 0.70 | Does the answer directly address the question? |
| **Context Precision** | ≥ 0.55 | Fraction of retrieved docs that are relevant |
| **Context Recall** | ≥ 0.55 | Fraction of relevant docs that were retrieved |

The CI gate in `eval/ragas_eval.py` blocks deployment if `faithfulness < 0.88`. RAGAS uses Azure OpenAI GPT-4.1-mini as the judge LLM and `text-embedding-3-small` for answer relevancy scoring.

---

## ⚡ Real-Time Pipeline (WebSocket)

```
Cricket Match WebSocket Feed
        │
        ▼
    BallEvent JSON (batter, bowler, over, ball, runs, phase, ...)
        │
   ┌────┴─────────────────────────────┐
   │           (asyncio parallel)      │
   ▼                                   ▼
XGBoost Intent Classifier         Qdrant RAG Matchup Context
20-feature inference (~20ms)      H2H context from Redis/Qdrant (~50ms)
   │                                   │
   └─────────────────┬─────────────────┘
                     ▼
           GPT-4.1-mini Commentary
           (150ms budget)
                     │
                     ▼
           IntentVerdict WebSocket Push
           { intent, confidence, commentary, shap_features }
           Total latency < 300ms
```

---

## 🧪 Testing

```bash
# Unit tests
pytest tests/ -v

# Load test (200 concurrent users, 60 seconds)
locust -f locust_test.py --headless -u 200 -r 20 -t 60s \
  --host http://localhost:8000

# 2024 IPL Final ball-by-ball simulation
python scripts/simulate_match.py

# Run Foundry IQ setup only (after docs are generated)
python agents/foundry_iq.py
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Intelligence Layer** | Microsoft Foundry IQ (Azure AI Search agentic retrieval) |
| **LLM** | Azure OpenAI GPT-4.1-mini |
| **Embeddings** | Azure OpenAI text-embedding-3-small (Foundry IQ) + BGE-M3 (local RAG) |
| **Orchestration** | LangGraph 0.1 + LangChain 0.2 |
| **Vector DB** | Azure AI Search (Foundry IQ) + Qdrant HNSW INT8 (local fallback) |
| **Sparse Retrieval** | rank-bm25 |
| **Reranker** | ms-marco-MiniLM-L-6-v2 (CrossEncoder) |
| **ML Models** | XGBoost 2.0 + Optuna + SHAP + sklearn calibration |
| **Database** | PostgreSQL 16 |
| **Cache** | Redis 7 |
| **API** | FastAPI 0.111, SSE, WebSockets |
| **Frontend** | Streamlit 1.36 + Plotly |
| **Evaluation** | RAGAS 0.1 (Azure OpenAI judge) |
| **CI/CD** | GitHub Actions → Railway |
| **Containers** | Docker + Docker Compose |

---

## 👤 Author

**Buthuru Jagadeep Reddy**
AI/ML Engineer | Bengaluru, India
- Built on Azure OpenAI + Microsoft Foundry IQ
- Submitted to Microsoft Agents League Hackathon 2026 — Enterprise Agents track

---
