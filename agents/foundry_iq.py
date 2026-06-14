"""
agents/foundry_iq.py
────────────────────
Foundry IQ integration for the IPL AI Intelligence Platform.

Foundry IQ is Microsoft's managed knowledge layer built on Azure AI Search.
It provides agentic retrieval — treating knowledge retrieval as a multi-step
reasoning task rather than a single vector lookup:

  1. Query decomposition into subqueries (LLM-powered planning)
  2. Parallel hybrid search (keyword + vector) across the IPL knowledge base
  3. RRF merge + semantic reranking across all results
  4. Citation-backed grounded answers with source attribution

This module:
  - Creates the Azure AI Search index with vector + semantic config
  - Indexes all IPL documents (season, player, venue) into Azure AI Search
  - Exposes foundry_iq_narrative_node() for the LangGraph NarrativeQA agent
  - Falls back gracefully to local Qdrant RAG if Foundry IQ is unavailable

Required .env variables:
  AZURE_SEARCH_ENDPOINT     – e.g. https://ipl-search.search.windows.net
  AZURE_SEARCH_API_KEY      – Admin or query key from Azure portal
  AZURE_OPENAI_ENDPOINT     – existing Azure OpenAI endpoint (already set)
  AZURE_OPENAI_API_KEY      – existing Azure OpenAI key (already set)
  AZURE_OPENAI_DEPLOYMENT   – existing deployment (gpt-4.1-mini, already set)
  AZURE_OPENAI_EMBEDDING_DEPLOYMENT – embedding deployment (text-embedding-3-small)

Optional:
  AZURE_SEARCH_INDEX_NAME   – index name (default: ipl-knowledge)
  FOUNDRY_IQ_ENABLED        – set to "false" to bypass (default: "true")
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.router import AgentState

logger = logging.getLogger(__name__)

# ── Feature flag ──────────────────────────────────────────────────────────────
_FOUNDRY_IQ_ENABLED = os.environ.get("FOUNDRY_IQ_ENABLED", "true").lower() != "false"

# ── Config ────────────────────────────────────────────────────────────────────
SEARCH_ENDPOINT   = os.environ.get("AZURE_SEARCH_ENDPOINT", "")
SEARCH_API_KEY    = os.environ.get("AZURE_SEARCH_API_KEY", "")
INDEX_NAME        = os.environ.get("AZURE_SEARCH_INDEX_NAME", "ipl-knowledge")
OAI_ENDPOINT      = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
OAI_KEY           = os.environ.get("AZURE_OPENAI_API_KEY", "")
OAI_DEPLOYMENT    = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
OAI_VERSION       = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
EMB_DEPLOYMENT    = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
VECTOR_DIMS       = 1536  # text-embedding-3-small output dims

# ── Azure AI Search index schema ──────────────────────────────────────────────

INDEX_SCHEMA = {
    "name": INDEX_NAME,
    "fields": [
        {"name": "id",       "type": "Edm.String", "key": True, "filterable": True},
        {"name": "doc_id",   "type": "Edm.String", "filterable": True, "searchable": False},
        {"name": "doc_type", "type": "Edm.String", "filterable": True, "facetable": True},
        {"name": "season",   "type": "Edm.Int32",  "filterable": True, "sortable": True, "searchable": False},
        {"name": "team",     "type": "Edm.String", "filterable": True, "searchable": True},
        {"name": "player",   "type": "Edm.String", "filterable": True, "searchable": True},
        {"name": "venue",    "type": "Edm.String", "filterable": True, "searchable": True},
        {"name": "text",     "type": "Edm.String", "searchable": True, "analyzer": "en.microsoft"},
        {
            "name": "text_vector",
            "type": "Collection(Edm.Single)",
            "searchable": True,
            "dimensions": VECTOR_DIMS,
            "vectorSearchProfile": "ipl-vector-profile",
        },
    ],
    "vectorSearch": {
        "algorithms": [
            {
                "name": "ipl-hnsw",
                "kind": "hnsw",
                "hnswParameters": {"m": 4, "efConstruction": 400, "metric": "cosine"},
            }
        ],
        "profiles": [{"name": "ipl-vector-profile", "algorithm": "ipl-hnsw"}],
    },
    "semantic": {
        "configurations": [
            {
                "name": "ipl-semantic",
                "prioritizedFields": {
                    "titleField": {"fieldName": "doc_id"},
                    "contentFields": [{"fieldName": "text"}],
                    "keywordsFields": [
                        {"fieldName": "team"},
                        {"fieldName": "player"},
                        {"fieldName": "venue"},
                    ],
                },
            }
        ]
    },
}


# ── Client helpers ────────────────────────────────────────────────────────────

def _search_client():
    from azure.search.documents import SearchClient
    from azure.core.credentials import AzureKeyCredential
    return SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_API_KEY),
    )


def _index_client():
    from azure.search.documents.indexes import SearchIndexClient
    from azure.core.credentials import AzureKeyCredential
    return SearchIndexClient(
        endpoint=SEARCH_ENDPOINT,
        credential=AzureKeyCredential(SEARCH_API_KEY),
    )


def _oai_client():
    from openai import AzureOpenAI
    return AzureOpenAI(
        azure_endpoint=OAI_ENDPOINT,
        api_key=OAI_KEY,
        api_version=OAI_VERSION,
    )


def _embed(text: str) -> list[float]:
    """Generate a text-embedding-3-small vector via Azure OpenAI."""
    client = _oai_client()
    resp = client.embeddings.create(
        input=text[:8000],
        model=EMB_DEPLOYMENT,
    )
    return resp.data[0].embedding


# ── Index management ──────────────────────────────────────────────────────────

def create_index_if_not_exists() -> bool:
    """Create the Azure AI Search index if it doesn't already exist."""
    from azure.search.documents.indexes.models import SearchIndex
    client = _index_client()
    existing = {idx.name for idx in client.list_indexes()}
    if INDEX_NAME in existing:
        logger.info("Foundry IQ: index '%s' already exists.", INDEX_NAME)
        return False
    index = SearchIndex.from_dict(INDEX_SCHEMA)
    client.create_index(index)
    logger.info("Foundry IQ: created index '%s'.", INDEX_NAME)
    return True


def upload_documents(
    docs_path: Path = Path("rag/documents.json"),
    batch_size: int = 50,
) -> int:
    """
    Upload all IPL documents to the Azure AI Search index.
    Generates text-embedding-3-small vectors for each document.
    Returns total documents uploaded.
    """
    if not SEARCH_ENDPOINT or not SEARCH_API_KEY:
        logger.warning("Foundry IQ: missing credentials, skipping upload.")
        return 0

    docs = json.loads(docs_path.read_text(encoding="utf-8", errors="replace"))
    client = _search_client()
    batch, total = [], 0

    for doc in docs:
        try:
            vector = _embed(doc["text"])
        except Exception as e:
            logger.warning("Embedding failed for %s: %s. Using zero vector.", doc["doc_id"], e)
            vector = [0.0] * VECTOR_DIMS

        batch.append({
            "id":          doc["doc_id"].replace(" ", "_").replace("/", "_"),
            "doc_id":      doc["doc_id"],
            "doc_type":    doc.get("doc_type", "unknown"),
            "season":      int(doc["season"]) if doc.get("season") else None,
            "team":        doc.get("team") or "",
            "player":      doc.get("player") or "",
            "venue":       doc.get("venue") or "",
            "text":        doc["text"],
            "text_vector": vector,
        })

        if len(batch) >= batch_size:
            client.upload_documents(batch)
            total += len(batch)
            logger.info("Foundry IQ: uploaded %d/%d documents.", total, len(docs))
            batch = []
            time.sleep(0.5)

    if batch:
        client.upload_documents(batch)
        total += len(batch)
        logger.info("Foundry IQ: upload complete. Total: %d documents.", total)

    return total


# ── Agentic retrieval (Foundry IQ core pattern) ───────────────────────────────

def foundry_iq_search(query: str, top_k: int = 8) -> list[dict]:
    """
    Execute Foundry IQ-style agentic retrieval:

    Step 1 – Query decomposition: LLM breaks the question into 2-3 subqueries
             targeting different aspects (player stats / team history / venue)
    Step 2 – Parallel hybrid search: each subquery runs keyword + vector + semantic
             search against Azure AI Search simultaneously
    Step 3 – RRF merge: results from all subqueries merged via Reciprocal Rank Fusion
             exactly as Foundry IQ's agentic retrieval engine does internally
    Step 4 – Return top_k deduplicated docs with source attribution
    """
    if not SEARCH_ENDPOINT or not SEARCH_API_KEY:
        return []

    # ── Step 1: Query decomposition ────────────────────────────────────────────
    client = _oai_client()
    try:
        decomp = client.chat.completions.create(
            model=OAI_DEPLOYMENT,
            messages=[{
                "role": "user",
                "content": (
                    f"You are a query planner for an IPL cricket knowledge base with documents "
                    f"about players, seasons, and venues. Break this question into 2-3 focused "
                    f"subqueries that together would fully answer it. "
                    f"Reply ONLY as a JSON array of strings, no explanation.\n"
                    f"Question: {query}"
                ),
            }],
            max_tokens=150,
            temperature=0.0,
        )
        raw = decomp.choices[0].message.content.strip().strip("```json").strip("```").strip()
        subqueries = json.loads(raw)
        if not isinstance(subqueries, list) or not subqueries:
            subqueries = [query]
    except Exception as e:
        logger.warning("Foundry IQ: query decomposition failed (%s), using raw query.", e)
        subqueries = [query]

    logger.info("Foundry IQ: subqueries = %s", subqueries)

    # ── Step 2: Parallel hybrid search ────────────────────────────────────────
    sc = _search_client()

    def _hybrid_search(sq: str) -> list[dict]:
        try:
            from azure.search.documents.models import VectorizedQuery
            vq = VectorizedQuery(
                vector=_embed(sq),
                k_nearest_neighbors=top_k,
                fields="text_vector",
            )
            results = sc.search(
                search_text=sq,
                vector_queries=[vq],
                query_type="semantic",
                semantic_configuration_name="ipl-semantic",
                top=top_k,
                select=["id", "doc_id", "doc_type", "text", "team", "player", "venue", "season"],
            )
            return [dict(r) for r in results]
        except Exception as e:
            logger.warning("Foundry IQ hybrid search failed for '%s': %s. Trying keyword only.", sq, e)
            try:
                results = sc.search(
                    search_text=sq, top=top_k,
                    select=["id", "doc_id", "doc_type", "text", "team", "player", "venue", "season"],
                )
                return [dict(r) for r in results]
            except Exception:
                return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        all_results = list(ex.map(_hybrid_search, subqueries))

    # ── Step 3: RRF merge ──────────────────────────────────────────────────────
    K = 60
    rrf: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for rank_list in all_results:
        for rank, doc in enumerate(rank_list):
            did = doc.get("id", "")
            if not did:
                continue
            rrf[did] = rrf.get(did, 0.0) + 1.0 / (K + rank + 1)
            docs[did] = doc

    ranked = sorted(docs.values(), key=lambda d: rrf.get(d.get("id", ""), 0), reverse=True)
    top = ranked[:top_k]
    logger.info("Foundry IQ: %d unique docs merged, returning top %d.", len(docs), len(top))
    return top


# ── LangGraph node ────────────────────────────────────────────────────────────

def foundry_iq_narrative_node(state: "AgentState") -> "AgentState":
    """
    LangGraph node: answers NARRATIVE queries using Foundry IQ agentic retrieval.

    Pipeline:
      User Query
        → Foundry IQ Query Decomposition (LLM)
        → Parallel Hybrid Search (Azure AI Search: keyword + vector + semantic)
        → RRF Merge + Deduplication
        → Grounded LLM Answer with [SOURCE N] Citations

    Returns:
      - state with backend_used="FOUNDRY_IQ" on success
      - state with answer=None to trigger fallback to local Qdrant RAG
    """
    if not _FOUNDRY_IQ_ENABLED:
        logger.info("Foundry IQ: disabled via env flag.")
        return {**state, "answer": None}

    if not SEARCH_ENDPOINT or not SEARCH_API_KEY:
        logger.warning("Foundry IQ: AZURE_SEARCH_ENDPOINT or AZURE_SEARCH_API_KEY not configured.")
        return {**state, "answer": None}

    try:
        retrieved = foundry_iq_search(state["query"], top_k=8)

        if not retrieved:
            logger.warning("Foundry IQ: no results for '%s'.", state["query"][:60])
            return {**state, "answer": None}

        # Build context with Foundry IQ-style source attribution
        context_parts = []
        for i, doc in enumerate(retrieved):
            source = doc.get("doc_type", "doc").upper()
            doc_id = doc.get("doc_id", f"doc_{i}")
            text = doc.get("text", "")[:1200]
            context_parts.append(f"[SOURCE {i+1}: {source} – {doc_id}]\n{text}")
        context_str = "\n\n---\n\n".join(context_parts)

        # Grounded answer with citations
        client = _oai_client()
        response = client.chat.completions.create(
            model=OAI_DEPLOYMENT,
            messages=[{
                "role": "user",
                "content": (
                    "You are an expert IPL cricket analyst. Your knowledge is retrieved via "
                    "Foundry IQ agentic search from a comprehensive IPL knowledge base. "
                    "Answer the question using ONLY the provided sources. "
                    "Be specific, factual, and cite sources as [SOURCE N]. "
                    "If sources lack enough information, say so explicitly.\n\n"
                    f"Sources:\n{context_str}\n\n"
                    f"Question: {state['query']}\n\nAnswer:"
                ),
            }],
            max_tokens=600,
            temperature=0.2,
        )
        answer = response.choices[0].message.content.strip()

        context_docs = [
            {
                "doc_id":   d.get("doc_id", ""),
                "doc_type": d.get("doc_type", ""),
                "text":     d.get("text", ""),
                "score":    d.get("@search.score", 0.0),
                "source":   "foundry_iq",
            }
            for d in retrieved
        ]

        logger.info("Foundry IQ: answered '%s' with %d sources.", state["query"][:60], len(retrieved))
        return {
            **state,
            "context":      context_docs,
            "answer":       answer,
            "backend_used": "FOUNDRY_IQ",
            "confidence":   0.90,
        }

    except Exception as e:
        logger.error("Foundry IQ: failed (%s). Will fall back to local RAG.", e)
        return {**state, "answer": None}


# ── Setup entrypoint (called from scripts/setup.py Phase 1d) ──────────────────

def setup_foundry_iq(docs_path: Path = Path("rag/documents.json")) -> None:
    """
    One-time setup: create Azure AI Search index + upload all IPL documents.
    Gracefully skips if credentials are not configured.
    """
    if not SEARCH_ENDPOINT or not SEARCH_API_KEY:
        logger.warning(
            "Foundry IQ setup skipped: add AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY "
            "to .env to enable Foundry IQ agentic retrieval."
        )
        return

    logger.info("Foundry IQ: starting index setup...")
    create_index_if_not_exists()
    n = upload_documents(docs_path)
    logger.info("Foundry IQ: setup complete — %d documents indexed in '%s'.", n, INDEX_NAME)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from dotenv import load_dotenv
    load_dotenv(override=True)
    setup_foundry_iq()
