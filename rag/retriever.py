"""
rag/retriever.py
────────────────
Phase 1 Day 4 – Hybrid RAG retrieval pipeline.
  • BGE-M3 dense embeddings stored in Qdrant (INT8, HNSW m=16)
  • BM25 sparse index via rank-bm25
  • Reciprocal Rank Fusion (k=60)
  • Cross-encoder reranker (ms-marco-MiniLM-L-6-v2) → top 8
"""

from __future__ import annotations

import json
import logging
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    SearchRequest,
    VectorParams,
)

logger = logging.getLogger(__name__)

COLLECTION_NAME = "ipl_knowledge"
VECTOR_DIM = 384
HNSW_M = 16
HNSW_EF_CONSTRUCT = 200
BM25_CANDIDATES = 80
DENSE_CANDIDATES = 80
RRF_K = 60
RERANK_TOP_N = 12


# ── Models (loaded once at startup) ──────────────────────────────────────────

class Models:
    _dense: SentenceTransformer | None = None
    _reranker: CrossEncoder | None = None

    @classmethod
    def dense(cls) -> SentenceTransformer:
        if cls._dense is None:
            logger.info("Loading BGE-M3 …")
            cls._dense = SentenceTransformer("BAAI/bge-small-en")
        return cls._dense

    @classmethod
    def reranker(cls) -> CrossEncoder:
        if cls._reranker is None:
            logger.info("Loading cross-encoder …")
            cls._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return cls._reranker


# ── Qdrant helpers ────────────────────────────────────────────────────────────

def get_qdrant_client(url: str, api_key: str | None = None) -> QdrantClient:
    return QdrantClient(url=url, api_key=api_key or None, prefer_grpc=False, timeout=120)


def create_collection(client: QdrantClient) -> None:
    """Create Qdrant collection with INT8 quantization if it doesn't exist."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        logger.info("Collection '%s' already exists.", COLLECTION_NAME)
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                always_ram=True,
            )
        ),
        hnsw_config={"m": HNSW_M, "ef_construct": HNSW_EF_CONSTRUCT},
    )
    logger.info("Collection '%s' created.", COLLECTION_NAME)


# ── Index builder ─────────────────────────────────────────────────────────────

def build_index(
    docs_path: Path,
    qdrant_url: str,
    bm25_path: Path = Path("rag/bm25_index.pkl"),
    qdrant_api_key: str | None = None,
    batch_size: int = 32,
) -> None:
    """Embed all documents and upsert into Qdrant; pickle the BM25 index."""
    docs = json.loads(docs_path.read_text())
    texts = [d["text"] for d in docs]

    logger.info("Embedding %d documents with BGE-M3 …", len(docs))
    encoder = Models.dense()
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    client = get_qdrant_client(qdrant_url, qdrant_api_key)
    create_collection(client)

    points = [
        PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={k: v for k, v in docs[i].items() if k != "text"}
            | {"text": texts[i]},
        )
        for i in range(len(docs))
    ]

    # Upsert in batches
    for start in range(0, len(points), 100):
        client.upsert(COLLECTION_NAME, points[start : start + 100])
    logger.info("Upserted %d points into Qdrant.", len(points))

    # Build BM25 on tokenised texts
    tokenised = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenised)
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "texts": texts, "docs": docs}, f)
    logger.info("BM25 index saved to %s", bm25_path)


# ── RRF ──────────────────────────────────────────────────────────────────────

def rrf_fuse(
    dense_hits: list[tuple[int, float]],   # (doc_idx, score)
    bm25_hits: list[tuple[int, float]],
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion – returns sorted list of (doc_idx, rrf_score)."""
    scores: dict[int, float] = {}
    for rank, (idx, _) in enumerate(dense_hits):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, (idx, _) in enumerate(bm25_hits):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Main retrieval pipeline ───────────────────────────────────────────────────

class IPLRetriever:
    def __init__(
        self,
        qdrant_url: str,
        bm25_path: Path = Path("rag/bm25_index.pkl"),
        qdrant_api_key: str | None = None,
    ):
        self.qdrant = get_qdrant_client(qdrant_url, qdrant_api_key)
        with open(bm25_path, "rb") as f:
            idx = pickle.load(f)
        self.bm25: BM25Okapi = idx["bm25"]
        self.texts: list[str] = idx["texts"]
        self.docs: list[dict] = idx["docs"]
        logger.info("IPLRetriever ready. %d docs in BM25.", len(self.texts))

    def _dense_search(
        self, query: str, top_k: int = DENSE_CANDIDATES, filters: dict | None = None
    ) -> list[tuple[int, float]]:
        vec = Models.dense().encode(query, normalize_embeddings=True).tolist()
        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
                if v is not None
            ]
            if conditions:
                qdrant_filter = Filter(must=conditions)

        results = self.qdrant.search(
            COLLECTION_NAME, query_vector=vec, limit=top_k, query_filter=qdrant_filter
        )
        return [(r.id, r.score) for r in results]

    def _bm25_search(self, query: str, top_k: int = BM25_CANDIDATES) -> list[tuple[int, float]]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return candidates
        pairs = [[query, c["text"]] for c in candidates]
        ce_scores = Models.reranker().predict(pairs)
        ranked = sorted(
            zip(candidates, ce_scores), key=lambda x: x[1], reverse=True
        )
        return [doc for doc, _ in ranked[:RERANK_TOP_N]]

    def retrieve(
        self,
        query: str,
        filters: dict | None = None,
        rerank: bool = True,
    ) -> list[dict]:
        """
        Full hybrid retrieval:
          Dense search → BM25 search → RRF fusion → cross-encoder rerank
        Returns top RERANK_TOP_N document dicts.
        """
        dense_hits = self._dense_search(query, filters=filters)
        bm25_hits = self._bm25_search(query)

        fused = rrf_fuse(dense_hits, bm25_hits)[:DENSE_CANDIDATES]
        candidates = [self.docs[idx] for idx, _ in fused if idx < len(self.docs)]

        if rerank and len(candidates) > 1:
            candidates = self._rerank(query, candidates)

        return candidates


# ── Live context retrieval (used in WebSocket pipeline) ──────────────────────

class LiveContextRetriever:
    """Lightweight retriever for real-time per-ball context (<80ms budget)."""

    def __init__(self, retriever: IPLRetriever):
        self.retriever = retriever

    def fetch_matchup_context(self, batter: str, bowler: str) -> list[dict]:
        query = f"{batter} vs {bowler} head to head performance"
        return self.retriever.retrieve(
            query,
            filters={"doc_type": "player_profile"},
            rerank=False,  # skip reranker to save ~60ms
        )[:3]


if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)
    build_index(
        docs_path=Path("rag/documents.json"),
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
    )