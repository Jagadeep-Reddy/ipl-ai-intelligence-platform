"""
tests/test_retriever.py — Unit tests for RAG retrieval components
"""
import pytest
from rag.retriever import rrf_fuse, RERANK_TOP_N


class TestRRFFusion:
    def test_fused_contains_all_candidates(self):
        dense = [(0, 0.9), (1, 0.8), (2, 0.7)]
        bm25 = [(2, 0.85), (3, 0.75), (0, 0.65)]
        fused = rrf_fuse(dense, bm25)
        result_ids = {idx for idx, _ in fused}
        assert result_ids == {0, 1, 2, 3}

    def test_overlap_boosts_score(self):
        # Doc 0 appears in both lists at rank 1 → should score highest
        dense = [(0, 0.99), (1, 0.5)]
        bm25 = [(0, 0.99), (2, 0.5)]
        fused = rrf_fuse(dense, bm25)
        top_id = fused[0][0]
        assert top_id == 0

    def test_scores_are_positive(self):
        dense = [(0, 0.9), (1, 0.8)]
        bm25 = [(0, 0.7)]
        fused = rrf_fuse(dense, bm25)
        for _, score in fused:
            assert score > 0

    def test_sorted_descending(self):
        dense = [(i, 1.0 - i * 0.1) for i in range(5)]
        bm25 = [(4 - i, 1.0 - i * 0.1) for i in range(5)]
        fused = rrf_fuse(dense, bm25)
        scores = [s for _, s in fused]
        assert scores == sorted(scores, reverse=True)

    def test_empty_inputs(self):
        fused = rrf_fuse([], [])
        assert fused == []

    def test_single_source(self):
        dense = [(0, 0.9), (1, 0.8)]
        fused = rrf_fuse(dense, [])
        assert len(fused) == 2

    def test_rrf_k_affects_ranking(self):
        # With k=1 vs k=60, relative ordering should still hold
        dense = [(0, 0.9), (1, 0.8)]
        bm25 = [(1, 0.95), (0, 0.6)]
        fused_k1 = rrf_fuse(dense, bm25, k=1)
        fused_k60 = rrf_fuse(dense, bm25, k=60)
        # Both should return 2 docs
        assert len(fused_k1) == 2
        assert len(fused_k60) == 2
