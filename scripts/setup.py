#!/usr/bin/env python3
"""
scripts/setup.py
────────────────
Master orchestration script – run once to bootstrap the entire platform.
Executes all 5 phases in sequence. Safe to rerun (idempotent).

Usage:
  python scripts/setup.py [--skip-download] [--skip-models] [--skip-docs]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("setup")


def phase(name: str):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            logger.info("=" * 60)
            logger.info("▶  %s", name)
            logger.info("=" * 60)
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            logger.info("✅ %s completed in %.1fs", name, time.perf_counter() - t0)
            return result
        return wrapper
    return decorator


@phase("Phase 1a – Data Ingestion & Intent Labelling")
def run_ingestion(download: bool = False):
    from data.ingest import build_pipeline
    build_pipeline(download=download)


@phase("Phase 1b – PostgreSQL Structured Database")
def run_database():
    from data.precompute import run_all
    run_all(os.environ["DATABASE_URL"])


@phase("Phase 1c – RAG Document Generation")
def run_doc_generation():
    from rag.doc_generator import generate_all
    docs = generate_all(
        db_url=os.environ["DATABASE_URL"],
        out_path=Path("rag/documents.json"),
    )
    logger.info("Generated %d documents.", len(docs))


@phase("Phase 1d – Embedding & Vector Index Build")
def run_indexing():
    from rag.retriever import build_index
    build_index(
        docs_path=Path("rag/documents.json"),
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=os.environ.get("QDRANT_API_KEY"),
    )


@phase("Phase 2 – Train Win Probability Model (Model A)")
def run_win_prob_training():
    from ml.train_win_prob import train_win_probability_model
    result = train_win_probability_model(
        db_url=os.environ["DATABASE_URL"],
        n_optuna_trials=100,
    )
    logger.info("Model A result: %s", result)


@phase("Phase 3 – Train Intent Classifier (Model B)")
def run_intent_training():
    from ml.train_intent import train_intent_classifier
    result = train_intent_classifier(
        deliveries_path=Path("data/processed/deliveries_labelled.csv"),
        matches_path=Path("data/processed/matches_clean.csv"),
        n_optuna_trials=50,
    )
    logger.info("Model B result: %s", result)


@phase("Phase 4 – RAGAS Baseline Evaluation")
def run_ragas():
    from eval.ragas_eval import evaluate, ci_check
    scores = evaluate(qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    ci_check(scores)


@phase("Phase 1d – Foundry IQ Indexing (Azure AI Search)")
def run_foundry_iq():
    from agents.foundry_iq import setup_foundry_iq
    from pathlib import Path
    setup_foundry_iq(docs_path=Path("rag/documents.json"))


def main():
    parser = argparse.ArgumentParser(description="IPL AI Platform Setup")
    parser.add_argument("--skip-download", action="store_true", help="Skip Kaggle download")
    parser.add_argument("--skip-models", action="store_true", help="Skip ML model training")
    parser.add_argument("--skip-docs", action="store_true", help="Skip RAG doc generation")
    parser.add_argument("--skip-eval", action="store_true", help="Skip RAGAS evaluation")
    parser.add_argument("--skip-foundry-iq", action="store_true", help="Skip Foundry IQ indexing")
    args = parser.parse_args()

    required_env = ["DATABASE_URL"]
    missing = [k for k in required_env if not os.environ.get(k)]
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("AZURE_OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY or AZURE_OPENAI_API_KEY")
    if missing:
        logger.error("Missing required environment variables: %s", missing)
        logger.error("Copy .env.example to .env and fill in values.")
        sys.exit(1)

    # Phase 1: Data
    run_ingestion(download=not args.skip_download)
    run_database()

    if not args.skip_docs:
        run_doc_generation()

    run_indexing()

    # Phase 1d: Foundry IQ indexing (optional – requires AZURE_SEARCH_* vars)
    if not args.skip_foundry_iq:
        run_foundry_iq()

    # Phase 2-3: Models
    if not args.skip_models:
        run_win_prob_training()
        run_intent_training()

    # Phase 4: Evaluation
    if not args.skip_eval:
        run_ragas()

    logger.info("=" * 60)
    logger.info("🎉 Platform setup complete!")
    logger.info("Start the stack: docker-compose up")
    logger.info("Or local:  uvicorn api.main:app --reload")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()