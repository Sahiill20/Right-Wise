"""Evaluate Pinecone dense search + local BM25 + RRF on the existing benchmark.

Run after indexing:
    python -m src.retrieval.evaluate_hybrid_pinecone

This script forces DENSE_BACKEND=pinecone for the evaluation process only; it
does not rewrite .env. It evaluates the same 20 expected-section questions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
os.environ["DENSE_BACKEND"] = "pinecone"

from src.rag.rag_service import RAGService  # noqa: E402

QUESTIONS_PATH = Path("data/eval/questions.json")
TOP_K = 10


def load_questions(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation questions not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    questions = load_questions(QUESTIONS_PATH)
    print(f"Loaded evaluation questions: {len(questions)}")
    print("Dense backend: PineconeVectorStore")
    print("Lexical backend: local BM25")
    print("Fusion: RRF (k=60)")

    service = RAGService()

    recall_at_5_hits = 0
    recall_at_10_hits = 0
    reciprocal_ranks = []

    print("\nRunning Pinecone-backed hybrid evaluation...\n")

    for question in questions:
        qid = question["id"]
        query = question["question"]
        expected_sections = {
            str(section) for section in question["expected_sections"]
        }

        results = service.retrieve(query, top_k=TOP_K)
        returned_sections = [
            str(chunk.get("section_number", ""))
            for chunk in results
        ]

        hit_at_5 = bool(expected_sections.intersection(returned_sections[:5]))
        hit_at_10 = bool(expected_sections.intersection(returned_sections[:10]))
        recall_at_5_hits += int(hit_at_5)
        recall_at_10_hits += int(hit_at_10)

        first_rank = None
        for rank, section in enumerate(returned_sections, start=1):
            if section in expected_sections:
                first_rank = rank
                break

        reciprocal_ranks.append(
            0.0 if first_rank is None else 1.0 / first_rank
        )

        print(
            f"{qid} | R@5={'YES' if hit_at_5 else 'NO'} | "
            f"R@10={'YES' if hit_at_10 else 'NO'} | rank={first_rank}"
        )

    total = len(questions)
    recall_at_5 = recall_at_5_hits / total
    recall_at_10 = recall_at_10_hits / total
    mrr = sum(reciprocal_ranks) / total

    print("\n" + "=" * 60)
    print("PINECONE + BM25 + RRF EVALUATION")
    print("=" * 60)
    print(f"Questions   : {total}")
    print(f"Recall@5    : {recall_at_5:.4f}")
    print(f"Recall@10   : {recall_at_10:.4f}")
    print(f"MRR         : {mrr:.4f}")

    print("\nReference NumPy baseline:")
    print("Recall@5    : 0.9500")
    print("Recall@10   : 1.0000")
    print("MRR         : 0.8338")
    print(
        "Note: compare these results on the same corpus and expected-section "
        "benchmark. Pinecone results include the embedding/indexing path used "
        "by this version of the application."
    )


if __name__ == "__main__":
    main()
