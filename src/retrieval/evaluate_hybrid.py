import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


QUESTIONS_PATH = Path("data/eval/questions.json")
CHUNKS_PATH = Path("data/processed/chunks.jsonl")
EMBEDDINGS_PATH = Path("data/processed/embeddings.jsonl")

MODEL_NAME = "BAAI/bge-small-en-v1.5"
RRF_K = 60


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_questions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def build_bm25(chunks: list[dict]) -> BM25Okapi:
    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_chunks)


def dense_ranking(
    question: str,
    chunks: list[dict],
    model: SentenceTransformer,
) -> list[int]:

    query_embedding = model.encode(
        question,
        normalize_embeddings=True,
    )

    chunk_embeddings = np.array(
        [chunk["embedding"] for chunk in chunks],
        dtype=np.float32,
    )

    scores = chunk_embeddings @ query_embedding

    return list(np.argsort(scores)[::-1])


def bm25_ranking(
    question: str,
    bm25: BM25Okapi,
) -> list[int]:

    query_tokens = tokenize(question)

    scores = bm25.get_scores(query_tokens)

    return list(np.argsort(scores)[::-1])


def rrf_fusion(
    dense_indices: list[int],
    bm25_indices: list[int],
    top_k: int = 10,
) -> list[int]:

    rrf_scores = {}

    for rank, index in enumerate(dense_indices, start=1):
        rrf_scores[index] = (
            rrf_scores.get(index, 0.0)
            + 1.0 / (RRF_K + rank)
        )

    for rank, index in enumerate(bm25_indices, start=1):
        rrf_scores[index] = (
            rrf_scores.get(index, 0.0)
            + 1.0 / (RRF_K + rank)
        )

    ranked_indices = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True,
    )

    return ranked_indices[:top_k]


def main() -> None:

    print("Loading evaluation questions...")
    questions = load_questions(QUESTIONS_PATH)
    print(f"Loaded questions: {len(questions)}")

    print("Loading chunks...")
    chunks = load_jsonl(CHUNKS_PATH)
    print(f"Loaded chunks: {len(chunks)}")

    print("Loading embeddings...")
    embeddings = load_jsonl(EMBEDDINGS_PATH)

    for chunk, embedding_record in zip(chunks, embeddings):
        chunk["embedding"] = embedding_record["embedding"]

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Building BM25 index...")
    bm25 = build_bm25(chunks)

    recall_at_5_hits = 0
    recall_at_10_hits = 0
    reciprocal_ranks = []

    print("\nRunning hybrid evaluation...\n")

    for question in questions:

        qid = question["id"]
        text = question["question"]

        expected_sections = {
            str(section)
            for section in question["expected_sections"]
        }

        dense_indices = dense_ranking(
            text,
            chunks,
            model,
        )

        bm25_indices = bm25_ranking(
            text,
            bm25,
        )

        hybrid_indices = rrf_fusion(
            dense_indices,
            bm25_indices,
            top_k=10,
        )

        returned_sections = [
            str(chunks[index]["section_number"])
            for index in hybrid_indices
        ]

        # Recall@5
        hit_at_5 = bool(
            expected_sections.intersection(
                set(returned_sections[:5])
            )
        )

        if hit_at_5:
            recall_at_5_hits += 1

        # Recall@10
        hit_at_10 = bool(
            expected_sections.intersection(
                set(returned_sections[:10])
            )
        )

        if hit_at_10:
            recall_at_10_hits += 1

        # MRR
        rank = None

        for i, section in enumerate(returned_sections, start=1):
            if section in expected_sections:
                rank = i
                break

        reciprocal_ranks.append(
            0.0 if rank is None else 1.0 / rank
        )

        print(
            f"{qid} | "
            f"R@5={'YES' if hit_at_5 else 'NO'} | "
            f"R@10={'YES' if hit_at_10 else 'NO'} | "
            f"rank={rank}"
        )

    total = len(questions)

    recall_at_5 = recall_at_5_hits / total
    recall_at_10 = recall_at_10_hits / total
    mrr = sum(reciprocal_ranks) / total

    print("\n" + "=" * 60)
    print("HYBRID RETRIEVAL EVALUATION")
    print("=" * 60)

    print(f"Questions   : {total}")
    print(f"Recall@5    : {recall_at_5:.4f}")
    print(f"Recall@10   : {recall_at_10:.4f}")
    print(f"MRR         : {mrr:.4f}")


if __name__ == "__main__":
    main()