import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


QUESTIONS_PATH = Path("data/eval/questions.json")
CHUNKS_PATH = Path("data/processed/chunks.jsonl")
EMBEDDINGS_PATH = Path("data/processed/embeddings.jsonl")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RRF_K = 60
CANDIDATE_K = 20
TOP_K = 10


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_questions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def build_bm25(chunks: list[dict]) -> BM25Okapi:
    tokenized = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]
    return BM25Okapi(tokenized)


def dense_ranking(
    query: str,
    chunks: list[dict],
    model: SentenceTransformer,
) -> list[int]:

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    chunk_embeddings = np.array(
        [chunk["embedding"] for chunk in chunks],
        dtype=np.float32,
    )

    scores = chunk_embeddings @ query_embedding

    return list(np.argsort(scores)[::-1])


def bm25_ranking(
    query: str,
    bm25: BM25Okapi,
) -> list[int]:

    scores = bm25.get_scores(tokenize(query))

    return list(np.argsort(scores)[::-1])


def rrf_fusion(
    dense_indices: list[int],
    bm25_indices: list[int],
    top_k: int,
) -> list[int]:

    scores = {}

    for rank, index in enumerate(dense_indices, start=1):
        scores[index] = (
            scores.get(index, 0.0)
            + 1.0 / (RRF_K + rank)
        )

    for rank, index in enumerate(bm25_indices, start=1):
        scores[index] = (
            scores.get(index, 0.0)
            + 1.0 / (RRF_K + rank)
        )

    ranked = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    return ranked[:top_k]


def rerank(
    query: str,
    candidate_indices: list[int],
    chunks: list[dict],
    model: CrossEncoder,
) -> list[int]:

    pairs = [
        (query, chunks[index]["text"])
        for index in candidate_indices
    ]

    scores = model.predict(pairs)

    scored = list(
        zip(candidate_indices, scores)
    )

    scored.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        index
        for index, _ in scored
    ]


def main() -> None:

    print("Loading evaluation questions...")
    questions = load_questions(QUESTIONS_PATH)
    print(f"Loaded questions: {len(questions)}")

    print("Loading chunks...")
    chunks = load_jsonl(CHUNKS_PATH)
    print(f"Loaded chunks: {len(chunks)}")

    print("Loading embeddings...")
    embeddings = load_jsonl(EMBEDDINGS_PATH)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding["embedding"]

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print("Building BM25 index...")
    bm25 = build_bm25(chunks)

    print(f"Loading reranker: {RERANKER_MODEL}")
    reranker = CrossEncoder(RERANKER_MODEL)

    recall_at_5_hits = 0
    recall_at_10_hits = 0
    reciprocal_ranks = []

    print("\nRunning hybrid + reranker evaluation...\n")

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
            embedding_model,
        )

        bm25_indices = bm25_ranking(
            text,
            bm25,
        )

        candidate_indices = rrf_fusion(
            dense_indices,
            bm25_indices,
            top_k=CANDIDATE_K,
        )

        reranked_indices = rerank(
            text,
            candidate_indices,
            chunks,
            reranker,
        )

        top_indices = reranked_indices[:TOP_K]

        returned_sections = [
            str(chunks[index]["section_number"])
            for index in top_indices
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

        for i, section in enumerate(
            returned_sections,
            start=1,
        ):
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
    print("HYBRID + RERANKER EVALUATION")
    print("=" * 60)

    print(f"Questions   : {total}")
    print(f"Recall@5    : {recall_at_5:.4f}")
    print(f"Recall@10   : {recall_at_10:.4f}")
    print(f"MRR         : {mrr:.4f}")


if __name__ == "__main__":
    main()