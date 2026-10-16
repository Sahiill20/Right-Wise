import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


CHUNKS_PATH = Path("data/processed/chunks.jsonl")
EMBEDDINGS_PATH = Path("data/processed/embeddings.jsonl")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RRF_K = 60


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def build_bm25(chunks):
    tokenized = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized)


def dense_ranking(query, chunks, model):
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


def bm25_ranking(query, bm25):
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    return list(np.argsort(scores)[::-1])


def rrf_fusion(dense_indices, bm25_indices, top_k=20):
    """
    Produce a candidate pool using Reciprocal Rank Fusion.
    We take 20 candidates instead of immediately taking 5,
    giving the reranker enough material to work with.
    """

    scores = {}

    for rank, index in enumerate(dense_indices, start=1):
        scores[index] = (
            scores.get(index, 0.0)
            + 1 / (RRF_K + rank)
        )

    for rank, index in enumerate(bm25_indices, start=1):
        scores[index] = (
            scores.get(index, 0.0)
            + 1 / (RRF_K + rank)
        )

    ranked = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    return ranked[:top_k]


def main():
    chunks = load_jsonl(CHUNKS_PATH)
    embeddings = load_jsonl(EMBEDDINGS_PATH)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding["embedding"]

    print(f"Loaded chunks: {len(chunks)}")

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print("Building BM25...")
    bm25 = build_bm25(chunks)

    print("Loading cross-encoder...")
    reranker = CrossEncoder(RERANKER_MODEL)

    query = input("\nEnter query: ").strip()

    # -------- candidate generation --------

    dense_indices = dense_ranking(
        query,
        chunks,
        embedding_model,
    )

    bm25_indices = bm25_ranking(
        query,
        bm25,
    )

    candidate_indices = rrf_fusion(
        dense_indices,
        bm25_indices,
        top_k=20,
    )

    candidates = [
        chunks[index]
        for index in candidate_indices
    ]

    print(f"\nCandidate pool: {len(candidates)}")

    # -------- reranking --------

    pairs = [
        (query, candidate["text"])
        for candidate in candidates
    ]

    scores = reranker.predict(pairs)

    reranked = []

    for candidate, score in zip(candidates, scores):
        result = candidate.copy()
        result["reranker_score"] = float(score)
        reranked.append(result)

    reranked.sort(
        key=lambda x: x["reranker_score"],
        reverse=True,
    )

    print("\nReranked results:")
    print("=" * 80)

    for rank, result in enumerate(
        reranked[:5],
        start=1,
    ):
        print(f"\nRank       : {rank}")
        print(f"Chunk ID   : {result['chunk_id']}")
        print(f"Section    : {result['section_number']}")
        print(f"Score      : {result['reranker_score']:.4f}")
        print(f"Text       : {result['text'][:500]}")


if __name__ == "__main__":
    main()