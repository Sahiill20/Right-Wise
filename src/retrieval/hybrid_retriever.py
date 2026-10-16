import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


CHUNKS_PATH = Path("data/processed/chunks.jsonl")
EMBEDDINGS_PATH = Path("data/processed/embeddings.jsonl")

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Standard RRF constant.
RRF_K = 60


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def build_bm25(chunks: list[dict]) -> BM25Okapi:
    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_chunks)


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
    chunks: list[dict],
    bm25: BM25Okapi,
) -> list[int]:

    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    return list(np.argsort(scores)[::-1])


def rrf_fusion(
    dense_indices: list[int],
    bm25_indices: list[int],
    top_k: int = 5,
) -> list[dict]:

    rrf_scores: dict[int, float] = {}

    # Add dense ranking contribution.
    for rank, index in enumerate(dense_indices, start=1):
        score = 1.0 / (RRF_K + rank)
        rrf_scores[index] = rrf_scores.get(index, 0.0) + score

    # Add BM25 ranking contribution.
    for rank, index in enumerate(bm25_indices, start=1):
        score = 1.0 / (RRF_K + rank)
        rrf_scores[index] = rrf_scores.get(index, 0.0) + score

    # Sort indices by combined RRF score.
    ranked_indices = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True,
    )[:top_k]

    return [
        {
            "index": index,
            "score": rrf_scores[index],
        }
        for index in ranked_indices
    ]


def retrieve(
    query: str,
    chunks: list[dict],
    model: SentenceTransformer,
    bm25: BM25Okapi,
    top_k: int = 5,
) -> list[dict]:

    dense_indices = dense_ranking(
        query,
        chunks,
        model,
    )

    bm25_indices = bm25_ranking(
        query,
        chunks,
        bm25,
    )

    fused_results = rrf_fusion(
        dense_indices,
        bm25_indices,
        top_k=top_k,
    )

    results = []

    for item in fused_results:
        result = chunks[item["index"]].copy()
        result["score"] = item["score"]
        results.append(result)

    return results


def main() -> None:
    print(f"Loading chunks from: {CHUNKS_PATH}")
    chunks = load_jsonl(CHUNKS_PATH)

    print(f"Loading embeddings from: {EMBEDDINGS_PATH}")
    embeddings = load_jsonl(EMBEDDINGS_PATH)

    # Our embeddings file and chunks file have the same order.
    # Add the embedding to the corresponding chunk.
    for chunk, embedding_record in zip(chunks, embeddings):
        chunk["embedding"] = embedding_record["embedding"]

    print(f"Loaded chunks: {len(chunks)}")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Building BM25 index...")
    bm25 = build_bm25(chunks)

    while True:
        query = input("\nEnter your query (or 'exit'): ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        results = retrieve(
            query=query,
            chunks=chunks,
            model=model,
            bm25=bm25,
            top_k=5,
        )

        print("\nHybrid Top results:")
        print("=" * 80)

        for rank, result in enumerate(results, start=1):
            print(f"\nRank       : {rank}")
            print(f"Chunk ID   : {result['chunk_id']}")
            print(f"Section    : {result['section_number']}")
            print(f"RRF Score  : {result['score']:.6f}")
            print(f"Text       : {result['text'][:500]}")


if __name__ == "__main__":
    main()