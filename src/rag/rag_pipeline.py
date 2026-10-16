import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.generation.llm_client import generate_answer
from src.generation.prompt_builder import build_prompt
from src.generation.citation_resolver import resolve_citations


CHUNKS_PATH = Path("data/processed/chunks.jsonl")
EMBEDDINGS_PATH = Path("data/processed/embeddings.jsonl")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

RRF_K = 60
TOP_K = 5


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

    ranked_indices = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    return ranked_indices[:top_k]


def retrieve(
    query: str,
    chunks: list[dict],
    model: SentenceTransformer,
    bm25: BM25Okapi,
) -> list[dict]:

    dense_indices = dense_ranking(
        query,
        chunks,
        model,
    )

    bm25_indices = bm25_ranking(
        query,
        bm25,
    )

    hybrid_indices = rrf_fusion(
        dense_indices,
        bm25_indices,
        top_k=TOP_K,
    )

    return [
        chunks[index]
        for index in hybrid_indices
    ]


def main() -> None:

    print("Loading chunks...")
    chunks = load_jsonl(CHUNKS_PATH)

    print("Loading embeddings...")
    embeddings = load_jsonl(EMBEDDINGS_PATH)

    # Attach embeddings to the corresponding chunks.
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding["embedding"]

    print(f"Loaded {len(chunks)} chunks.")

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print("Building BM25 index...")
    bm25 = build_bm25(chunks)

    print("\n RightWise RAG is ready.")

    while True:

        query = input(
            "\nAsk a consumer-rights question (or 'exit'): "
        ).strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        print("\nRetrieving relevant legal sections...")

        retrieved_chunks = retrieve(
            query,
            chunks,
            embedding_model,
            bm25,
        )

        print("\nRetrieved sections:")

        for chunk in retrieved_chunks:
            print(
                f"- Section {chunk['section_number']} "
                f"({chunk['chunk_id']})"
            )

        print("\nBuilding augmented prompt...")

        prompt = build_prompt(
            query,
            retrieved_chunks,
        )

        print("Generating answer...")

        answer = generate_answer(prompt)

        _, citations = resolve_citations(
            answer,
            retrieved_chunks,
        )

        print("\n" + "=" * 80)
        print("RIGHTWISE")
        print("=" * 80)
        print(answer)

        print("\nSources:")
        for citation in citations:
            print(
                f"- Section {citation['section_number']}: "
                f"{citation['section_title']} "
                f"(pages {citation['page_start']}-{citation['page_end']})"
            )


if __name__ == "__main__":
    main()