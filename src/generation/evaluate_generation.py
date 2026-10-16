import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.generation.llm_client import generate_answer
from src.generation.prompt_builder import build_prompt
from src.generation.citation_resolver import resolve_citations


QUESTIONS_PATH = Path("data/eval/questions.json")
CHUNKS_PATH = Path("data/processed/chunks.jsonl")
EMBEDDINGS_PATH = Path("data/processed/embeddings.jsonl")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

RRF_K = 60
TOP_K = 5


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
    embedding_model: SentenceTransformer,
    bm25: BM25Okapi,
) -> list[dict]:

    dense_indices = dense_ranking(
        query,
        chunks,
        embedding_model,
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

    print("Loading benchmark...")
    questions = load_questions(QUESTIONS_PATH)
    print(f"Loaded questions: {len(questions)}")

    print("Loading chunks...")
    chunks = load_jsonl(CHUNKS_PATH)
    print(f"Loaded chunks: {len(chunks)}")

    print("Loading embeddings...")
    embeddings = load_jsonl(EMBEDDINGS_PATH)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding["embedding"]

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print("Building BM25...")
    bm25 = build_bm25(chunks)

    total_citations = 0
    valid_citations = 0
    expected_section_citations = 0

    answer_results = []

    print("\nRunning generation evaluation...\n")

    for question in questions:

        qid = question["id"]
        text = question["question"]

        expected_sections = {
            str(section)
            for section in question["expected_sections"]
        }

        print(f"Evaluating {qid}...")

        retrieved_chunks = retrieve(
            text,
            chunks,
            embedding_model,
            bm25,
        )

        prompt = build_prompt(
            text,
            retrieved_chunks,
        )

        answer = generate_answer(prompt)

        _, citations = resolve_citations(
            answer,
            retrieved_chunks,
        )

        # Extract every citation token from the answer.
        citation_matches = re.findall(
            r"\[\s*S(\d+)\s*\]",
            answer,
        )

        total_citations += len(citation_matches)

        # A citation is valid if its source exists in retrieved_chunks.
        for citation_number in citation_matches:
            source_number = int(citation_number)

            if 1 <= source_number <= len(retrieved_chunks):
                valid_citations += 1

        # Check whether at least one cited section matches
        # one of the expected benchmark sections.
        cited_sections = {
            str(citation["section_number"])
            for citation in citations
        }

        if expected_sections.intersection(cited_sections):
            expected_section_citations += 1

        answer_results.append(
            {
                "id": qid,
                "question": text,
                "expected_sections": list(expected_sections),
                "retrieved_sections": [
                    str(chunk["section_number"])
                    for chunk in retrieved_chunks
                ],
                "answer": answer,
                "citations": citations,
                "cited_sections": list(cited_sections),
            }
        )

        print(
            f"  cited expected section: "
            f"{'YES' if expected_sections.intersection(cited_sections) else 'NO'}"
        )

    citation_validity = (
        valid_citations / total_citations
        if total_citations
        else 0.0
    )

    citation_coverage = (
        expected_section_citations / len(questions)
    )

    output_path = Path(
        "data/eval/generation_results.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            answer_results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 60)
    print("GENERATION EVALUATION")
    print("=" * 60)

    print(f"Questions              : {len(questions)}")
    print(f"Total citations        : {total_citations}")
    print(f"Valid citations        : {valid_citations}")
    print(f"Citation validity      : {citation_validity:.4f}")
    print(
        f"Citation coverage      : {citation_coverage:.4f}"
    )

    print(
        f"\nSaved detailed results to: {output_path}"
    )


if __name__ == "__main__":
    main()