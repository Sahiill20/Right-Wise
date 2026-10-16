import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


QUESTIONS_PATH = Path("data/eval/questions.json")
EMBEDDINGS_PATH = Path("data/processed/embeddings.jsonl")

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_questions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_embeddings(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def retrieve(
    question: str,
    chunks: list[dict],
    model: SentenceTransformer,
    top_k: int = 10,
) -> list[dict]:

    # Convert question into a 384-dimensional vector.
    query_embedding = model.encode(
        question,
        normalize_embeddings=True,
    )

    chunk_embeddings = np.array(
        [chunk["embedding"] for chunk in chunks],
        dtype=np.float32,
    )

    # Cosine similarity because vectors are normalized.
    scores = chunk_embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        result = chunks[index].copy()
        result["score"] = float(scores[index])
        results.append(result)

    return results


def calculate_mrr(ranks: list[int | None]) -> float:
    reciprocal_ranks = []

    for rank in ranks:
        if rank is None:
            reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(1.0 / rank)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def main() -> None:
    print("Loading evaluation questions...")
    questions = load_questions(QUESTIONS_PATH)
    print(f"Loaded questions: {len(questions)}")

    print("Loading embeddings...")
    chunks = load_embeddings(EMBEDDINGS_PATH)
    print(f"Loaded embeddings: {len(chunks)}")

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    recall_at_5_hits = 0
    recall_at_10_hits = 0

    reciprocal_ranks = []

    print("\nRunning evaluation...\n")

    for question in questions:

        qid = question["id"]
        text = question["question"]

        expected_sections = set(
            str(section)
            for section in question["expected_sections"]
        )

        results = retrieve(
            question=text,
            chunks=chunks,
            model=model,
            top_k=10,
        )

        returned_sections = [
            str(result["section_number"])
            for result in results
        ]

        # Recall@5
        top_5_sections = set(returned_sections[:5])

        hit_at_5 = bool(
            expected_sections.intersection(top_5_sections)
        )

        if hit_at_5:
            recall_at_5_hits += 1

        # Recall@10
        top_10_sections = set(returned_sections[:10])

        hit_at_10 = bool(
            expected_sections.intersection(top_10_sections)
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
    print("DENSE RETRIEVAL EVALUATION")
    print("=" * 60)

    print(f"Questions   : {total}")
    print(f"Recall@5    : {recall_at_5:.4f}")
    print(f"Recall@10   : {recall_at_10:.4f}")
    print(f"MRR         : {mrr:.4f}")


if __name__ == "__main__":
    main()